import os
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import torch
import icevision.all as ia
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import train_utils, train_utils_img
from training.train_utils import GPUManager
from segmentation import mask_utils, segmentation_utils as segm_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP
from object_detection import object_detection_utils as obj_utils


class ImageObjectDetectionTrainer(train_utils_img.ImageTrainer):
    @staticmethod
    def get_argparser(desc="Fastai image obj detec trainer arguments", pdef=dict(), phelp=dict()):
        parser = super(ImageObjectDetectionTrainer, ImageObjectDetectionTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument("--backbone", type=str, help="backbone")
        parser.add_argument('--ious', default=[.1, .3, .5], nargs='+', type=float, help="Metrics IoUs")
        parser.add_argument('--with-segm', action='store_true', help="Metrics will be also computed on segmentation")
        return parser

    @staticmethod
    def prepare_training(args):
        if args.with_segm: assert args.model == 'mask_rcnn', f"Used --with-segm but requested model is {args.model}"
        args.exp_name = "img_obj_detec_" + args.exp_name
        super(ImageObjectDetectionTrainer, ImageObjectDetectionTrainer).prepare_training(args)

    def __init__(self, args, stratify=False, full_img_sep=CROP_SEP, **kwargs):
        self.SEGM_PERF = 'segm_'
        self.metrics_types = [ia.COCOMetricType.bbox] + ([ia.COCOMetricType.mask] if args.with_segm else [])
        self.scores_steps = np.array(range(10)) / 10
        super().__init__(args, stratify, full_img_sep, **kwargs)
        if self.args.encrypted:
            self.coco_parser_cls = obj_utils.COCOMaskParserEncrypted
            def _crypted_load(mixin_self):
                mixin_self.img = self.load_image_item(mixin_self.filepath)
                mixin_self.height, mixin_self.width, _ = mixin_self.img.shape
                super(ia.core.FilepathRecordMixin, mixin_self)._load()
            ia.core.FilepathRecordMixin._load = _crypted_load
            # TODO: hack, solve this better
        self.coco_parser_cls = obj_utils.COCOMaskParserEncrypted if self.args.encrypted else ia.parsers.COCOMaskParser

    def load_coco_anno_file(self, anno_file):
        anno_path, img_dir = obj_utils.get_obj_det_ds_paths(self.args.data, anno_file)
        coco_parser = self.coco_parser_cls(annotations_filepath=anno_path, img_dir=img_dir)
        idmap = ia.IDMap()
        records = fv.L(coco_parser.parse(data_splitter=ia.SingleSplitSplitter(), idmap=idmap, show_pbar=False)[0])
        return coco_parser, idmap, records

    def load_items(self, anno_file):
        _, _, records = self.load_coco_anno_file(anno_file)
        # dummy cls so that the split_data pipeline works
        return records, fv.L([1]*len(records))

    def get_patch_full_img(self, patch): return super().get_patch_full_img(str(patch.filepath))

    def get_cats_idxs(self):
        # index 0 is the bg class which is ignored by COCO metrics
        return list(range(1, len(self.args.cats)+1))

    def get_cats_with_all(self):
        # category at index 0 is the background class, which should be ignored
        return [self.ALL_CATS, *self.args.cats[1:]]

    def get_segm_cats_with_all(self):
        return super().get_cats_with_all()

    def get_cat_metric_name(self, perf_fn, cat, iou, mtype):
        return f'{super().get_cat_metric_name(perf_fn, cat)}_iou{iou}_{mtype.name}'

    def get_cat_segm_metric_name(self, perf_fn, cat, score):
        return f'{self.SEGM_PERF}{super().get_cat_metric_name(perf_fn, cat)}{"" if score is None else f"_score{score}"}'

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        def custom_coco_eval_metric(name, **kwargs):
            class CocoEvalTemplate(obj_utils.CustomCOCOMetric):
                def __init__(self): super().__init__(**kwargs)
            return type(name, (CocoEvalTemplate,), {})

        for mtype in self.metrics_types:
            for iou in self.args.ious:
                cls_name = self.get_cat_metric_name(perf_fn, cat, iou, mtype)
                cls = custom_coco_eval_metric(cls_name, cat_id=cat_id, iou=iou, metric=perf_fn, metric_type=mtype)
                metrics_fn[cls_name] = cls()

    def ordered_test_perfs_per_cats(self):
        ordered = []
        for perf_fn in self.args.metrics_fns:
            for mtype in self.metrics_types:
                for iou in self.args.ious:
                    mns = [self.get_cat_metric_name(perf_fn, cat, iou, mtype) for cat in self.get_cats_with_all()]
                    ordered.append((mns, f'{perf_fn}_iou{iou}_{mtype.name}'))
        if self.args.with_segm:
            for perf_fn in self.args.metrics_fns:
                mns = [self.get_cat_segm_metric_name(perf_fn, cat, None) for cat in self.get_segm_cats_with_all()]
                ordered.append((mns, f'{self.SEGM_PERF}{perf_fn}'))
                for score in self.scores_steps:
                    mns = [self.get_cat_segm_metric_name(perf_fn, cat, score) for cat in self.get_segm_cats_with_all()]
                    ordered.append((mns, f'{self.SEGM_PERF}{perf_fn}_score{score}'))
        return ordered

    def compute_conf_mat(self, targs, preds):
        if self.args.with_segm:
            return segm_utils.pixel_conf_mat(targs, preds, self.args.cats)
        else:
            raise NotImplementedError

    def compute_test_predictions(self, learn, test_name, test_items_with_cls):
        GPUManager.sync_distributed_process()
        (arch, _), vtfms = self.get_arch(), self.get_tfms()[1]
        test_ds = ia.Dataset(test_items_with_cls[0], vtfms)

        # OD preds for OD perfs, IGNORE test items without objects
        test_dl = arch.infer_dl(test_ds, batch_size=learn.dls.bs)
        with GPUManager.running_context(learn, self.args.gpu_ids):
            interp = SimpleNamespace()
            interp.od_targs, interp.od_preds = arch.predict_dl(model=learn.model, infer_dl=test_dl)

        if self.args.with_segm:
            print(f"Applying model on {test_name} pictures without annotations")
            # OD preds for SEGM perfs, INCLUDE test items without objects
            _, imdir = obj_utils.get_obj_det_ds_paths(self.args.data, test_name)
            parser, idmap, records = self.load_coco_anno_file(test_name)
            im_no_gt = [(i, imd['file_name']) for i, imd in parser._imageid2info.items() if i not in idmap.name2id]
            with GPUManager.running_context(learn, self.args.gpu_ids):
                od_preds_no_gt, empty_gt = [], None
                for batch in common.batch_list(im_no_gt, bs=learn.dls.bs):
                    ims = [self.load_image_item(os.path.join(imdir, im_name)) for _, im_name in batch]
                    batch_dl, samples = arch.build_infer_batch(ia.Dataset.from_images(ims, vtfms))
                    if empty_gt is None: empty_gt = np.zeros((samples[0]['height'], samples[0]['width']))
                    od_preds_no_gt.append(arch.predict(model=learn.model, batch=batch_dl))
                od_preds_no_gt = {i: p for (i, _), p in zip(im_no_gt, [x for b in od_preds_no_gt for x in b])}

            gt_masks, pred_masks = [], {s: [] for s in self.scores_steps}
            for imid, imdict in parser._imageid2info.items():
                if imid in idmap.name2id:
                    r = idmap.name2id[imid]
                    gtm = mask_utils.merge_obj_masks(interp.od_targs[r]['masks'].data, interp.od_targs[r]['labels'])
                    gt_masks.append(gtm)
                    im_pred = interp.od_preds[r]
                else:
                    gt_masks.append(empty_gt.copy())
                    im_pred = od_preds_no_gt[imid]
                for s, p in pred_masks.items():
                    om, ol, oidx = im_pred['masks'].data, im_pred['labels'], im_pred['scores'] >= s
                    p.append(mask_utils.merge_obj_masks(om[oidx], ol[oidx]) if oidx.any() else empty_gt.copy())
            interp.segm_targs = torch.stack([torch.tensor(gtm) for gtm in gt_masks])
            interp.segm_decoded = {s: torch.stack([torch.tensor(p) for p in sp]) for s, sp in pred_masks.items()}
            GPUManager.clean_gpu_memory(test_dl, batch_dl)
            return interp

    def evaluate_on_test_sets(self, learn, run):
        """Evaluate test sets, clears GPU memory held by test dl(s)"""
        for test_name, test_items_with_cls in self.get_test_items(merged=False):
            print("Testing model", run, "on", test_name)
            interp = self.compute_metrics(self.compute_test_predictions(learn, test_name, test_items_with_cls))
            del interp.od_preds, interp.od_targs, interp.segm_targs, interp.segm_decoded
            self.test_set_results[test_name][self.get_sorting_run_key(run)].append(interp)

    def compute_metrics(self, interp):
        interp.metrics = {}
        for mn, mfn in self.cust_metrics.items():
            mfn.accumulate(interp.od_targs, interp.od_preds)
            interp.metrics[mn] = torch.tensor(next(iter(mfn.finalize().values()))).float()
        if self.args.with_segm:
            segm_perf = partial(segm_utils.cls_perf, cats=self.args.cats, bg=None, axis=None)
            for s, preds in interp.segm_decoded.items():
                interp.metrics[f'{self.SEGM_PERF}cm_score{s}'] = self.compute_conf_mat(interp.segm_targs, preds)
            for fname, pfn in [(fname, getattr(train_utils, fname)) for fname in self.args.metrics_fns]:
                for cid, cat in zip([None, 0, *self.get_cats_idxs()], self.get_segm_cats_with_all()):
                    all_scores_res = []
                    for s in self.scores_steps:
                        mn = self.get_cat_segm_metric_name(fname, cat, s)
                        interp.metrics[mn] = segm_perf(pfn, interp.segm_decoded[s], interp.segm_targs, cid)
                        all_scores_res.append(interp.metrics[mn])
                    interp.metrics[self.get_cat_segm_metric_name(fname, cat, None)] = torch.stack(all_scores_res)
        return interp

    def plot_precision_recall(self, ax, pre_with_err, rec_with_err, show_val):
        (pre, pstd), rec, rstd = pre_with_err, rec_with_err
        svals = [[f'{(r, p)}' for r, p in zip(rr, pp)] for rr, pp in zip(rec, pre)] if show_val else None
        common.plot_lines_with_err(ax, rec, pre, self.get_segm_cats_with_all(), pstd, rstd, svals)
        score_steps = np.repeat(self.scores_steps[None, ], len(self.get_segm_cats_with_all()), 0)
        best_scores = [sorted(zip(*x))[-1][-1] for x in zip(pre + rec, pre, rec, score_steps)]
        for r, p, s in zip(rec, pre, best_scores):
            ax.plot(r[self.scores_steps == s], p[self.scores_steps == s], 'ro')
        return best_scores[0]   # all cat

    def plot_test_performance(self, test_path, run, agg_perf):
        show_val = not self.args.no_plot_val
        od_agg_perf = {k: v for k, v in agg_perf.items() if not k.startswith(self.SEGM_PERF)}
        fig, axs = common.new_fig_with_axs(1, len(self.metrics_types), self.args.test_figsize, sharey=True)
        for ax, mtype in zip([axs] if len(self.metrics_types) < 2 else axs, self.metrics_types):
            mtype_agg_perf = {k: v for k, v in od_agg_perf.items() if k.endswith(f'_{mtype.name}')}
            self.plot_custom_metrics(ax, mtype_agg_perf, show_val, title=f"OD {mtype.name} metrics")
        fig.tight_layout(pad=.2)
        save_path = self.plot_save_path(test_path, run, show_val, custom="_od_perf")
        plt.savefig(save_path, dpi=400)
        if self.args.with_segm:
            segm_agg_perf = {k: v for k, v in agg_perf.items() if k.startswith(self.SEGM_PERF)}
            fig, axs = common.new_fig_with_axs(1, 2, self.args.test_figsize)
            pre_with_err, rec_with_err = tuple(segm_agg_perf[f"{self.SEGM_PERF}{f}"] for f in ("precision", "recall"))
            best_score = self.plot_precision_recall(axs[0], pre_with_err, rec_with_err, show_val)
            common.plot_confusion_matrix(axs[1], agg_perf[f'{self.SEGM_PERF}cm_score{best_score}'], self.args.cats)
            fig.tight_layout(pad=.2)
            save_path = self.plot_save_path(test_path, run, show_val, custom="_segm_perf")
            plt.savefig(save_path, dpi=400)

    def get_arch(self):
        return getattr(ia, self.args.model), {}

    def get_tfms(self):
        train_tfms = ia.tfms.A.Adapter([
            *ia.tfms.A.aug_tfms(size=self.args.input_size, presize=int(1.5 * self.args.input_size)),
            ia.tfms.A.Normalize()
        ])
        valid_tfms = ia.tfms.A.Adapter([
            *ia.tfms.A.resize_and_pad(self.args.input_size),
            ia.tfms.A.Normalize()
        ])
        return train_tfms, valid_tfms

    @staticmethod
    def convert_ia_dls_to_fastai_dls(ia_dls):
        fastai_dls = []
        for dl in ia_dls:
            if isinstance(dl, ia.DataLoader):
                fastai_dl = ia.engines.fastai.adapters.convert_dataloader_to_fastai(dl)
            elif isinstance(dl, ia.fastai.DataLoader):
                fastai_dl = dl
            else:
                raise ValueError(f"dl type {type(dl)} not supported")

            fastai_dls.append(fastai_dl)
        return ia.fastai.DataLoaders(*fastai_dls).to(ia.fastai.default_device())

    def create_dls(self, tr, val, bs, size):
        train_tfms, valid_tfms = self.get_tfms()
        train_ds = ia.Dataset(tr[0], train_tfms)
        valid_ds = ia.Dataset(val[0], valid_tfms)
        arch, _ = self.get_arch()
        train_dl = arch.train_dl(train_ds, batch_size=bs, num_workers=4, shuffle=True)
        valid_dl = arch.valid_dl(valid_ds, batch_size=bs, num_workers=4, shuffle=False)
        return ImageObjectDetectionTrainer.convert_ia_dls_to_fastai_dls([train_dl, valid_dl])

    def create_learner(self, dls):
        arch, arch_params = self.get_arch()
        model = arch.model(num_classes=len(ia.ClassMap(self.args.cats[1:])), **arch_params)
        return arch.fastai.learner(dls=dls, model=model, metrics=self.cust_metrics.values())


def main(args):
    od = ImageObjectDetectionTrainer(args)
    od.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'mask_rcnn', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageObjectDetectionTrainer.get_argparser(desc="Fastai image object detection", pdef=defaults)
    args = parser.parse_args()

    ImageObjectDetectionTrainer.prepare_training(args)

    common.time_method(main, args)

