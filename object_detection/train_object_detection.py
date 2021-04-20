import os
import sys
from types import SimpleNamespace

import torch
import icevision.all as ia
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import train_utils_img
from training.train_utils import GPUManager
from segmentation.crop_to_thresh import SEP as CROP_SEP
from object_detection import object_detection_utils as obj_utils


class ImageObjectDetectionTrainer(train_utils_img.ImageTrainer):
    @staticmethod
    def get_argparser(desc="Fastai image obj detec trainer arguments", pdef=dict(), phelp=dict()):
        parser = super(ImageObjectDetectionTrainer, ImageObjectDetectionTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument("--backbone", type=str, help="backbone")
        parser.add_argument('--ious', default=[.15, .25], nargs='+', type=float, help="Metrics IoUs")
        parser.add_argument('--with-segm', action='store_true', help="Metrics will be also computed on segmentation")
        return parser

    @staticmethod
    def prepare_training(args):
        args.exp_name = "img_obj_detec_" + args.exp_name
        super(ImageObjectDetectionTrainer, ImageObjectDetectionTrainer).prepare_training(args)

    def __init__(self, args, stratify=False, full_img_sep=CROP_SEP, **kwargs):
        self.metrics_types = [ia.COCOMetricType.bbox] + ([ia.COCOMetricType.mask] if args.with_segm else [])
        super().__init__(args, stratify, full_img_sep, **kwargs)
        if self.args.encrypted:
            def _crypted_load(mixin_self):
                mixin_self.img = self.load_image_item(mixin_self.filepath)
                mixin_self.height, mixin_self.width, _ = mixin_self.img.shape
                super(ia.core.FilepathRecordMixin, mixin_self)._load()
            ia.core.FilepathRecordMixin._load = _crypted_load
            # TODO: hack, solve this better

    def load_items(self, anno_file):
        anno_path = os.path.join(self.args.data, 'annotations', anno_file)
        img_dir = os.path.join(self.args.data, 'images', os.path.splitext(anno_file)[0])
        no_split = ia.SingleSplitSplitter()
        coco_parser = obj_utils.COCOMaskParserEncrypted if self.args.encrypted else ia.parsers.COCOMaskParser
        records = fv.L(coco_parser(annotations_filepath=anno_path, img_dir=img_dir).parse(data_splitter=no_split)[0])
        # dummy cls so that the split_data pipeline works
        return records, fv.L([1]*len(records))

    def get_patch_full_img(self, patch): return super().get_patch_full_img(str(patch.filepath))

    def get_cats_with_all(self):
        # category at index 0 is the background class, which should be ignored
        return [self.ALL_CATS, *self.args.cats[1:]]

    def get_cat_metric_name(self, perf_fn, cat, iou, mtype):
        return f'{super().get_cat_metric_name(perf_fn, cat)}_iou{iou}_{mtype.name}'

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
        return ordered

    def evaluate_on_test_sets(self, learn, run):
        """Evaluate test sets, clears GPU memory held by test dl(s)"""
        for test_name, test_items_with_cls in self.get_test_items(merged=False):
            print("Testing model", run, "on", test_name)
            GPUManager.sync_distributed_process()
            arch, _ = self.get_arch()
            test_ds = ia.Dataset(test_items_with_cls[0], self.get_tfms()[1])
            test_dl = arch.infer_dl(test_ds, batch_size=learn.dls.bs)
            with GPUManager.running_context(learn, self.args.gpu_ids):
                interp = SimpleNamespace()
                interp.targs, interp.preds = arch.predict_dl(model=learn.model, infer_dl=test_dl)
            interp = self.process_test_preds(interp)
            del interp.preds, interp.targs
            GPUManager.clean_gpu_memory(test_dl)
            self.test_set_results[test_name][self.get_sorting_run_key(run)].append(interp)

    def process_test_preds(self, interp):
        interp.metrics = {}
        for mn, mfn in self.cust_metrics.items():
            mfn.accumulate(interp.targs, interp.preds)
            interp.metrics[mn] = torch.tensor(next(iter(mfn.finalize().values()))).float()
        return interp

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
    defaults = {'--bs': 6, '--model': 'faster_rcnn', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageObjectDetectionTrainer.get_argparser(desc="Fastai image object detection", pdef=defaults)
    args = parser.parse_args()

    ImageObjectDetectionTrainer.prepare_training(args)

    common.time_method(main, args)

