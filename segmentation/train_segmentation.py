import os
import sys
import datetime
from functools import partial

import numpy as np

import torch
import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import train_utils, train_utils_img
import segmentation.mask_utils as mask_utils
import segmentation.segmentation_utils as segm_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP
from object_detection.object_detection_utils import CustomCocoEval, segm_dataset_to_coco_format


class ImageSegmentationTrainer(train_utils_img.ImageTrainer):
    @staticmethod
    def get_argparser(desc="Fastai segmentation image trainer arguments", pdef=dict(), phelp=dict()):
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(ImageSegmentationTrainer, ImageSegmentationTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument('--coco-metrics', action='store_true', help="Also computes coco metrics")
        parser.add_argument('--rm-small-objs', action='store_true', help="Remove objs smaller than --min-size")
        parser.add_argument('--min-size', default=60, type=int, help="Objs below this size will be discarded")
        return segm_utils.common_segm_args(parser, pdef, phelp)

    @staticmethod
    def prepare_training(args):
        args.exp_name = "img_segm_" + args.exp_name
        super(ImageSegmentationTrainer, ImageSegmentationTrainer).prepare_training(args)

    def __init__(self, args, stratify=False, full_img_sep=CROP_SEP, **kwargs):
        self.NO_BG = '_no_bg'    # used to differentiate metrics ignoring background
        super().__init__(args, stratify, full_img_sep, **kwargs)
        self.loss_axis = 1

    def load_items(self, set_dir):
        path = os.path.join(self.args.data, set_dir)
        images = common.list_images(os.path.join(path, self.args.img_dir), full_path=True, posix_path=True)
        return fv.L(images), fv.L([self.get_image_mask_path(img_path) for img_path in images])

    def get_image_mask_path(self, img_path):
        return segm_utils.get_mask_path(img_path, self.args.img_dir, self.args.mask_dir, self.args.mext)

    def load_mask(self, item):
        if type(item) is np.ndarray: mask = item
        elif common.is_path(item): mask = self.load_image_item(item)
        else: mask = mask_utils.rles_to_non_binary_mask(item)
        if self.args.rm_small_objs:
            if common.is_path(mask): mask = mask_utils.load_mask_array(mask)
            mask = mask_utils.rm_small_objs_from_non_bin_mask(mask, self.args.min_size, range(len(self.args.cats)), self.args.bg)
        return mask

    def get_cat_metric_name(self, perf_fn, cat, bg):
        return f'{super().get_cat_metric_name(perf_fn, cat)}{"" if bg is None else self.NO_BG}'

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        for bg in [None, self.args.bg]:
            cat_perf = partial(segm_utils.cls_perf, cls_idx=cat_id, cats=self.args.cats, bg=bg)
            signature = f'{self.get_cat_metric_name(perf_fn, cat, bg)}(inp, targ)'
            code = f"def {signature}: return cat_perf(train_utils.{perf_fn}, inp, targ).to(inp.device)"
            exec(code, {"cat_perf": cat_perf, 'train_utils': train_utils}, metrics_fn)

    def ordered_test_perfs_per_cats(self):
        ordered = []
        for perf_fn in self.args.metrics_fns:
            for bg in [None, self.args.bg]:
                mns = [self.get_cat_metric_name(perf_fn, cat, bg) for cat in self.get_cats_with_all()]
                ordered.append((mns, perf_fn + ("" if bg is None else self.NO_BG)))
        return ordered

    def compute_conf_mat(self, targs, preds): return segm_utils.pixel_conf_mat(targs, preds, self.args.cats)

    def compute_metrics(self, interp):
        interp.targs = interp.targs.as_subclass(torch.Tensor)   # otherwise issues with fastai PILMask custom class
        interp = super().compute_metrics(interp)
        if self.args.coco_metrics:
            to_coco = partial(segm_dataset_to_coco_format, cats=self.args.cats, bg=self.args.bg)
            with common.elapsed_timer() as elapsed:
                gt, dt = to_coco(interp.targs), to_coco(interp.decoded, scores=True)
                print(f"Segmentation dataset converted in {datetime.timedelta(seconds=elapsed())}.")
            cocoEval = CustomCocoEval(gt, dt, all_cats=self.ALL_CATS)
            cocoEval.eval_acc_and_summarize(verbose=False)
            self.coco_param_labels, areaRng, maxDets, stats = cocoEval.get_precision_recall_with_labels()
            interp.metrics['cocoeval'] = torch.Tensor(stats)
            interp.metrics['cocoeval_areaRng'] = torch.Tensor(areaRng)
            interp.metrics['cocoeval_maxDets'] = torch.Tensor(maxDets)
        return interp

    def plot_test_performance(self, test_path, run, agg_perf):
        super().plot_test_performance(test_path, run, agg_perf)
        if self.args.coco_metrics:
            figsize = self.args.test_figsize
            cocoeval = agg_perf['cocoeval']
            show_val = not self.args.no_plot_val
            save_path = self.plot_save_path(test_path, run, show_val, custom="_coco")
            CustomCocoEval.plot_coco_eval(self.coco_param_labels, cocoeval, figsize, save_path, show_val)

    def create_dls(self, tr, val, bs, size):
        blocks = fv.ImageBlock, fv.MaskBlock(self.args.cats)
        return self.create_dls_from_lst(blocks, tr, val, bs, size, get_y=self.load_mask)

    def get_class_weights(self, train_items):
        masks = np.stack([self.load_mask(mpath) for _, mpath in train_items])
        _, class_counts = np.unique(masks, return_counts=True)
        return torch.FloatTensor(class_counts.max() / class_counts)

    def create_learner(self, dls):
        learn_kwargs = self.get_learner_kwargs(dls)
        metrics = list(self.cust_metrics.values()) + [fv.foreground_acc]
        learn = fv.unet_learner(dls, getattr(fv, self.args.model), metrics=metrics, **learn_kwargs)
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, preds):
        wl_items, labels = wl_items_with_labels
        labels = [mask_utils.non_binary_mask_to_rles(self.load_image_item(pred.numpy())) for pred in preds]
        # preds size is self.args.input_size but wl_items are orig size => first item_tfms should resize the input
        return wl_items, fv.L(labels)

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='foreground_acc', min_delta=0.01, patience=3)

    @staticmethod
    def load_pretrained_backbone_weights(weights_path, model):
        new_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        model_state_dict = model.state_dict()
        for name, param in model_state_dict.items():
            new_name = name.replace('layers.', '')
            if new_name in new_state_dict:
                input_param = new_state_dict[new_name]
                if input_param.shape == param.shape:
                    param.copy_(input_param)
                else:
                    print('Shape mismatch at:', name, 'skipping')
            else:
                print(f'{name} weight of the model not in pretrained weights')
        model.load_state_dict(model_state_dict)

def main(args):
    segm = ImageSegmentationTrainer(args)
    segm.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageSegmentationTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    ImageSegmentationTrainer.prepare_training(args)

    common.time_method(main, args)

