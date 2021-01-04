import os
import sys
import datetime
from pathlib import PosixPath
from functools import partial

import numpy as np

import torch
import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils, train_utils_img
import segmentation.mask_utils as mask_utils
import segmentation.segmentation_utils as segm_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP
from object_detection.object_detection_utils import CustomCocoEval, segm_dataset_to_coco_format


class ImageSegmentationTrainer(train_utils_img.ImageTrainer):
    @staticmethod
    def get_argparser(desc="Fastai segmentation image trainer arguments", pdef=dict(), phelp=dict()):
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(ImageSegmentationTrainer, ImageSegmentationTrainer).get_argparser(desc, pdef, phelp)
        return segm_utils.common_segm_args(parser, pdef, phelp)

    @staticmethod
    def prepare_training(args):
        args.exp_name = "img_segm_" + args.exp_name
        super(ImageSegmentationTrainer, ImageSegmentationTrainer).prepare_training(args)

    def __init__(self, args, stratify=False, full_img_sep=CROP_SEP):
        self.NO_BG = '_no_bg'    # used to differentiate metrics ignoring background
        super().__init__(args, stratify, full_img_sep)

    def load_items(self, path):
        images = np.array(common.list_files(os.path.join(path, self.args.img_dir), full_path=True, posix_path=True))
        return images, np.array([self.get_image_mask_path(img_path) for img_path in images])

    def get_image_mask_path(self, img_path):
        return segm_utils.get_mask_path(img_path, self.args.img_dir, self.args.mask_dir, self.args.mext)

    def load_mask(self, item):
        is_path = type(item) in (str, PosixPath)
        return self.load_image_item(item) if is_path else mask_utils.rles_to_non_binary_mask(item)

    def get_cat_metric_name(self, perf_fn, cat, bg=None):
        return f'{super().get_cat_metric_name(perf_fn, cat)}{"" if bg is None else self.NO_BG}'

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        for bg in [None, self.args.bg]:
            cat_perf = partial(segm_utils.cls_perf, cls_idx=cat_id, cats=self.args.cats, bg=bg)
            signature = f'{self.get_cat_metric_name(perf_fn, cat, bg)}(inp, targ)'
            code = f"def {signature}: return cat_perf(train_utils.{perf_fn}, inp, targ).to(inp.device)"
            exec(code, {"cat_perf": cat_perf, 'train_utils': train_utils}, metrics_fn)

    def aggregate_test_performance(self, folds_res):
        """Returns a dict with perf_fn as keys and values a tuple of lsts of categories mean/std"""
        agg = super().aggregate_test_performance(folds_res)
        for perf_fn in self.BASIC_PERF_FNS:
            mns = [self.get_cat_metric_name(perf_fn, cat, self.args.bg) for cat in self.get_cats_with_all()]
            agg[f'{perf_fn}{self.NO_BG}'] = tuple(np.stack(s) for s in zip(*[agg.pop(mn) for mn in mns]))
        return agg

    def compute_conf_mat(self, targs, preds): return segm_utils.pixel_conf_mat(targs, preds, self.args.cats)

    def process_test_preds(self, interp):
        interp = super().process_test_preds(interp)
        to_coco = partial(segm_dataset_to_coco_format, cats=self.args.cats, bg=self.args.bg)
        with common.elapsed_timer() as elapsed:
            gt, dt = to_coco(interp.targs), to_coco(interp.decoded, scores=True)
            print(f"Segmentation dataset converted in {datetime.timedelta(seconds=elapsed())}.")
        cocoEval = CustomCocoEval(gt, dt, all_cats=self.ALL_CATS)
        cocoEval.eval_acc_and_summarize(verbose=False)
        self.coco_param_labels, stats = cocoEval.getPrecisionRecall()  # cocoeval labels is the same for all tests
        interp.metrics['cocoeval'] = torch.Tensor(stats)
        return interp

    def plot_test_performance(self, test_path, run, agg_perf):
        super().plot_test_performance(test_path, run, agg_perf)
        figsize = self.args.test_figsize
        for show_val in [False, True]:
            save_path = os.path.join(test_path, f'{run}_coco{"_show_val" if show_val else ""}.jpg')
            CustomCocoEval.plot_coco_eval(self.coco_param_labels, agg_perf['cocoeval'], figsize, save_path, show_val)

    def create_dls(self, tr, val, bs, size):
        tr, val = map(lambda x: tuple(map(np.ndarray.tolist, x)), (tr, val))
        blocks = fv.ImageBlock, fv.MaskBlock(self.args.cats)
        return self.create_dls_from_lst(blocks, tr, val, bs, size, get_y=self.load_mask)

    def create_learner(self, dls):
        metrics = list(self.cust_metrics.values()) + [fv.foreground_acc]
        learn = fv.unet_learner(dls, getattr(fv, self.args.model), metrics=metrics)
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, preds):
        wl_items, labels = wl_items_with_labels
        labels = [mask_utils.non_binary_mask_to_rles(self.load_image_item(pred.numpy())) for pred in preds]
        # preds size is self.args.input_size but wl_items are orig size => first item_tfms should resize the input
        return wl_items, np.array(labels)

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='foreground_acc', min_delta=0.01, patience=3)


def main(args):
    segm = ImageSegmentationTrainer(args)
    segm.train_model()
    if args.inference: segm.inference()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageSegmentationTrainer.get_argparser(desc="Fastai image classification", pdef=defaults)
    args = parser.parse_args()

    ImageSegmentationTrainer.prepare_training(args)

    common.time_method(main, args)

