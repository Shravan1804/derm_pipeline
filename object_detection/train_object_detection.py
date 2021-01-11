import os
import sys

import numpy as np

import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils, train_utils_img
import segmentation.mask_utils as mask_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP
from object_detection.object_detection_utils import CustomCocoEval, segm_dataset_to_coco_format


class ImageObjectDetectionTrainer(train_utils_img.ImageTrainer):
    @staticmethod
    def get_argparser(desc="Fastai segmentation image trainer arguments", pdef=dict(), phelp=dict()):
        pass

    @staticmethod
    def prepare_training(args):
        args.exp_name = "img_obj_detec_" + args.exp_name
        pass

    def __init__(self, args, stratify=False, full_img_sep=CROP_SEP):
        super().__init__(args, stratify, full_img_sep)

    def load_items(self, path):
        pass

    def compute_conf_mat(self, targs, preds):
        pass

    def process_test_preds(self, interp):
        pass

    def plot_test_performance(self, test_path, run, agg_perf):
        super().plot_test_performance(test_path, run, agg_perf)
        figsize = self.args.test_figsize
        for show_val in [False, True]:
            save_path = os.path.join(test_path, f'{run}_coco{"_show_val" if show_val else ""}.jpg')
            CustomCocoEval.plot_coco_eval(self.coco_param_labels, agg_perf['cocoeval'], figsize, save_path, show_val)

    def create_dls(self, tr, val, bs, size):
        pass

    def create_learner(self, dls):
        pass

    def correct_wl(self, wl_items_with_labels, preds):
        pass

    def early_stop_cb(self):
        pass


def main(args):
    segm = ImageObjectDetectionTrainer(args)
    segm.train_model()
    if args.inference: segm.inference()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageObjectDetectionTrainer.get_argparser(desc="Fastai image classification", pdef=defaults)
    args = parser.parse_args()

    ImageObjectDetectionTrainer.prepare_training(args)

    common.time_method(main, args)

