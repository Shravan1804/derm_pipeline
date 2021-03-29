import os
import sys

import numpy as np

import icevision.all as ia

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import train_utils, train_utils_img
import segmentation.mask_utils as mask_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP
from object_detection import object_detection_utils as obj_utils


class ImageObjectDetectionTrainer(train_utils_img.ImageTrainer):
    @staticmethod
    def get_argparser(desc="Fastai image obj detec trainer arguments", pdef=dict(), phelp=dict()):
        parser = super(ImageObjectDetectionTrainer, ImageObjectDetectionTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument("--backbone", type=str, help="backbone")
        return parser

    @staticmethod
    def prepare_training(args):
        args.exp_name = "img_obj_detec_" + args.exp_name
        super(ImageObjectDetectionTrainer, ImageObjectDetectionTrainer).prepare_training(args)

    def __init__(self, args, stratify=False, full_img_sep=CROP_SEP, **kwargs):
        super().__init__(args, stratify, full_img_sep, **kwargs)

    def load_items(self, anno_file):
        anno_path = os.path.join(self.args.data, 'annotation', anno_file)
        img_dir = os.path.join(self.args.data, 'images', os.path.splitext(anno_file)[0])
        no_split = ia.SingleSplitSplitter()
        records = ia.parsers.coco(annotations_file=anno_path, img_dir=img_dir).parse(data_splitter=no_split)[0]
        return records, np.ones_like(records)

    def get_cat_metric_name(self, perf_fn, cat, bg=None):
        pass

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        pass

    def aggregate_test_performance(self, folds_res):
        pass

    def compute_conf_mat(self, targs, preds):
        pass

    def process_test_preds(self, interp):
        pass

    def plot_test_performance(self, test_path, run, agg_perf):
        pass

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

