import os
import sys
from pathlib import PosixPath
from functools import partial

import numpy as np

import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils, train_utils_img
import segmentation.segmentation_utils as segm_utils
import segmentation.mask_utils as mask_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP


class ImageSegmentationTrainer(train_utils_img.ImageTrainer):
    def get_argparser(desc="Fastai segmentation image trainer arguments", pdef=dict(), phelp=dict()):
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(ImageSegmentationTrainer, ImageSegmentationTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument('--img-dir', type=str, default=pdef.get('--img-dir', "images"),
                            help=phelp.get('--img-dir', "Images dir"))
        parser.add_argument('--mask-dir', type=str, default=pdef.get('--mask-dir', "masks"),
                            help=phelp.get('--mask-dir', "Masks dir"))
        parser.add_argument('--mext', type=str, default=pdef.get('--mext', ".png"),
                            help=phelp.get('--mext', "Masks file extension"))
        parser.add_argument('--bg', type=int, default=pdef.get('--bg', 0),
                            help=phelp.get('--bg', "Background mask code"))
        return parser

    def load_items(self, path):
        images = np.array(common.list_files(os.path.join(path, self.args.img_dir), full_path=True, posix_path=True))
        return images, np.array([self.get_image_mask_path(img_path) for img_path in images])

    def get_image_mask_path(self, img_path):
        return segm_utils.get_mask_path(img_path, self.args.img_dir, self.args.mask_dir, self.args.mext)

    def load_mask(self, item):
        is_path = type(item) in (str, PosixPath)
        return self.load_image_item(item) if is_path else mask_utils.rles_to_non_binary_mask(item)

    def get_metrics(self):
        metrics_fn = {}
        for cat_id, cat in zip([*range(len(self.args.cats))] + [None], self.args.cats + [self.ALL_CATS]):
            for bg in [None, self.args.bg] if cat_id != self.args.bg else [None]:
                cat_perf = partial(segm_utils.cls_perf, cls_idx=cat_id, cats=self.args.cats, bg=bg)
                for perf_fn in self.BASIC_PERF_FNS:
                    fn_name = f'{cat}_{perf_fn}{"" if bg is None else "_no_bg"}'
                    code = f"def {fn_name}(inp, targ): return cat_perf(train_utils.{perf_fn}, inp, targ).to(inp.device)"
                    exec(code, {"cat_perf": cat_perf, 'train_utils': train_utils}, metrics_fn)
        return metrics_fn

    def create_dls(self, tr, val, bs, size):
        tr, val = map(lambda x: tuple(map(np.ndarray.tolist, x)), (tr, val))
        blocks = fv.ImageBlock, fv.MaskBlock(args.cats)
        return self.create_dls_from_lst(blocks, tr, val, bs, size, get_y=self.load_mask)

    def create_learner(self, dls):
        metrics = list(self.cats_metrics.values()) + [fv.foreground_acc]
        learn = fv.unet_learner(dls, getattr(fv, self.args.model), metrics=metrics)
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, preds):
        wl_items, labels = wl_items_with_labels
        labels = [mask_utils.non_binary_mask_to_rles(self.load_image_item(pred.numpy())) for pred in preds]
        # preds size is self.args.input_size but wl_items are orig size => first item_tfms should resize the input
        return (wl_items, np.array(labels)), ""

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='foreground_acc', min_delta=0.01, patience=3)

    def process_preds(self, interp):
        interp.cm = segm_utils.pixel_conf_mat(self.args.cats, interp.decoded, interp.targs)
        return interp


def main(args):
    segm = ImageSegmentationTrainer(args, stratify=False, full_img_sep=CROP_SEP)
    segm.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageSegmentationTrainer.get_argparser(desc="Fastai image classification", pdef=defaults)
    args = parser.parse_args()

    args.exp_name = "img_segm_" + args.exp_name

    train_utils_img.ImageTrainer.prepare_training(args)

    common.time_method(main, args, prepend=f"GPU {args.proc_gpu} proc: ")
