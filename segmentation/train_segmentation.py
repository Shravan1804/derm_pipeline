import os
import sys
import argparse
from functools import partial

import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto, train_utils
import segmentation.segmentation_utils as segm_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP


class ImageSegmentationTrainer(train_utils.ImageTrainer):
    def load_data(self, path):
        return common.list_files(os.path.join(path, self.args.img_dir), full_path=True, posix_path=True)

    def get_img_mask(self, img_path):
        file, ext = os.path.splitext(img_path)
        mpath = f'{file.replace(self.args.img_dir, self.args.mask_dir)}{self.args.mext}'
        return crypto.decrypt_img(mpath, self.args.user_key) if self.args.encrypted else mpath

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
        return self.create_dls_from_lst((fv.ImageBlock, fv.MaskBlock(args.cats)), tr.tolist(), val.tolist(),
                                        self.get_img_mask, bs, size)

    def create_learner(self, dls):
        metrics = list(self.cats_metrics.values()) + [fv.foreground_acc]
        learn = fv.unet_learner(dls, getattr(fv, self.args.model), metrics=metrics)
        return self.prepare_learner(learn)

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='foreground_acc', min_delta=0.01, patience=3)

    def process_preds(self, interp):
        return interp


def main(args):
    segm = ImageSegmentationTrainer(args, stratify=False, full_img_sep=CROP_SEP)
    segm.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai segmentation")
    train_utils.common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34',
                                                '--cats': ["other", "pustules", "spots"]})
    train_utils.common_img_args(parser)
    segm_utils.common_segm_args(parser)
    args = parser.parse_args()

    train_utils.prepare_training(args, image_data=True, custom="segmentation")

    common.time_method(main, args, prepend=f"GPU {args.gpu} proc: ")
