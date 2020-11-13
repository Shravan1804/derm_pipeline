import os
import sys
import argparse

import fastai.vision.all as fv
import fastai.distributed as fd
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto, train_utils
import segmentation.segmentation_utils as segm_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP


class ImageSegmentationTrainer(train_utils.ImageTrainer):
    def get_img_mask(self, img_path):
        file, ext = os.path.splitext(img_path)
        mpath = f'{file.replace(self.args.img_dir, self.args.mask_dir)}{self.args.mext}'
        return crypto.decrypt_img(mpath, self.args.user_key) if self.args.encrypted else mpath

    def get_items(self, train=True):
        if not train:
            test_dirs = self.get_data_path(train=False)
            test_dirs = [(os.path.basename(p), os.path.join(p, self.args.img_dir)) for p in test_dirs]
            return [(p, common.list_files(fp, full_path=True, posix_path=True)) for p, fp in test_dirs]
        sl_path = os.path.join(self.get_data_path(), self.args.img_dir)
        sl_images = common.list_files(sl_path, full_path=True, posix_path=True)
        if self.args.use_wl:
            wl_path = os.path.join(self.get_data_path(weak_labels=True), self.args.img_dir)
            wl_images = common.list_files(wl_path, full_path=True, posix_path=True)
        return sl_images, wl_images if self.args.use_wl else None

    def get_metrics(self):
        metrics_fn = {}
        device = f"'cuda:{self.args.gpu}'"
        for cat_id, cat in zip([*range(len(self.args.cats))] + [None], self.args.cats + [self.ALL_CATS]):
            for bg in [None, 0] if cat_id != 0 else [None]:
                for perf_fn in self.BASIC_PERF_FNS:
                    fn_name = f'{cat}_{perf_fn}{"" if bg is None else "_no_bg"}'
                    code = f"def {fn_name}(inp, targ): return cls_perf(common.{perf_fn}, inp, targ, {cat_id}, " \
                           f"{self.args.cats}, {bg}).to({device})"
                    exec(code, {"cls_perf": segm_utils.cls_perf, 'common': common}, metrics_fn)
        return list(metrics_fn.keys()), list(metrics_fn.values())

    def create_dls(self, tr, val, bs, size):
        return self.create_dls_from_lst((fv.ImageBlock, fv.MaskBlock(args.cats)), tr.tolist(), val.tolist(),
                                        lambda x: self.get_img_mask(x), bs, size)

    def create_learner(self, dls):
        metrics = self.cats_metrics_fn + [fv.foreground_acc]
        learn = fd.rank0_first(lambda: fv.unet_learner(dls, getattr(fv, self.args.model), metrics=metrics))
        return self.prepare_learner(learn)

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='foreground_acc', min_delta=0.01, patience=3)

    def interpret_preds(self, interp):
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
