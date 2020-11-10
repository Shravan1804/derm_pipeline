import os
import sys
import argparse
from functools import partial

import numpy as np

import torch
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
import common
import crypto
import train_utils
import segmentation.segmentation_utils as segm_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP


def get_img_mask(args, img_path):
    file, ext = os.path.splitext(img_path)
    mpath = f'{file.replace(args.img_dir, args.mask_dir)}{args.mext}'
    return crypto.decrypt_img(mpath, args.user_key) if args.encrypted else mpath


def segm_dls(bs, size, tr, val, args):
    return train_utils.create_dls_from_lst((fv.ImageBlock, fv.MaskBlock(args.cats)), lambda x: get_img_mask(args, x),
                                           bs, size, tr, val, args)


def get_segm_metrics(args):
    metrics_fn = {}
    device = f"'cuda:{args.gpu}'"
    for cat_id, cat in zip([*range(len(args.cats))] + [None], args.cats + ["all"]):
        for bg in [None, 0] if cat_id != 0 else [None]:
            for perf_fn in ['acc', 'prec', 'rec']:
                fn_name = f'{cat}_{perf_fn}{"" if bg is None else "_no_bg"}'
                code = f"def {fn_name}(inp, targ):" \
                       f"return cls_perf(common.{perf_fn}, inp, targ, {cat_id}, {args.cats}, {bg}).to({device})"
                exec(code, {"cls_perf": segm_utils.cls_perf, 'common': common}, metrics_fn)
    return list(metrics_fn.keys()), list(metrics_fn.values())


def create_learner(args, dls, metrics):
    return fv.unet_learner(dls, getattr(fv, args.model, None), metrics=metrics)


def main(args):
    print("Running script with args:", args)
    images = np.array(common.list_files(os.path.join(args.data, args.img_dir), full_path=True, posix_path=True))
    get_splits, get_dls = train_utils.get_data_fn(args, full_img_sep=CROP_SEP, stratify=False)
    metrics_names, metrics = get_segm_metrics(args)
    for fold, tr, val in get_splits(images):
        for it, run, dls in get_dls(partial(segm_dls, tr=tr, val=val, args=args), max_bs=len(tr)):
            learn = train_utils.prepare_learner(args, create_learner(args, dls, metrics)) if it == 0 else learn
            train_utils.basic_train(args, learn, fold, run, dls, metrics_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai segmentation")
    train_utils.common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34',
                                                '--cats': ["other", "pustules", "spots"]})
    train_utils.common_img_args(parser)
    segm_utils.common_segm_args(parser)
    args = parser.parse_args()

    train_utils.prepare_training(args, image_data=True)

    common.time_method(main, args)
