import os
import sys
import argparse
from functools import partial

import numpy as np

import torch
import fastai.vision.all as fv

sys.path.insert(0, '..')
import common
import crypto
import train_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP


def get_img_mask(args, img_path):
    file, ext = os.path.splitext(img_path)
    mpath = f'{file.replace(args.img_dir, args.mask_dir)}{args.mext}'
    return crypto.decrypt_img(mpath, args.user_key) if args.encrypted else mpath


def segm_dls(bs, size, tr, val, args):
    return train_utils.create_dls_from_lst((fv.ImageBlock, fv.MaskBlock(args.cats)), lambda x: get_img_mask(args, x),
                                           bs, size, tr, val, args)


def get_segm_metrics(cats):
    def cls_perf(perf, inp, targ, cls_idx, bg=None, axis=1):
        """If bg sets then computes perf without background"""
        assert bg != cls_idx or cls_idx is None, f"Cannot compute class {cls_idx} perf as bg = {bg}"
        if axis is not None:
            inp = inp.argmax(dim=axis)
        if bg is not None:
            mask = targ != bg
            inp, targ = inp[mask], targ[mask]
        if cls_idx is None:
            res = [cls_perf(perf, inp, targ, c, bg, axis=None) for c in range(0 if bg is None else 1, len(cats))]
            return torch.tensor(perf(*torch.cat([r.unsqueeze(0) for r in res], dim=0).sum(axis=0).tolist()))
        else:
            return torch.tensor(perf(*common.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx)))

    metrics_fn = {}
    for cat_id, cat in zip([*range(len(cats))] + [None], cats + ["all"]):
        for bg in [None, 0] if cat_id != 0 else [None]:
            for perf_fn in ['acc', 'prec', 'rec']:
                fn_name = f'{cat}_{perf_fn}{"" if bg is None else "_no_bg"}'
                code = f"def {fn_name}(inp, targ): return cls_perf(common.{perf_fn}, inp, targ, {cat_id}, {bg})"
                exec(code, {"cls_perf": cls_perf, 'common': common}, metrics_fn)
    return list(metrics_fn.keys()), list(metrics_fn.values())


def create_learner(args, dls, metrics):
    model = getattr(fv, args.model, None)
    assert model is not None, f"Provided model architecture {args.model} unknown."
    learn = fv.unet_learner(dls, model, metrics=metrics)
    if not args.full_precision:
        learn.to_fp16()
    return learn


def main(args):
    print("Running script with args:", args)
    images = np.array(common.list_files(os.path.join(args.data, args.img_dir), full_path=True, posix_path=True))
    get_splits, get_dls = train_utils.get_data_fn(args, full_img_sep=CROP_SEP, stratify=False)
    metrics_names, metrics = get_segm_metrics(args.cats)
    for fold, tr, val in get_splits(images):
        for it, run, dls in get_dls(partial(segm_dls, tr=tr, val=val, args=args), max_bs=len(tr)):
            learn = create_learner(args, dls, metrics) if it == 0 else learn
            learn.dls = dls
            train_utils.setup_tensorboard(learn, args.exp_logdir, run, metrics_names)
            learn.fine_tune(args.epochs)
            save_path = os.path.join(args.exp_logdir, f'{common.zero_pad(fold, args.nfolds)}_{run}_model')
            train_utils.save_learner(learn, is_fp16=(not args.full_precision), save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai segmentation")
    parser.add_argument('--data', type=str, required=True, help="Root dataset dir")
    parser.add_argument('--img-dir', type=str, default="images", help="Images dir")
    parser.add_argument('--mask-dir', type=str, default="masks", help="Masks dir")
    parser.add_argument('--mext', type=str, default=".png", help="Masks file extension")
    parser.add_argument('--cats', type=str, nargs='+', default=["other", "pustules", "spots"], help="Segm categories")
    parser.add_argument('--encrypted', action='store_true', help="Data is encrypted")
    parser.add_argument('--user-key', type=str, help="Data encryption key")

    parser.add_argument('--cross-val', action='store_true', help="Perform 5-fold cross validation on sl train set")
    parser.add_argument('--nfolds', default=5, type=int, help="Number of folds for cross val")
    parser.add_argument('--valid-size', default=.2, type=float, help='If no cross val, splits train set with this %')

    parser.add_argument('--input-size', default=512, type=int, help="Model input will be resized to this value")
    parser.add_argument('--progr-size', action='store_true', help="Applies progressive resizing")
    parser.add_argument('--size-facts', default=[.25, .5, 1], nargs='+', type=float, help='Incr. progr. size factors')

    parser.add_argument('--norm', action='store_true', help="Normalizes images to imagenet stats")
    parser.add_argument('--full-precision', action='store_true', help="Train with full precision (more gpu memory)")
    train_utils.add_common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34'})
    train_utils.add_multi_gpus_args(parser)
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    common.set_seeds(args.seed)
    train_utils.maybe_set_gpu(args.gpuid, args.num_gpus)

    if args.encrypted:
        args.user_key = crypto.request_key(args.data, args.user_key)

    if args.exp_logdir is None:
        args.exp_logdir = common.maybe_create(args.logdir, train_utils.get_exp_logdir(args, custom="progr_size"
        if args.progr_size else ""))
    print("Creation of log directory: ", args.exp_logdir)

    common.time_method(main, args)
