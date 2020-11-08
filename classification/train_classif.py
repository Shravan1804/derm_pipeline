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
from PatchExtractor import PatchExtractor


def classif_dls(bs, size, tr, val, args):
    return train_utils.create_dls_from_lst((fv.ImageBlock, fv.CategoryBlock), fv.parent_label, bs, size, tr, val, args)


def get_classif_metrics(cats):
    def cls_perf(perf, inp, targ, cls_idx, axis=-1):
        if axis is not None:
            inp = inp.argmax(dim=axis)
        if cls_idx is None:
            res = [cls_perf(perf, inp, targ, c, axis=None) for c in range(len(cats))]
            return torch.tensor(perf(*torch.cat([r.unsqueeze(0) for r in res], dim=0).sum(axis=0).tolist()))
        else:
            return torch.tensor(perf(*common.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx)))

    metrics_fn = {}
    for cat_id, cat in zip([*range(len(cats))] + [None], cats + ["all"]):
        for perf_fn in ['acc', 'prec', 'rec']:
            code = f"def {cat}_{perf_fn}(inp, targ): return cls_perf(common.{perf_fn}, inp, targ, {cat_id})"
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
    images = np.array(common.list_files_in_dirs(train_utils.get_data_path(args), full_path=True, posix_path=True))
    get_splits, get_dls = train_utils.get_data_fn(args, full_img_sep=PatchExtractor.SEP, stratify=True)
    cats = common.list_dirs(train_utils.get_data_path(args), full_path=False)
    metrics_names, metrics = get_classif_metrics(cats, args.metrics)
    for fold, tr, val in get_splits(images):
        for it, run, dls in get_dls(partial(classif_dls, tr=tr, val=val, args=args), max_bs=len(tr)):
            assert cats == dls.vocab, f"Category missmatch between metrics cats ({cats}) and dls cats ({dls.vocab})"
            learn = create_learner(args, dls, metrics) if it == 0 else learn
            learn.dls = dls
            train_utils.setup_tensorboard(learn, args.exp_logdir, run, metrics_names)
            learn.fine_tune(args.epochs)
            save_path = os.path.join(args.exp_logdir, f'{common.zero_pad(fold, args.nfolds)}_{run}_model')
            train_utils.save_learner(learn, is_fp16=(not args.full_precision), save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai segmentation")
    parser.add_argument('--data', type=str, required=True, help="Root dataset dir")
    parser.add_argument('--encrypted', action='store_true', help="Data is encrypted")
    parser.add_argument('--user-key', type=str, help="Data encryption key")
    parser.add_argument('--use-wl-sl', action='store_true', help="Data dir contains wl and sl data")
    parser.add_argument('--wl-train', type=str, default='weak_labels', help="weak labels (wl) dir")
    parser.add_argument('--sl-train', type=str, default='strong_labels_train', help="strong labels (sl) dir")
    parser.add_argument('--sl-test', type=str, default='strong_labels_test', help="sl test dir")

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
