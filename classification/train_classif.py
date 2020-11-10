import os
import sys
import argparse
from functools import partial

import numpy as np

import torch
import fastai.vision.all as fv
import fastai.distributed as fd

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils
from general.PatchExtractor import PatchExtractor
import classification.classification_utils as classif_utils


def classif_dls(bs, size, tr, val, args):
    return train_utils.create_dls_from_lst((fv.ImageBlock, fv.CategoryBlock(vocab=args.cats)), fv.parent_label,
                                           bs, size, tr, val, args)


def get_classif_metrics(args):
    metrics_fn = {}
    device = f"'cuda:{args.gpu}'"
    for cat_id, cat in zip([*range(len(args.cats))] + [None], args.cats + ["all"]):
        for perf_fn in ['acc', 'prec', 'rec']:
            code = f"def {cat}_{perf_fn}(inp, targ):" \
                   f"return cls_perf(common.{perf_fn}, inp, targ, {cat_id}, {args.cats}).to({device})"
            exec(code, {"cls_perf": classif_utils.cls_perf, 'common': common}, metrics_fn)
    return list(metrics_fn.keys()), list(metrics_fn.values())


def create_learner(args, dls, metrics):
    if "efficientnet" in args.model:
        from efficientnet_pytorch import EfficientNet
        model = fd.rank0_first(lambda: EfficientNet.from_pretrained(args.model))
        model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
        return fv.Learner(dls, model, metrics=metrics,
                          splitter=lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params))
    else:
        return fd.rank0_first(lambda: fv.cnn_learner(dls, getattr(fv, args.model, None), metrics=metrics))


def main(args):
    print("Running script with args:", args)
    images = np.array(common.list_files_in_dirs(train_utils.get_data_path(args), full_path=True, posix_path=True))
    get_splits, get_dls = train_utils.get_data_fn(args, full_img_sep=PatchExtractor.SEP, stratify=True)
    metrics_names, metrics = get_classif_metrics(args)
    for fold, tr, val in get_splits(images):
        for it, run, dls in get_dls(partial(classif_dls, tr=tr, val=val, args=args), max_bs=len(tr)):
            learn = train_utils.prepare_learner(args, create_learner(args, dls, metrics)) if it == 0 else learn
            train_utils.basic_train(args, learn, fold, run, dls, metrics_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai classification")
    train_utils.common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34'})
    train_utils.common_img_args(parser, pdef={'--input-size': 256})
    args = parser.parse_args()

    train_utils.prepare_training(args, image_data=True)

    if args.cats is None:
        args.cats = common.list_dirs(train_utils.get_data_path(args), full_path=False)

    common.time_method(main, args)
