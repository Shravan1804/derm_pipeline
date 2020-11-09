import os
import sys
import argparse
from functools import partial

import numpy as np

import torch
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
import common
import train_utils
from PatchExtractor import PatchExtractor


def classif_dls(bs, size, tr, val, args):
    return train_utils.create_dls_from_lst((fv.ImageBlock, fv.CategoryBlock), fv.parent_label, bs, size, tr, val, args)


def get_classif_metrics(cats):
    def cls_perf(perf, inp, targ, cls_idx, axis=-1):
        if axis is not None:
            inp = inp.argmax(dim=axis)
        if cls_idx is None:
            res = [common.get_cls_TP_TN_FP_FN(targ == c, inp == c) for c in range(len(cats))]
            res = torch.cat([torch.tensor(r).unsqueeze(0) for r in res], dim=0).sum(axis=0).tolist()
            return torch.tensor(perf(*res))
        else:
            return torch.tensor(perf(*common.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx)))

    metrics_fn = {}
    for cat_id, cat in zip([*range(len(cats))] + [None], cats + ["all"]):
        for perf_fn in ['acc', 'prec', 'rec']:
            code = f"def {cat}_{perf_fn}(inp, targ): return cls_perf(common.{perf_fn}, inp, targ, {cat_id})"
            exec(code, {"cls_perf": cls_perf, 'common': common}, metrics_fn)
    return list(metrics_fn.keys()), list(metrics_fn.values())


def create_learner(args, dls, metrics):
    if "efficientnet" in args.model:
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(args.model)
        model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
        return fv.Learner(dls, model, metrics=metrics,
                          splitter=lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params))
    else:
        return fv.cnn_learner(dls, getattr(fv, args.model, None), metrics=metrics)


def main(args):
    print("Running script with args:", args)
    images = np.array(common.list_files_in_dirs(train_utils.get_data_path(args), full_path=True, posix_path=True))
    get_splits, get_dls = train_utils.get_data_fn(args, full_img_sep=PatchExtractor.SEP, stratify=True)
    cats = common.list_dirs(train_utils.get_data_path(args), full_path=False)
    metrics_names, metrics = get_classif_metrics(cats)
    for fold, tr, val in get_splits(images):
        for it, run, dls in get_dls(partial(classif_dls, tr=tr, val=val, args=args), max_bs=len(tr)):
            assert cats == dls.vocab, f"Category missmatch between metrics cats ({cats}) and dls cats ({dls.vocab})"
            learn = train_utils.prepare_learner(args, create_learner(args, dls, metrics)) if it == 0 else learn
            learn.dls = dls
            train_utils.setup_tensorboard(learn, args.exp_logdir, run, metrics_names)
            with learn.distrib_ctx():
                learn.fine_tune(args.epochs)
            save_path = os.path.join(args.exp_logdir, f'{common.zero_pad(fold, args.nfolds)}_{run}_model')
            train_utils.save_learner(learn, is_fp16=(not args.full_precision), save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai classification")
    train_utils.common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34'})
    train_utils.common_img_args(parser, pdef={'--input-size': 256})
    args = parser.parse_args()

    train_utils.prepare_training(args, image_data=True)

    common.time_method(main, args)
