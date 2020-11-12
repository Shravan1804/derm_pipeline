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


class ImageClassificationTrainer(train_utils.ImageTrainer):
    def get_items(self):
        sl_images = common.list_files_in_dirs(self.get_data_path(), full_path=True, posix_path=True)
        if self.args.use_wl:
            wl_path = self.get_data_path(weak_labels=True)
            wl_images = common.list_files_in_dirs(wl_path, full_path=True, posix_path=True)
        return sl_images, wl_images if self.args.use_wl else None

    def get_metrics(self):
        metrics_fn = {}
        device = f"'cuda:{self.args.gpu}'"
        for cat_id, cat in zip([*range(len(self.args.cats))] + [None], self.args.cats + ["all"]):
            for perf_fn in self.perf_fns:
                code = f"def {cat}_{perf_fn}(inp, targ):" \
                       f"return cls_perf(common.{perf_fn}, inp, targ, {cat_id}, {self.args.cats}).to({device})"
                exec(code, {"cls_perf": classif_utils.cls_perf, 'common': common}, metrics_fn)
        return list(metrics_fn.keys()), list(metrics_fn.values())

    def create_dls(self, tr, val, bs, size):
        return self.create_dls_from_lst((fv.ImageBlock, fv.CategoryBlock(vocab=self.args.cats)),
                                        tr.tolist(), val.tolist(), fv.parent_label, bs, size)

    def create_learner(self, dls):
        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = fd.rank0_first(lambda: EfficientNet.from_pretrained(self.args.model))
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            learn = fv.Learner(dls, model, metrics=self.metrics_fn,
                               splitter=lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params))
        else:
            learn = fd.rank0_first(lambda: fv.cnn_learner(dls, getattr(fv, self.args.model), metrics=self.metrics_fn))
        return self.prepare_learner(learn)


def create_dls(bs, size, tr, val, args):
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
        learn = fv.Learner(dls, model, metrics=metrics,
                           splitter=lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params))
    else:
        learn = fd.rank0_first(lambda: fv.cnn_learner(dls, getattr(fv, args.model, None), metrics=metrics))
    return train_utils.prepare_learner(args, learn)


def get_classif_imgs(args):
    sl_images = common.list_files_in_dirs(train_utils.get_data_path(args), full_path=True, posix_path=True)
    if args.use_wl:
        wl_path = train_utils.get_data_path(args, weak_labels=True)
        wl_images = common.list_files_in_dirs(wl_path, full_path=True, posix_path=True)
    return sl_images, wl_images if args.use_wl else None


def correct_wl(args):
    pass


def main(args):
    #print("Running script with args:", args)
    #metrics_names, metrics_fn = get_classif_metrics(args)
    #get_splits, get_dls = train_utils.get_data_fn(args, full_img_sep=PatchExtractor.SEP, stratify=True)
    #sl_images, wl_images = get_classif_imgs(args)

    #train_utils.train_model(args, get_splits, sl_images, get_dls, create_dls, create_learner,
    #                        metrics_names, metrics_fn, correct_wl, wl_images)

    classif = ImageClassificationTrainer(args, stratify=True, full_img_sep=PatchExtractor.SEP)
    classif.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai classification")
    train_utils.common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34'})
    train_utils.common_img_args(parser, pdef={'--input-size': 256})
    args = parser.parse_args()

    train_utils.prepare_training(args, image_data=True)

    if args.cats is None:
        args.cats = common.list_dirs(train_utils.get_data_path(args), full_path=False)

    common.time_method(main, args, prepend=f"GPU {args.gpu} proc: ")
