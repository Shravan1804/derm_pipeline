import os
import sys
import argparse

import torch
import fastai.vision.all as fv
import fastai.distributed as fd
from fastai.callback.tracker import EarlyStoppingCallback

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
        for cat_id, cat in zip([*range(len(self.args.cats))] + [None], self.args.cats + [self.ALL_CATS]):
            for perf_fn in self.BASIC_PERF_FNS:
                code = f"def {cat}_{perf_fn}(inp, targ):" \
                       f"return cls_perf(common.{perf_fn}, inp, targ, {cat_id}, {self.args.cats}).to({device})"
                exec(code, {"cls_perf": classif_utils.cls_perf, 'common': common}, metrics_fn)
        return list(metrics_fn.keys()), list(metrics_fn.values())

    def create_dls(self, tr, val, bs, size):
        return self.create_dls_from_lst((fv.ImageBlock, fv.CategoryBlock(vocab=self.args.cats)),
                                        tr.tolist(), val.tolist(), fv.parent_label, bs, size)

    def create_learner(self, dls):
        metrics = self.cats_metrics_fn + [fv.accuracy]
        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = fd.rank0_first(lambda: EfficientNet.from_pretrained(self.args.model))
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            learn = fv.Learner(dls, model, metrics=self.cats_metrics_fn,
                               splitter=lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params))
        else:
            learn = fd.rank0_first(lambda: fv.cnn_learner(dls, getattr(fv, self.args.model), metrics=metrics))
        return self.prepare_learner(learn)

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=3)


def main(args):
    classif = ImageClassificationTrainer(args, stratify=True, full_img_sep=PatchExtractor.SEP)
    classif.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai classification")
    train_utils.common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34'})
    train_utils.common_img_args(parser, pdef={'--input-size': 256})
    args = parser.parse_args()

    train_utils.prepare_training(args, image_data=True)

    if args.cats is None:
        data_path = args.data if not args.use_wl else os.path.join(args.data, args.sl_train)
        args.cats = common.list_dirs(data_path, full_path=False)

    common.time_method(main, args, prepend=f"GPU {args.gpu} proc: ")
