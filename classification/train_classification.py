import os
import sys
from functools import partial

import numpy as np

import torch
import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils, train_utils_img
from general.PatchExtractor import PatchExtractor
import classification.classification_utils as classif_utils


class ImageClassificationTrainer(train_utils_img.ImageTrainer):
    def load_items(self, path):
        images = common.list_files_in_dirs(path, full_path=True, posix_path=True)
        return np.array(images), np.array([classif_utils.get_image_cls(img_path) for img_path in images])

    def get_full_img_cls(self, img_path): return classif_utils.get_image_cls(img_path)

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        cat_perf = partial(classif_utils.cls_perf, cats=self.args.cats)
        signature = f'{self.get_cat_metric_name(perf_fn, cat)}(inp, targ)'
        code = f"def {signature}: return cat_perf(train_utils.{perf_fn}, inp, targ, {cat_id}).to(inp.device)"
        exec(code, {"cat_perf": cat_perf, 'train_utils': train_utils}, metrics_fn)

    def compute_conf_mat(self, targs, preds): return classif_utils.conf_mat(targs, preds, self.args.cats)

    def create_dls(self, tr, val, bs, size):
        tr, val = map(lambda x: tuple(map(np.ndarray.tolist, x)), (tr, val))
        return self.create_dls_from_lst((fv.ImageBlock, fv.CategoryBlock), tr, val, bs, size)

    def create_learner(self, dls):
        metrics = list(self.cust_metrics.values()) + [fv.accuracy]  # for early stop callback
        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained(self.args.model)
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            model_splitter = lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params)
            learn = fv.Learner(dls, model, metrics=metrics, splitter=model_splitter)
        else:
            learn = fv.cnn_learner(dls, getattr(fv, self.args.model), metrics=metrics)
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, preds):
        wl_items, labels = wl_items_with_labels
        preds = np.array(self.args.cats)[preds.numpy()]
        corr = labels != preds
        changelog = "\n".join([f'{i};{ic};{p}' for i, ic, p in zip(wl_items[corr], labels[corr], preds[corr])])
        labels[corr] = np.array([p for p in preds[corr]])
        return (wl_items, labels), changelog

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=3)


def main(args):
    classif = ImageClassificationTrainer(args, stratify=True, full_img_sep=PatchExtractor.SEP)
    classif.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256}
    parser = train_utils_img.ImageTrainer.get_argparser(desc="Fastai image classification", pdef=defaults)
    args = parser.parse_args()

    args.exp_name = "img_classif_"+args.exp_name
    if args.cats is None:
        args.cats = common.list_dirs(os.path.join(args.data, args.sl_train), full_path=False)

    train_utils_img.ImageTrainer.prepare_training(args)

    common.time_method(main, args, prepend=f"GPU {args.proc_gpu} proc: ")

