#!/usr/bin/env python

"""train_classification.py: Train classification model. Can be extended for more specific classification tasks."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)


import os
import sys
import collections
from functools import partial

import numpy as np
import sklearn.metrics as skm

import torch
import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import metrics, train_utils
from training.image_trainer import ImageTrainer
from general.PatchExtractor import PatchExtractor
import classification.classification_utils as classif_utils


class ImageClassificationTrainer(ImageTrainer):
    """Class used to train image classification model. Can be extended for specific more tasks."""
    @staticmethod
    def get_argparser(desc="Fastai image classification trainer arguments", pdef=dict(), phelp=dict()):
        """Creates classification argparser
        :param desc: str, argparser description
        :param pdef: dict, default arguments values
        :param phelp: dict, argument help strings
        :return: argparser
        """
        psr = super(ImageClassificationTrainer, ImageClassificationTrainer).get_argparser(desc, pdef, phelp)
        psr.add_argument('--oversample', action='store_true', help="Uses weighted dls based on class distrib")
        return psr

    @staticmethod
    def get_exp_logdir(args, custom=""):
        """Creates experiment log dir
        :param args: command line arguments
        :param custom: custom string to be added in experiment log dirname
        :return: str, experiment log dir
        """
        d = ''
        if args.oversample: d += '_oversample'
        custom = f'{d}_{custom}'
        return super(ImageClassificationTrainer, ImageClassificationTrainer).get_exp_logdir(args, custom=custom)

    @staticmethod
    def prepare_training(args):
        """Sets up training, check args validity.
        :param args: command line args
        """
        args.exp_name = "img_classif_"+args.exp_name
        if args.cats is None: args.cats = common.list_dirs(os.path.join(args.data, args.sl_train[0]), full_path=False)
        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(args.logdir, ImageClassificationTrainer.get_exp_logdir(args))
        super(ImageClassificationTrainer, ImageClassificationTrainer).prepare_training(args)

    def __init__(self, args, stratify=True, full_img_sep=PatchExtractor.SEP, **kwargs):
        """Creates classification trainer
        :param args: command line args
        :param stratify: bool, whether to stratify data when splitting
        :param full_img_sep: str, when splitting the data train/valid, will make sure to split on full images
        """
        super().__init__(args, stratify, full_img_sep, **kwargs)

    def load_items(self, set_dir):
        """Loads training items from directory
        :param set_dir: str, directory containing training items
        :return: tuple of fastai L lists, first list contain items, second list contains labels
        """
        path = os.path.join(self.args.data, set_dir)
        images = common.list_files_in_dirs(path, full_path=True, posix_path=True)
        return fv.L(images), fv.L([classif_utils.get_image_cls(img_path) for img_path in images])

    def get_full_img_cls(self, img_path):
        """Get label of full image
        :param img_path: str, full image path
        :return: str, label
        """
        return classif_utils.get_image_cls(img_path)

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        """Generates metrics functions for the individual categories
        :param perf_fn: function, metrics to apply, e.g. precision
        :param cat_id: int, category id for which to compute metric
        :param cat: str, label of category for which to compute metric
        :param metrics_fn: dict, contains generated metrics function names as keys and metrics functions as values
        """
        cat_perf = partial(classif_utils.cls_perf, cats=self.args.cats)
        signature = f'{self.get_cat_metric_name(perf_fn, cat)}(inp, targ)'
        code = f"def {signature}: return cat_perf(metrics.{perf_fn}, inp, targ, {cat_id}).to(inp.device)"
        exec(code, {"cat_perf": cat_perf, 'metrics': metrics}, metrics_fn)

    def ordered_test_perfs_per_cats(self):
        """Returns custom metrics ordered per category order and metrics type
        :return: list of tuples, each tuple is the list of category metrics with the metrics function name
        """
        all_cats = self.get_cats_with_all()
        return [([self.get_cat_metric_name(f, c) for c in all_cats], f) for f in self.args.metrics_base_fns]

    def compute_conf_mat(self, targs, decoded):
        """Compute confusion matrix from predictions
        :param targs: tensor, ground truth, size B
        :param decoded: tensor, decoded predictions, size B
        :return: tensor, confusion metrics N x N (with N categories)
        """
        return classif_utils.conf_mat(targs, decoded, self.args.cats)

    def create_dls(self, tr, val, bs, size):
        """Create classification dataloaders
        :param tr: tuple of fastai lists, (items, labels), training split
        :param val: tuple of fastai lists, (items, labels), validation split
        :param bs: int, batch size
        :param size: int, input size
        :return: train/valid dataloaders
        """
        blocks = fv.ImageBlock, fv.CategoryBlock(vocab=self.args.cats)
        if self.args.oversample:
            kwargs = {'dl_type': fv.WeightedDL, 'wgts': self.get_train_items_weights(list(zip(*tr))),
                      'dl_kwargs': [{}, {'cls': fv.TfmdDL}]}
        else: kwargs = {}
        return self.create_dls_from_lst(blocks, tr, val, bs, size, **kwargs)

    def get_class_weights(self, train_items):
        """Compute class weights based on train items labels frequency
        :param train_items: tuple of fastai lists, (items, labels)
        :return: tensor, weight for each categories
        """
        counts = collections.Counter([x[1] for x in train_items])
        class_counts = np.array([counts[c] for c in self.args.cats])
        return torch.FloatTensor(class_counts.max() / class_counts)

    def get_train_items_weights(self, train_items):
        """Computes each train item weight based on class weights
        :param train_items: tuple of fastai lists, (items, labels)
        :return:
        """
        labels, class_weights = [x[1] for x in train_items], self.get_class_weights(train_items).numpy()
        return class_weights[fv.CategoryMap(self.args.cats).map_objs(labels)]

    def create_learner(self, dls):
        """Creates learner with callbacks
        :param dls: train/valid dataloaders
        :return: learner
        """
        learn_kwargs = self.get_learner_kwargs(dls)
        metrics = list(self.cust_metrics.values())
        metrics += [fv.Precision(average='micro'), fv.Recall(average='micro')] + [fv.accuracy]  # acc for early stop cb
        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained(self.args.model)
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            msplitter = lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params)
            learn = fv.Learner(dls, model, metrics=metrics, splitter=msplitter, **learn_kwargs)
        else:
            model = getattr(fv, self.args.model)
            learn = fv.cnn_learner(dls, model, metrics=metrics, **learn_kwargs)
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, decoded):
        """Corrects weakly labeled data with model predictions
        :param wl_items_with_labels: tuple of fastai lists, (items, labels)
        :param decoded: tensor, decoded predictions
        :return: tuple of fastai lists, (items, corrected labels)
        """
        wl_items, labels = wl_items_with_labels[0], np.array(wl_items_with_labels[1])
        decoded = np.array(self.args.cats)[decoded.numpy()]
        corr = np.array(labels) != decoded
        labels[corr] = np.array([p for p in decoded[corr]])
        return wl_items, fv.L(labels.tolist())

    def early_stop_cb(self, monitor='accuracy', min_delta=0.01, patience=3):
        """Creates early stopping callback
        :param monitor: str, which metric to monitor
        :param min_delta: float, min difference
        :param patience: int, how many epochs before stopping
        :return: Early stopping callback
        """
        return EarlyStoppingCallback(monitor=monitor, min_delta=min_delta, patience=patience)

    def compute_metrics(self, interp):
        """Apply metrics functions on test set predictions
        :param interp: namespace with predictions, targs, decoded preds, test set predictions
        :return: same namespace but with metrics results dict
        """
        interp = super().compute_metrics(interp)
        d, t = fv.flatten_check(interp.decoded, interp.targs)
        print(skm.classification_report(t, d, labels=list(interp.dl.vocab.o2i.values()),
                                        target_names=[str(v) for v in interp.dl.vocab]))
        return interp


def main(args):
    """Creates classification trainer
    :param args: command line args
    """
    classif = ImageClassificationTrainer(args)
    classif.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256}
    parser = ImageClassificationTrainer.get_argparser(desc="Fastai image classification", pdef=defaults)
    args = parser.parse_args()

    ImageClassificationTrainer.prepare_training(args)

    common.time_method(main, args)

