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
from collections import OrderedDict

import numpy as np
import sklearn.metrics as skm

import torch
import fastai.vision.all as fv
from fastai.vision.all import *
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

    def get_image_cls(self, img_path):
        """Assumes image class is its directory name
        :param img_path: str or Path, image path
        :return: str, class name of image
        """
        return os.path.basename(os.path.dirname(img_path)) if type(img_path) is str else str(img_path.parent.name)

    def load_items(self, set_dir):
        """Loads training items from directory
        :param set_dir: str, directory containing training items
        :return: tuple of fastai L lists, first list contain items, second list contains labels
        """
        path = os.path.join(self.args.data, set_dir)
        images = common.list_files_in_dirs(path, full_path=True, posix_path=True)
        return fv.L(images), fv.L([self.get_image_cls(img_path) for img_path in images])

    def get_full_img_cls(self, img_path):
        """Get label of full image (useful when working with patches)
        :param img_path: str, full image path
        :return: str, label
        """
        return self.get_image_cls(img_path)

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        """Generates metrics functions for the individual categories
        :param perf_fn: function, metrics to apply, e.g. precision
        :param cat_id: int, category id for which to compute metric
        :param cat: str, label of category for which to compute metric
        :param metrics_fn: dict, contains generated metrics function names as keys and metrics functions as values
        """
        cat_perf = partial(classif_utils.cls_perf, cats=self.args.cats)
        signature = f'{self.get_cat_metric_name(perf_fn, cat)}(inp, targ, prm=dict())'
        code = f"def {signature}: return cat_perf(metrics.{perf_fn}, inp, targ, {cat_id}, precomp=prm).to(inp.device)"
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
        if self.args.wandb:
            import wandb
            wandb.log({
                "Test/Conf_mat":
                wandb.plot.confusion_matrix(probs=None,
                                            y_true=targs.tolist(),
                                            preds=decoded.tolist(),
                                            class_names=self.args.cats)
            })
        return classif_utils.conf_mat(fv.TensorBase(targs),
                                      fv.TensorBase(decoded), self.args.cats)

    def customize_datablock(self):
        """Provides experiment specific kwargs for DataBlock
        :return: dict with argnames and argvalues
        """
        return {'blocks': (fv.ImageBlock, fv.CategoryBlock(vocab=self.args.cats))}

    def create_dls(self, tr, val, bs, size):
        """Create classification dataloaders
        :param tr: tuple of fastai lists, (items, labels), training split
        :param val: tuple of fastai lists, (items, labels), validation split
        :param bs: int, batch size
        :param size: int, input size
        :return: train/valid dataloaders
        """
        kwargs = {}
        if self.args.oversample:
            kwargs = {'dl_type': fv.WeightedDL, 'wgts': self.get_train_items_weights(list(zip(*tr))),
                      'dl_kwargs': [{}, {'cls': fv.TfmdDL}]}
        return super().create_dls(tr, val, bs, size, **kwargs)

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

    def customize_learner(self, dls):
        """Provides experiment specific kwargs for Learner
        :return: kwargs dict
        """
        kwargs = super().customize_learner(dls)
        kwargs['metrics'].extend([
            fv.Precision(average='macro'),
            fv.Recall(average='macro'),
            fv.BalancedAccuracy(),
            fv.F1Score(average='macro'),
            fv.RocAuc(average='macro'),
            fv.MatthewsCorrCoef(),
        ])
        return kwargs

    def create_learner(self, dls):
        """Creates learner with callbacks
        :param dls: train/valid dataloaders
        :return: learner
        """
        learn_kwargs = self.customize_learner(dls)
        callbacks = []
        if self.args.mixup:
            callbacks += [MixUp()]
        if self.args.wandb:
            import wandb
            from fastai.callback.wandb import WandbCallback
            callbacks += [WandbCallback(log='all', log_preds=False)]
            # update the name of the wandb run
            run_name = f'{self.args.model}-{wandb.run.name}'
            wandb.run.name = run_name
            wandb.run.save()

        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained(self.args.model)
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            msplitter = lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params)
            learn = fv.Learner(dls, model, splitter=msplitter, cbs=callbacks, **learn_kwargs)
        elif "ssl" in self.args.model:
            from self_supervised_dermatology import Embedder
            ssl_model = self.args.model.replace('ssl_', '')
            model, info = Embedder.load_pretrained(ssl_model, return_info=True)
            print(f'Loaded pretrained SSL model: {info}')
            model = torch.nn.Sequential(OrderedDict([
                ('backbone', model),
                ('flatten', torch.nn.Flatten()),
                ('fc', classif_utils.LinearClassifier(info.out_dim, dls.c)),
            ]))
            learn = fv.Learner(dls, model, cbs=callbacks, **learn_kwargs)
        else:
            model = getattr(fv, self.args.model)
            learn = fv.cnn_learner(dls, model, cbs=callbacks, **learn_kwargs)
        # also log the training metrics
        # then train + valid metrics are reported
        learn.recorder.train_metrics = True
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

    def precompute_metrics(self, interp):
        """Precomputes values useful to speed up metrics calculations (e.g. class TP TN FP FN)
        :param interp: namespace with predictions, targets, decoded predictions
        :return: dict, with precomputed values. Keys are category id.
        """
        d, t = fv.flatten_check(interp.decoded, interp.targs)
        return {cid: metrics.get_cls_TP_TN_FP_FN(t == cid, d == cid) for cid in self.get_cats_idxs()}

    def compute_metrics(self, interp, print_summary=False, with_ci=True):
        """Apply metrics functions on test set predictions
        :param interp: namespace with predictions, targs, decoded preds, test set predictions
        :return: same namespace but with metrics results dict
        """
        interp = super().compute_metrics(interp, print_summary, with_ci)
        targs, dec = interp.targs.flatten(), interp.decoded.flatten()
        interp.metrics['cm'] = classif_utils.conf_mat(targs, dec, self.args.cats)
        if print_summary:
            print(skm.classification_report(targs, dec, target_names=self.args.cats, labels=self.get_cats_idxs()))
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
