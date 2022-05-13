#!/usr/bin/env python

"""image_trainer.py: Image trainer class. Extendable to any training tasks (classification, segmentation, etc)."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

import re
import os
import sys
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import fastai.vision.all as fv
import fastai.distributed as fd   # needed for fastai multi gpu

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, common_plot as cplot, common_img as cimg, crypto
from training import train_utils, custom_losses as closs
from training.base_trainer import FastaiTrainer
import training.fastai_monkey_patches as fmp


class ImageTrainer(FastaiTrainer):
    """Image training class. Should be extended to fulfill task specific requirements."""
    @staticmethod
    def get_argparser(desc="Fastai image trainer arguments", pdef=dict(), phelp=dict()):
        """Creates image trainer argparser
        :param desc: str, argparser description
        :param pdef: dict, default arguments values
        :param phelp: dict, argument help strings
        :return: argparser
        """
        parser = super(ImageTrainer, ImageTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument('--data', type=str, required=True, help="Root dataset dir")

        parser.add_argument('--cats', type=str, nargs='+', default=pdef.get('--cats', None),
                            help=phelp.get('--cats', "Object categories"))

        parser.add_argument('--input-size', default=pdef.get('--input-size', 512), type=int,
                            help=phelp.get('--input-size', "Model input will be resized to this value"))
        parser.add_argument('--progr-size', action='store_true',
                            help=phelp.get('--progr-size', "Applies progressive resizing"))
        parser.add_argument('--size-facts', default=pdef.get('--size-facts', [.5, .75, 1]), nargs='+', type=float,
                            help=phelp.get('--size-facts', 'Increase progressive size factors'))

        parser.add_argument('--label-smoothing-loss', action='store_true', help="For unsure labels")
        parser.add_argument('--focal-loss', action='store_true', help="In imbalanced ds, favor hard cases")
        parser.add_argument('--focal-loss-plus-ce-loss', action='store_true', help="Focal loss + cross entropy loss")
        parser.add_argument('--focal-loss-plus-dice-focal-loss', action='store_true', help="Focal loss + dice focal loss")
        parser.add_argument('--ce-loss', action='store_true', help="cross entropy loss")
        parser.add_argument('--weighted-loss', action='store_true', help="Uses weighted loss based on class distrib")
        return parser

    @staticmethod
    def get_exp_logdir(args, custom=""):
        """Creates experiment log dir
        :param args: command line arguments
        :param custom: custom string to be added in experiment log dirname
        :return: str, experiment log dir
        """
        d = f'_input{args.input_size}'
        if args.label_smoothing_loss: d += '_smooth-loss'
        elif args.focal_loss: d += '_focal-loss'
        elif args.focal_loss_plus_ce_loss: d += '_focal-plus-ce-loss'
        elif args.focal_loss_plus_dice_focal_loss: d += '_focal-plus-dice_focal-loss'
        elif args.ce_loss: d += '_ce-loss'
        if args.weighted_loss: d += '_weighted-loss'
        if args.progr_size: d += f'_progr-size{"_".join(map(str, args.size_facts))}'
        custom = f'{d}_{custom}'
        return super(ImageTrainer, ImageTrainer).get_exp_logdir(args, custom=custom)

    @staticmethod
    def prepare_training(args):
        """Sets up training, checks if provided args valid, initiates distributed mode if needed
        :param args: command line args
        """
        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(args.logdir, ImageTrainer.get_exp_logdir(args))
        super(ImageTrainer, ImageTrainer).prepare_training(args)

    def __init__(self, args, stratify, full_img_sep, **kwargs):
        """Creates base trainer
        :param args: command line args
        :param stratify: bool, whether to stratify data when splitting
        :param full_img_sep: str, when splitting the data train/valid, will make sure to split on full images
        """
        self.ALL_CATS = 'all'   # for cat macro averaging
        self.full_img_sep = full_img_sep
        self.loss_axis = -1     # predictions argmax axis to get decoded preds
        super().__init__(args, stratify)

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        """Generates metrics functions for the individual classes
        :param perf_fn: function, metrics to apply, e.g. precision
        :param cat_id: int, category id for which to compute metric
        :param cat: str, label of category for which to compute metric
        :param metrics_fn: dict, contains generated metrics function names as keys and metrics functions as values
        """
        raise NotImplementedError

    def get_class_weights(self, dls):
        """Compute class weights based on train items labels frequency
        :param train_items: tuple of fastai lists, (items, labels)
        :return: tensor, weight for each categories
        """
        raise NotImplementedError

    def get_loss_fn(self, dls):
        """Select loss function based on command line args
        :param dls: dataloaders, used to compute class weights
        :return:
        """
        class_weights = self.get_class_weights(dls.train_ds.items).to(dls.device) if self.args.weighted_loss else None
        if self.args.label_smoothing_loss:
            loss_func = fmp.FixedLabelSmoothingCrossEntropyFlat(weight=class_weights, axis=self.loss_axis)
        elif self.args.focal_loss:
            loss_func = fmp.FixedFocalLossFlat(weight=class_weights, axis=self.loss_axis)
        elif self.args.focal_loss_plus_ce_loss:
            loss_func = closs.FocalLossPlusCElossFlat(weight=class_weights, axis=self.loss_axis)
        elif self.args.focal_loss_plus_dice_focal_loss:
            loss_func = closs.FocalLossPlusFocalDiceLoss(weight=class_weights, axis=self.loss_axis)
        elif self.args.ce_loss:
            loss_func = fv.CrossEntropyLossFlat(weight=class_weights, axis=self.loss_axis)
        else:
            loss_func = None
        return loss_func

    def customize_learner(self, dls):
        """Provides experiment specific kwargs for Learner
        :return: kwargs dict
        """
        kwargs = super().customize_learner()
        kwargs['loss_func'] = self.get_loss_fn(dls)
        kwargs['metrics'] = list(self.cust_metrics.values())
        return kwargs

    def get_cats_idxs(self):
        """:return category indices"""
        return list(range(len(self.args.cats)))

    def get_cats_with_all(self):
        """:return list with all cats code and categories"""
        return [self.ALL_CATS, *self.args.cats]

    def get_cat_metric_name(self, perf_fn, cat):
        """Gets category specific metric name
        :param perf_fn: str, base metric name (e.g. precision)
        :param cat: str, category name
        :return str, category specific metric function name
        """
        return f'{perf_fn}_{cat}'

    def prepare_custom_metrics(self):
        """Creates category specific metrics for all provided base metrics
        :return dict, keys the function names, values the functions"""
        metrics_fn = {}
        for perf_fn in self.args.metrics_base_fns:
            for cat_id, cat in zip([None, *self.get_cats_idxs()], self.get_cats_with_all()):
                self.create_cats_metrics(perf_fn, cat_id, cat, metrics_fn)
        return metrics_fn

    def print_metrics_summary(self, metrics):
        """Prints summary of metrics
        :param metrics, dict with metrics results
        :return str of printed txt
        """
        metric_names, agg_keys = zip(*self.ordered_test_perfs_per_cats())
        s = 'category;' + ';'.join(agg_keys) + '\n'
        for cid, cat in enumerate(self.get_cats_with_all()):
            mns = [mns[cid] for mns in metric_names]
            res = [f"{metrics[mn]:.0%}" for mn in mns if mn in metrics]
            s += cat + ';' + ';'.join([f"{r:.0%}" for r in metrics[mn]]) + '\n'
        print(s)
        return s

    def plot_custom_metrics(self, ax, agg_perf, show_val, title=None):
        """Plots aggregated metrics results.
        :param ax: axis
        :param agg_perf: dict, fold aggregated metrics results
        :param show_val: bool, print values on plot
        :param title: str, plot title
        """
        ax.axis('on')
        # if p in mn because mn may be variation of perfs: e.g. in segm perf to ignore bg (precision, precision_no_bg)
        bar_perf = {mn: cat_mres for p in self.args.metrics_base_fns for mn, cat_mres in agg_perf.items() if p in mn
                    and self.PERF_CI not in mn}
        bar_cats = self.get_cats_with_all()
        cplot.grouped_barplot_with_err(ax, bar_perf, bar_cats, xlabel='Classes', show_val=show_val, title=title)

    def aggregate_test_performance(self, folds_res):
        """Merges metrics over different folds, computes mean and std.
        Then groups categories metric results per metrics type (e.g. all precision together)
        :param folds_res: list of metrics results over folds
        :return: dict with merged metrics results
        """
        agg = super().aggregate_test_performance(folds_res)
        for mns, agg_key in self.ordered_test_perfs_per_cats():
            # if mn in agg: in case some metrics fns were not computed
            mns = [mn for mn in mns if mn in agg]
            if len(mns) > 0:
                agg[agg_key] = tuple(np.stack(s) for s in zip(*[agg.pop(mn) for mn in mns if mn in agg]))
        return agg

    def ordered_test_perfs_per_cats(self):
        """Returns list of tuples, each tuple is the list of metrics names (following cat order) with corresponding
        aggregated key"""
        raise NotImplementedError

    def plot_save_path(self, test_path, run, show_val, custom=""):
        """Prepare performance plot save path.
        :param test_path: str, test identifier
        :param run: str, run identifier
        :param show_val: bool, if values were printed on plot
        :param custom: str, custom string
        :return str, path of plot
        """
        return os.path.join(test_path, f'{run}{custom}{"_show_val" if show_val else ""}.jpg')

    def plot_test_performance(self, test_path, run, agg_perf):
        """Plots aggregated performance
        :param test_path: str, test identifier
        :param run: str, run identifier
        :param agg_perf: dict, aggregated performance
        """
        show_val = not self.args.no_plot_val
        save_path = self.plot_save_path(test_path, run, show_val)
        fig, axs = cplot.new_fig_with_axs(1, 2, self.args.test_figsize)
        self.plot_custom_metrics(axs[0], agg_perf, show_val)
        cplot.plot_confusion_matrix(axs[1], agg_perf['cm'], self.args.cats)
        fig.tight_layout(pad=.2)
        plt.savefig(save_path, dpi=400)

    def tensorboard_cb(self, run_info):
        """Creates tensorboard (TB) callback
        :param run_info: str, run identifier for logs
        :return: TB callback
        """
        tbdir = common.maybe_create(self.args.exp_logdir, 'tb_logs')
        return train_utils.ImageTBCb(tbdir, run_info, self.cust_metrics.keys(), self.ALL_CATS)

    def load_items(self, set_dir):
        """Loads training items from directory
        :param set_dir: str, directory containing training items
        :return: tuple of fastai L lists, first list contain items, second list contains labels
        """
        raise NotImplementedError

    def load_multiple_items_sets(self, set_locs, merged):
        """Loads all sets of items. Merge them if specified. Returns
        :param set_locs: list of str, directories from which data should be loaded
        :param merged: bool, whether the different data sets should be merged
        :return: list of tuples of items and labels lists
        """
        if set_locs is None:
            items_with_cls = fv.L(), fv.L()
            return items_with_cls if merged else (None, items_with_cls)
        else:
            items_sets = [(cl, self.load_items(cl)) for cl in set_locs]
            # cannot use map_zip since it is the starmap which may break with single lists (in object detec)
            return fv.L(map(fv.L, fv.L([t[1] for t in items_sets]).zip())).map(fv.L.concat) if merged else items_sets

    def get_test_items(self, merged=True):
        """Return test items
        :param merged: bool, whether to merge all test sets together
        :return: list of tuples of items and labels lists, one for each test set
        """
        return self.load_multiple_items_sets(self.args.sl_tests, merged)

    def get_train_items(self, merged=True):
        """Return train items
        :param merged: bool, whether to merge all train sets together
        :return: tuple (sl, wl) of list of tuples of items and labels lists, one for each train set
        """
        sl, wl = self.args.sl_train, self.args.wl_train
        return self.load_multiple_items_sets(sl, merged), self.load_multiple_items_sets(wl, merged)

    def get_full_img_cls(self, img_path):
        """Should be reimplemented if stratification needed. Currently all full images will have same class
        :param img_path: str, full image path
        :return: int, label
        """
        if self.stratify: raise NotImplementedError
        else: return 1  # when stratified splits are not needed, consider all images have the same cls

    def get_patch_full_img(self, patch):
        """Get full image filename from patch path
        :param patch: str, patch path
        :return: str, full image filename
        """
        file, ext = os.path.splitext(os.path.basename(patch))
        return f'{file.split(self.full_img_sep)[0] if self.full_img_sep in file else file}{ext}'

    def get_full_img_dict(self, images, targets):
        """Match all patches with their full full image
        :param images: list of patches
        :param targets: list of corresponding labels
        :return: dict, key is a tuple (full image, label), value is a list of tuples (patch, label)
        """
        full_images_dict = defaultdict(fv.L)
        for img, tgt in zip(images, targets):
            full_images_dict[(self.get_patch_full_img(img), self.get_full_img_cls(img))].append((img, tgt))
        return full_images_dict

    def split_data(self, items, items_cls):
        """This overloading makes sure that patches from the same image do not leak between train/val sets
        :param items: list, train items to be splitted
        :param items_cls: list labels of items
        :yield: tuple, fold id, train items and labels, valid items and labels
        """
        fi_dict = self.get_full_img_dict(items, items_cls)
        for fold, tr, val in super().split_data(*fv.L(fi_dict.keys()).map_zip(fv.L)):
            tr = fv.L([fi_dict[fik] for fik in zip(*tr)]).concat().map_zip(fv.L)
            if not self.args.full_data:    # if not True, then valid are patches from test set => no possible leaks
                val = fv.L([fi_dict[fik] for fik in zip(*val)]).concat().map_zip(fv.L)
            yield fold, tr, val

    def load_image_item(self, item, load_im_array=False):
        """Load image item, decrypts if needed
        :param item: str, path of image
        :param load_im_array: bool, optional, force image array loading in memory
        :return: array of image if data is encrypted else image path
        """
        if type(item) is np.ndarray: return item    # image/mask is already loaded as np array
        elif self.args.encrypted: return crypto.decrypt_img(item, self.args.ckey)
        elif load_im_array: return cimg.load_img(item)
        else: return item

    def create_data_augm_tfms(self, size):
        """Creates image augmentation transforms
        :param size: int, final size of images
        :return: list of transform operations
        """
        tfms = fv.aug_transforms(size=size, flip_vert=True, max_rotate=45)
        if not self.args.no_norm:
            tfms.append(fv.Normalize.from_stats(*fv.imagenet_stats))
        return tfms

    def create_dls(self, tr, val, bs, size, **kwargs):
        """Create train/valid dataloaders
        :param blocks: Problem specific data blocks
        :param tr: tuple of train items and labels
        :param val: tuple of valid items and labels
        :param bs: int, batch size
        :param size: int, input side size, will be resized as square
        :param get_x: function, additional step to get item
        :param get_y: function, additional step to get target
        :param kwargs: dict, dataloaders kwargs
        :return: train/valid dataloaders
        """
        datablock_args = {
            'get_items': lambda s: list(zip(*tuple(v+t for v, t in zip(val, tr)))),
            'get_x': fv.Pipeline([fv.ItemGetter(0), self.load_image_item]),
            'get_y': fv.Pipeline([fv.ItemGetter(1)]),
            'splitter': fv.IndexSplitter(list(range(len(val[0])))),
            'item_tfms': fv.Resize(int(1.5*self.args.input_size), method=fv.ResizeMethod.Squish),
            'batch_tfms': self.create_data_augm_tfms(size)
        }
        datablock_args.update(self.customize_datablock())
        d = fv.DataBlock(**datablock_args)
        if self.args.debug_dls:
            if train_utils.GPUManager.is_master_process(): d.summary(self.args.data, bs=bs)
            sys.exit()
        # set path args so that learner objects use it
        return d.dataloaders(self.args.data, path=self.args.exp_logdir, bs=bs, seed=self.args.seed, **kwargs)

    def maybe_progressive_resizing(self, tr, val, fold_suffix):
        """Creates dataloaders with progressive resizing if requested
        :param tr: tuple of train items and labels
        :param val: tuple of valid items and labels
        :param fold_suffix: str, fold identifier for run info
        :yield: tuple with iteration id, run info, dataloaders
        """
        if self.args.progr_size:
            input_sizes = [int(self.args.input_size * f) for f in self.args.size_facts]
            batch_sizes = [max(1, min(int(self.args.bs / f / f), len(tr[0])) // 2 * 2) for f in self.args.size_facts]
        else:
            input_sizes = [self.args.input_size]
            batch_sizes = [self.args.bs]

        for it, (bs, size) in enumerate(zip(batch_sizes, input_sizes)):
            run = f'__S{size}px_bs{bs}__{fold_suffix}'
            if self.args.progr_size:
                print(f"Progressive resizing {it + 1}/{len(batch_sizes)}: running {run}")
            yield it, run, self.create_dls(tr, val, bs, size)

    def train_procedure(self, tr, val, fold_suffix, run_prefix="", learn=None):
        """Training procedure
        :param tr: tuple of train items and labels
        :param val: tuple of valid items and labels
        :param fold_suffix: str, fold identifier for run info
        :param run_prefix: str, run indentifier for run info
        :param learn: learner object, optional, if not set creates new one
        :return: tuple with learn object and run info
        """
        for it, run, dls in self.maybe_progressive_resizing(tr, val, fold_suffix):
            run = f'{run_prefix}{run}'
            if it == 0 and learn is None: learn = fd.rank0_first(lambda: self.create_learner(dls))
            common.time_method(self.basic_train, run, learn, dls, text="Training model")
            common.time_method(self.evaluate_on_test_sets, learn, run, text="Test sets evaluation")
            train_utils.GPUManager.clean_gpu_memory(learn.dls)
        return learn, run

    def extract_run_params(self, run_info):
        """Extracts run parameters from run info: #repeat, progr_size?, size, bs, fold
        :param run_info: str, run info
        :return: namespace with all run params
        """
        regex = r"^(?:__R(?P<repeat>\d+)__)?__S(?P<progr_size>\d+)px_bs(?P<bs>\d+)____F(?P<fold>\d+)__.*$"
        run_params = SimpleNamespace(**re.match(regex, run_info).groupdict())
        run_params.progr_size = int(run_params.progr_size)
        run_params.bs = int(run_params.bs)
        return run_params

    def get_sorting_run_key(self, run_info):
        """Gets key from run info string allowing to aggregate runs results.
        :param run_info: str, run info string
        :return str, sorting key
        """
        return run_info.replace(f'__F{self.extract_run_params(run_info).fold}__', '')

    def load_learner_from_run_info(self, run_info, tr, val, mpath=None):
        """Create learner object from run information and items. If mpath set will load saved model weights.
        :param run_info: str, run information string e.g. bs, size, etc
        :param tr: tuple of train items and labels
        :param val: tuple of valid items and labels
        :param mpath: str, model weights path, optional
        :return learner object
        """
        run_params = self.extract_run_params(run_info)
        dls = self.create_dls(tr, val, run_params.bs, run_params.progr_size)
        learn = fd.rank0_first(lambda: self.create_learner(dls))
        if mpath is not None: train_utils.load_custom_pretrained_weights(learn.model, mpath)
        return learn

