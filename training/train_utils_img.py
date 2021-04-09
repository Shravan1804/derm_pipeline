import re
import os
import sys
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import fastai.vision.all as fv
import fastai.distributed as fd   # needed for fastai multi gpu
import fastai.callback.tensorboard as fc

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto
from training import train_utils, custom_losses as closs
import training.fastai_monkey_patches as fmp


class ImageTrainer(train_utils.FastaiTrainer):
    @staticmethod
    def get_argparser(desc="Fastai image trainer arguments", pdef=dict(), phelp=dict()):
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
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
        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(args.logdir, ImageTrainer.get_exp_logdir(args))
        super(ImageTrainer, ImageTrainer).prepare_training(args)

    def __init__(self, args, stratify, full_img_sep, **kwargs):
        self.ALL_CATS = 'all'
        self.full_img_sep = full_img_sep
        self.loss_axis = -1
        super().__init__(args, stratify)

    def compute_conf_mat(self, targs, preds): raise NotImplementedError

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn): raise NotImplementedError

    def get_class_weights(self, dls): raise NotImplementedError

    def get_loss_fn(self, dls):
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

    def get_learner_kwargs(self, dls):
        kwargs = super().get_learner_kwargs()
        kwargs['loss_func'] = self.get_loss_fn(dls)
        return kwargs

    def get_cats_with_all(self):
        return [self.ALL_CATS, *self.args.cats]

    def get_cat_metric_name(self, perf_fn, cat): return f'{perf_fn}_{cat}'

    def prepare_custom_metrics(self):
        metrics_fn = {}
        for perf_fn in self.args.metrics_fns:
            for cat_id, cat in zip([None, *range(len(self.args.cats))], self.get_cats_with_all()):
                self.create_cats_metrics(perf_fn, cat_id, cat, metrics_fn)
        return metrics_fn

    def plot_custom_metrics(self, ax, agg_perf, show_val, title=None):
        ax.axis('on')
        bar_perf = {mn: cat_mres for p in self.args.metrics_fns for mn, cat_mres in agg_perf.items() if p in mn}
        bar_cats = self.get_cats_with_all()
        common.grouped_barplot_with_err(ax, bar_perf, bar_cats, xlabel='Classes', show_val=show_val, title=title)

    def process_test_preds(self, interp):
        interp = super().process_test_preds(interp)
        interp.metrics['cm'] = self.compute_conf_mat(interp.targs, interp.decoded)
        return interp

    def aggregate_test_performance(self, folds_res):
        agg = super().aggregate_test_performance(folds_res)
        for mns, agg_key in self.ordered_test_perfs_per_cats():
            mns = [m for m in mns if m in self.cust_metrics]  # in case we did no compute metrics for all cats
            agg[agg_key] = tuple(np.stack(s) for s in zip(*[agg.pop(mn) for mn in mns]))
        return agg

    def ordered_test_perfs_per_cats(self): raise NotImplementedError

    def reorder_aggregated_test_perfs_as_cats(self, agg, **kwargs):
        # for each perf_fn, combine results of each cats
        for perf_fn in self.args.metrics_fns:
            # order metrics results on self.get_cats_with_all()
            mns = [self.get_cat_metric_name(perf_fn, cat, **kwargs) for cat in self.get_cats_with_all()]
            mns = [m for m in mns if m in self.cust_metrics]    # in case we do no compute metrics for all cats
            agg[perf_fn] = tuple(np.stack(s) for s in zip(*[agg.pop(mn) for mn in mns]))
        return agg

    def plot_test_performance(self, test_path, run, agg_perf):
        for show_val in [False, True]:
            save_path = os.path.join(test_path, f'{run}{"_show_val" if show_val else ""}.jpg')
            fig, axs = plt.subplots(1, 2, figsize=self.args.test_figsize)
            self.plot_custom_metrics(axs[0], agg_perf, show_val)
            common.plot_confusion_matrix(axs[1], agg_perf['cm'], self.args.cats)
            fig.tight_layout(pad=.2)
            plt.savefig(save_path, dpi=400)

    def tensorboard_cb(self, run_info):
        tbdir = common.maybe_create(self.args.exp_logdir, 'tb_logs')
        return ImageTBCb(tbdir, run_info, self.cust_metrics.keys(), self.ALL_CATS)

    def load_multiple_items_sets(self, set_locs, merged):
        """Loads all sets of items. Merge them if specified. Returns tuples of item and item_cls lists"""
        if set_locs is None:
            items_with_cls = fv.L(), fv.L()
            return items_with_cls if merged else (None, items_with_cls)
        else:
            items_sets = [(cl, self.load_items(cl)) for cl in set_locs]
            # cannot use map_zip since it is the starmap which may break with single lists (in object detec)
            return fv.L(map(fv.L, fv.L([t[1] for t in items_sets]).zip())).map(fv.L.concat) if merged else items_sets

    def get_test_items(self, merged=True): return self.load_multiple_items_sets(self.args.sl_tests, merged)

    def get_train_items(self, merged=True):
        sl, wl = self.args.sl_train, self.args.wl_train
        return self.load_multiple_items_sets(sl, merged), self.load_multiple_items_sets(wl, merged)

    def get_full_img_cls(self, img_path):
        if self.stratify: raise NotImplementedError
        else: return 1  # when stratified splits are not needed, consider all images have the same cls

    def get_patch_full_img(self, patch):
        file, ext = os.path.splitext(os.path.basename(patch))
        return f'{file.split(self.full_img_sep)[0] if self.full_img_sep in file else file}{ext}'

    def get_full_img_dict(self, images, targets):
        """Returns a dict with keys (full images, tgt) and values the lst of corresponding images."""
        full_images_dict = defaultdict(fv.L)
        for img, tgt in zip(images, targets):
            full_images_dict[(self.get_patch_full_img(img), self.get_full_img_cls(img))].append((img, tgt))
        return full_images_dict

    def split_data(self, items, items_cls):
        """This version of split data makes sure that patches from the same image do not leak between train/val sets"""
        fi_dict = self.get_full_img_dict(items, items_cls)
        for fold, tr, val in super().split_data(*fv.L(fi_dict.keys()).map_zip(fv.L)):
            tr = fv.L([fi_dict[fik] for fik in zip(*tr)]).concat().map_zip(fv.L)
            if self.args.valid_size > 0:    # if not True, then valid are patches from test set => no possible leaks
                val = fv.L([fi_dict[fik] for fik in zip(*val)]).concat().map_zip(fv.L)
            yield fold, tr, val

    def load_image_item(self, item):
        if type(item) is np.ndarray: return item    # image/mask is already loaded as np array
        elif self.args.encrypted: return crypto.decrypt_img(item, self.args.ckey)
        else: return item

    def create_dls_from_lst(self, blocks, tr, val, bs, size, get_x=None, get_y=None, **kwargs):
        tfms = fv.aug_transforms(size=size)
        if not self.args.no_norm:
            tfms.append(fv.Normalize.from_stats(*fv.imagenet_stats))
        d = fv.DataBlock(blocks=blocks,
                         get_items=lambda source: list(zip(val[0] + tr[0], val[1] + tr[1])),
                         get_x=fv.Pipeline([fv.ItemGetter(0), self.load_image_item if get_x is None else get_x]),
                         get_y=fv.Pipeline([fv.ItemGetter(1), fv.noop if get_y is None else get_y]),
                         splitter=fv.IndexSplitter(list(range(len(val[0])))),
                         item_tfms=fv.Resize(int(1.5*self.args.input_size)),
                         batch_tfms=tfms)
        if self.args.debug_dls:
            if train_utils.GPUManager.is_master_process(): d.summary(self.args.data, bs=bs)
            sys.exit()
        # set path args so that learner objects use it
        return d.dataloaders(self.args.data, path=self.args.exp_logdir, bs=bs, seed=self.args.seed, **kwargs)

    def maybe_progressive_resizing(self, tr, val, fold_suffix):
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
        for it, run, dls in self.maybe_progressive_resizing(tr, val, fold_suffix):
            run = f'{run_prefix}{run}'
            if it == 0 and learn is None: learn = fd.rank0_first(lambda: self.create_learner(dls))
            self.basic_train(run, learn, dls)
            self.evaluate_on_test_sets(learn, run)
            train_utils.GPUManager.clean_gpu_memory(learn.dls)
        return learn, run

    def get_run_params(self, run_info):
        regex = r"^(?:__R(?P<repeat>\d+)__)?__S(?P<progr_size>\d+)px_bs(?P<bs>\d+)____F(?P<fold>\d+)__.*$"
        run_params = SimpleNamespace(**re.match(regex, run_info).groupdict())
        run_params.progr_size = int(run_params.progr_size)
        run_params.bs = int(run_params.bs)
        return run_params

    def get_sorting_run_key(self, run_info):
        return run_info.replace(f'__F{self.get_run_params(run_info).fold}__', '')

    def load_learner_from_run_info(self, run_info, tr, val, mpath=None):
        run_params = self.get_run_params(run_info)
        dls = self.create_dls(tr, val, run_params.bs, run_params.progr_size)
        learn = fd.rank0_first(lambda: self.create_learner(dls))
        if mpath is not None: train_utils.load_custom_pretrained_weights(learn.model, mpath)
        return learn


class ImageTBCb(fc.TensorBoardBaseCallback):
    def __init__(self, log_dir, run_info, grouped_metrics, all_cats):
        super().__init__()
        self.log_dir = log_dir
        self.run_info = run_info
        self.grouped_metrics = grouped_metrics
        self.all_cats = all_cats

    def can_run(self):
        return not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")\
               and train_utils.GPUManager.is_master_process()

    def before_fit(self):
        self.run = self.can_run()
        if self.run: self._setup_writer()

    def after_batch(self):
        if not self.run: return
        # if no self.smooth_loss then -1: when loss is nan, Recorder does not set smooth loss causing exception else
        self.writer.add_scalar(f'{self.run_info}_Loss/train_loss', getattr(self, "smooth_loss", -1), self.train_iter)
        for i, h in enumerate(self.opt.hypers):
            for k, v in h.items(): self.writer.add_scalar(f'{self.run_info}_Opt_hyper/{k}_{i}', v, self.train_iter)

    def after_epoch(self):
        if not self.run: return
        grouped, reduced = defaultdict(dict), defaultdict(dict)
        for n, v in zip(self.recorder.metric_names[2:-1], self.recorder.log[2:-1]):
            if n in self.grouped_metrics:
                perf = n.split('_')[0]
                cat_code = n.replace(f'{perf}_', '')
                # check if all_cat IN cat_code because there can be different variation (eg all & all_no_bg in segm)
                if self.all_cats in cat_code: reduced[n] = v
                else: grouped[perf][cat_code] = v
            else:
                log_group = 'Loss' if "loss" in n else 'Metrics'
                self.writer.add_scalar(f'{self.run_info}_{log_group}/{n}', v, self.train_iter)
        for perf, v in grouped.items():
            self.writer.add_scalars(f'{self.run_info}_Metrics/{perf}', v, self.train_iter)
        self.writer.add_scalars(f'{self.run_info}_Metrics/ALL', reduced, self.train_iter)

