import re
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import fastai.vision.all as fv
import fastai.distributed as fd   # needed for fastai multi gpu
import fastai.callback.tensorboard as fc

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto, train_utils


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
        return parser

    @staticmethod
    def get_exp_logdir(args, custom=""):
        d = f'_input{args.input_size}'
        d += f'_progr-size{"_".join(map(str, args.size_facts))}' if args.progr_size else ""
        return super(ImageTrainer, ImageTrainer).get_exp_logdir(args, custom=f'{d}_{custom}')

    @staticmethod
    def prepare_training(args):
        common.check_dir_valid(args.data)
        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(args.logdir, ImageTrainer.get_exp_logdir(args))
        super(ImageTrainer, ImageTrainer).prepare_training(args)

    def __init__(self, args, stratify, full_img_sep):
        self.ALL_CATS = '__all__'
        self.full_img_sep = full_img_sep
        self.BASIC_PERF_FNS = ['accuracy', 'precision', 'recall']
        super().__init__(args, stratify)

    def get_cats_with_all(self):
        return [*self.args.cats, self.ALL_CATS]

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn): raise NotImplementedError

    def get_metrics_cats_name(self, perf_fn, cat): return f'{perf_fn}_{cat}'

    def prepare_cats_metrics(self):
        metrics_fn = {}
        for perf_fn in self.BASIC_PERF_FNS:
            for cat_id, cat in zip([*range(len(self.args.cats)), None], self.get_cats_with_all()):
                self.create_cats_metrics(perf_fn, cat_id, cat, metrics_fn)
        return metrics_fn

    def aggregate_test_performance(self, folds_res):
        """Returns a dict with perf_fn as keys and values a tuple of lsts of categories mean/std"""
        agg = super().aggregate_test_performance(folds_res)
        for perf_fn in self.BASIC_PERF_FNS:
            mns = [self.get_metrics_cats_name(perf_fn, cat) for cat in self.get_cats_with_all()]
            agg[perf_fn] = tuple(np.stack(s) for s in zip(*[agg.pop(mn) for mn in mns]))
        return agg

    def tensorboard_cb(self, run_name):
        return ImageTBCb(self.args.exp_logdir, run_name, self.cust_metrics.keys(), self.ALL_CATS)

    def get_test_sets_items(self):
        if self.args.sl_tests is None:
            return np.array([])
        test_sets_items = []
        for test_dir in self.args.sl_tests:
            test_sets_items.append((test_dir, self.load_items(os.path.join(self.args.data, test_dir))))
        return test_sets_items

    def get_train_items(self):
        sl_images = self.load_items(os.path.join(self.args.data, self.args.sl_train))
        if self.args.use_wl:
            wl_images = self.load_items(os.path.join(self.args.data, self.args.wl_train))
        else:
            wl_images = (np.array([]), np.array([]))
        return sl_images, wl_images

    def get_full_img_cls(self, img_path):
        if self.stratify: raise NotImplementedError
        else: return 1

    def get_full_img_dict(self, images, targets):
        """Returns a dict with keys (full images, tgt) and values the lst of corresponding images."""
        full_images_dict = defaultdict(list)
        for img, tgt in zip(images, targets):
            file, ext = os.path.splitext(os.path.basename(img))
            fi = f'{file.split(self.full_img_sep)[0] if self.full_img_sep in file else file}{ext}'
            full_images_dict[(fi, self.get_full_img_cls(img))].append((img, tgt))
        return full_images_dict

    def split_data(self, items: np.ndarray, items_cls: np.ndarray):
        fi_dict = self.get_full_img_dict(items, items_cls)
        for fold, tr, val in super().split_data(*tuple(np.array(lst) for lst in zip(*fi_dict.keys()))):
            tr = tuple(np.array(lst) for lst in zip(*[img for fik in zip(*tr) for img in fi_dict[fik]]))
            val = tuple(np.array(lst) for lst in zip(*[img for fik in zip(*val) for img in fi_dict[fik]]))
            _ = list(map(np.random.shuffle, [t for tup in (tr, val) for t in tup]))
            yield fold, tr, val

    def load_image_item(self, path):
        return crypto.decrypt_img(path, self.args.ckey) if self.args.encrypted else path

    def create_dls_from_lst(self, blocks, tr, val, bs, size, get_y=None):
        tfms = fv.aug_transforms(size=size)
        if not self.args.no_norm:
            tfms.append(fv.Normalize.from_stats(*fv.imagenet_stats))
        d = fv.DataBlock(blocks=blocks,
                         get_items=lambda source: list(zip(val[0] + tr[0], val[1] + tr[1])),
                         get_x=train_utils.CustomItemGetter(0, self.load_image_item),
                         get_y=train_utils.CustomItemGetter(1, fv.noop if get_y is None else get_y),
                         splitter=fv.IndexSplitter(list(range(len(val[0])))),
                         item_tfms=fv.Resize(self.args.input_size),
                         batch_tfms=tfms)
        return d.dataloaders(self.args.data, bs=bs)

    def maybe_progressive_resizing(self, tr, val, fold_suffix):
        if self.args.progr_size:
            input_sizes = [int(self.args.input_size * f) for f in self.args.size_facts]
            batch_sizes = [max(1, min(int(self.args.bs / f / f), tr[0].size) // 2 * 2) for f in self.args.size_facts]
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
            if it == 0 and learn is None: learn = fd.rank0_first(lambda: self.create_learner(dls))
            self.basic_train(learn, f'{run_prefix}{run}', dls)
            self.evaluate_on_test_sets(learn, run)
        return learn, run

    def get_sorting_run_key(self, run_name):
        regex = r"^(?:__R(?P<repeat>\d+)__)?__S(?P<progr_size>\d+)px_bs\d+____F(?P<fold>\d+)__.*$"
        m = re.match(regex, run_name)
        return run_name.replace(f'__F{m.group("fold")}__', '')

    def plot_test_performance(self, test_path, run, agg_run_perf):
        for show_val in [False, True]:
            save_path = os.path.join(test_path, f'{run}{"_show_val" if show_val else ""}.jpg')
            fig, axs = plt.subplots(1, 2, figsize=self.args.test_figsize)
            bar_perf = {p: cat_vals for p, cat_vals in agg_run_perf.items() if p != 'cm'}
            bar_cats = self.get_cats_with_all()
            common.grouped_barplot_with_err(axs[0], bar_perf, bar_cats, xlabel='Classes', show_val=show_val)
            common.plot_confusion_matrix(axs[1], agg_run_perf['cm'], self.args.cats)
            fig.tight_layout(pad=.2)
            plt.savefig(save_path, dpi=400)


class ImageTBCb(fc.TensorBoardBaseCallback):
    def __init__(self, log_dir, run_name, grouped_metrics, all_cats):
        super().__init__()
        self.log_dir = log_dir
        self.run_name = run_name
        self.grouped_metrics = grouped_metrics
        self.all_cats = all_cats

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds") \
                   and int(os.environ.get('RANK', 0)) == 0
        if not self.run: return
        self._setup_writer()

    def after_batch(self):
        if not self.run: return
        # if no self.smooth_loss then -1: when loss is nan, Recorder does not set smooth loss causing exception else
        self.writer.add_scalar(f'{self.run_name}_Loss/train_loss', getattr(self, "smooth_loss", -1), self.train_iter)
        for i, h in enumerate(self.opt.hypers):
            for k, v in h.items(): self.writer.add_scalar(f'{self.run_name}_Opt_hyper/{k}_{i}', v, self.train_iter)

    def after_epoch(self):
        if not self.run: return
        grouped, reduced = defaultdict(dict), defaultdict(dict)
        for n, v in zip(self.recorder.metric_names[2:-1], self.recorder.log[2:-1]):
            if n in self.grouped_metrics:
                perf = n.split('_')[1]
                if self.all_cats in n:
                    reduced[perf][n] = v
                else:
                    grouped[perf][n] = v
            else:
                log_group = 'Loss' if "loss" in n else 'Metrics'
                self.writer.add_scalar(f'{self.run_name}_{log_group}/{n}', v, self.train_iter)
        for perf, v in grouped.items():
            self.writer.add_scalars(f'{self.run_name}_Metrics/{perf}', v, self.train_iter)
        for n, v in reduced.items():
            self.writer.add_scalars(f'{self.run_name}_Metrics/{self.all_cats}_{n}', v, self.train_iter)

