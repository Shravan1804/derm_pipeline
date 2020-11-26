import re
import os
import sys
from collections import defaultdict

import numpy as np

import torch
import fastai.vision.all as fv
import fastai.distributed as fd   # needed for fastai multi gpu
import fastai.callback.tensorboard as fc

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto, train_utils


def get_full_img_dict(images, sep):
    """Returns a dict with keys the full images names and values the lst of corresponding images.
    sep is the string which separates the full img names
    Assumes all img parts have same class (located in same dir) which will be attributed to full image"""
    full_images_dict = {}
    for fpath in images:
        cls = os.path.basename(os.path.dirname(fpath))
        file, ext = os.path.splitext(os.path.basename(fpath))
        fi = os.path.join(cls, f'{file.split(sep)[0] if sep in file else file}{ext}')
        if fi in full_images_dict:
            full_images_dict[fi].append(fpath)
        else:
            full_images_dict[fi] = [fpath]
    return full_images_dict


def crop_im(im, bbox):
    wmin, hmin, wmax, hmax = bbox
    return im[hmin:hmax+1, wmin:wmax+1]


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

    def tensorboard_cb(self, run_name):
        return ImageTBCb(self.args.exp_logdir, run_name, self.cats_metrics.keys(), self.ALL_CATS)

    def get_test_sets_items(self):
        if self.args.sl_tests is None:
            return np.array([])
        test_sets_items = []
        for test_dir in self.args.sl_tests:
            test_sets_items.append((test_dir, self.load_data(os.path.join(self.args.data, test_dir))))
        return test_sets_items

    def get_train_items(self):
        sl_images = self.load_data(os.path.join(self.args.data, self.args.sl_train))
        if self.args.use_wl:
            wl_images = self.load_data(os.path.join(self.args.data, self.args.wl_train))
        else:
            wl_images = (np.array([]), np.array([]))
        return sl_images, wl_images

    def get_images_cls(self, images):
        if self.stratify: raise NotImplementedError
        else: return np.ones_like(images)

    def load_image_item(self, path):
        return crypto.decrypt_img(path, self.args.user_key) if self.args.encrypted else path

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

    def split_data(self, items: np.ndarray, items_cls: np.ndarray):
        full_images_dict = get_full_img_dict(items, self.full_img_sep)
        full_images = np.array(list(full_images_dict.keys()))

        for fold, tr, val in super().split_data(full_images, self.get_images_cls(full_images)):
            tr_images = np.array([i for fi in tr[0] for i in full_images_dict[fi]])
            val_images = np.array([i for fi in val[0] for i in full_images_dict[fi]])
            np.random.shuffle(tr_images)
            np.random.shuffle(val_images)
            yield fold, (tr_images, self.get_images_cls(tr_images)), (val_images, self.get_images_cls(val_images))

    def progressive_resizing(self, tr, val, fold_suffix):
        if self.args.progr_size:
            input_sizes = [int(self.args.input_size * f) for f in self.args.size_facts]
            batch_sizes = [max(1, min(int(self.args.bs / f / f), tr[0].size) // 2 * 2) for f in self.args.size_facts]
        else:
            input_sizes = [self.args.input_size]
            batch_sizes = [self.args.bs]

        for it, (bs, size) in enumerate(zip(batch_sizes, input_sizes)):
            run = f'__S{size}px_bs{bs}__{fold_suffix}'
            print(f"Progressive resizing {it + 1}/{len(batch_sizes)}: running {run}")
            yield it, run, self.create_dls(tr, val, bs, size)

    def progressive_resizing_train(self, tr, val, fold_suffix, run_prefix="", learn=None):
        for it, run, dls in self.progressive_resizing(tr, val, fold_suffix):
            if it == 0 and learn is None: learn = fd.rank0_first(lambda: self.create_learner(dls))
            self.basic_train(learn, f'{run_prefix}{run}', dls)
            self.evaluate_on_test_sets(learn, run)
        return learn, run

    def train_model(self):
        sl_images, wl_images = self.get_train_items()
        for fold, tr, val in self.split_data(*sl_images):
            fold_suffix = f'__F{common.zero_pad(fold, self.args.nfolds)}__'
            if fold == 0 or not self.args.use_wl:
                learn, last_run = self.progressive_resizing_train(tr, val, f'{fold_suffix}sl_only')

            if self.args.use_wl:
                if fold == 0: wl_images = self.evaluate_and_correct_wl(learn, wl_images, last_run)
                for repeat in range(self.args.nrepeats):
                    repeat_prefix = f'__R{common.zero_pad(repeat, self.args.nrepeats)}__'
                    print(f"WL-SL train procedure {repeat + 1}/{self.args.nrepeats}")
                    learn, _ = self.progressive_resizing_train(wl_images, val, f'{fold_suffix}wl_only', repeat_prefix)
                    learn, last_run = self.progressive_resizing_train(tr, val, f'{fold_suffix}_wl_sl', repeat_prefix, learn)
                    wl_images = self.evaluate_and_correct_wl(learn, wl_images, last_run)
        self.generate_tests_reports()

    def get_sorting_run_key(self, run_name):
        regex = r"^(?:__R(?P<repeat>\d+)__)?__S(?P<progr_size>\d+)px_bs\d+____F(?P<fold>\d+)__.*$"
        m = re.match(regex, run_name)
        return run_name.replace(f'__F{m.group("fold")}__', '')

    def aggregate_test_performance(self, folds_res):
        """Returns a dict with perf_fn as keys and values a tuple of lsts of categories mean/std"""
        cats = self.args.cats + [self.ALL_CATS]
        res = {p: [[m.metrics_res[f'{c}_{p}'] for m in folds_res] for c in cats] for p in self.BASIC_PERF_FNS}
        res = {p: [train_utils.tensors_mean_std(vals) for vals in cat_vals] for p, cat_vals in res.items()}
        return {p: tuple([torch.stack(s).numpy() for s in zip(*cat_vals)]) for p, cat_vals in res.items()}


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
        self.writer.add_scalar(f'{self.run_name}_Loss/train_loss', self.smooth_loss, self.train_iter)
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

