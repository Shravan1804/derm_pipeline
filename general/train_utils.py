import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import dill
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

import torch
import fastai.vision.all as fv
import fastai.distributed as fd   # needed for fastai multi gpu
import fastai.callback.tensorboard as fc

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto, img_utils


def get_cls_TP_TN_FP_FN(cls_truth, cls_preds):
    TP = (cls_preds & cls_truth).sum().item()
    TN = (~cls_preds & ~cls_truth).sum().item()
    FP = (cls_preds & ~cls_truth).sum().item()
    FN = (~cls_preds & cls_truth).sum().item()
    return TP, TN, FP, FN


def accuracy(TP, TN, FP, FN, epsilon=1e-8):
    return (TP + TN) / (TP + TN + FP + FN + epsilon)


def precision(TP, TN, FP, FN, epsilon=1e-8):
    return TP / (TP + FP + epsilon)


def recall(TP, TN, FP, FN, epsilon=1e-8):
    return TP / (TP + FN + epsilon)


def tensors_mean_std(tensor_lst):
    tensors = torch.cat([t.unsqueeze(0) for t in tensor_lst], dim=0)
    mean = tensors.mean(axis=0)
    std = tensors.std(axis=0) if len(tensor_lst) > 1 else torch.zeros_like(mean)
    return mean, std


def save_learner(learn, is_fp16, save_path):
    learn.remove_cb(ImageTBCb)
    if is_fp16:
        learn.to_fp32()
    learn.save(save_path)
    if is_fp16:
        learn.to_fp16()


def split_model(model, splits):
    """Inspired from fastai 1, splits model on requested top level children"""
    top_children = list(model.children())
    idxs = [top_children.index(split) for split in splits]
    assert idxs == sorted(idxs), f"Provided splits ({splits}) are not sorted."
    assert len(idxs) > 0, f"Provided splits ({splits}) not found in top level children: {top_children}"
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(top_children): idxs.append(len(top_children))
    return [torch.nn.Sequential(*top_children[i:j]) for i, j in zip(idxs[:-1], idxs[1:])]


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
                grouped[perf][n] = v
                if self.all_cats in n:
                    reduced[perf][n] = v
            else:
                log_group = 'Loss' if "loss" in n else 'Metrics'
                self.writer.add_scalar(f'{self.run_name}_{log_group}/{n}', v, self.train_iter)
        for n, v in grouped.items():
            self.writer.add_scalars(f'{self.run_name}_Metrics/{n}', v, self.train_iter)
        for n, v in reduced.items():
            self.writer.add_scalars(f'{self.run_name}_Metrics/{self.all_cats}_{n}', v, self.train_iter)


class FastaiTrainer:
    @staticmethod
    def get_argparser(desc="Fastai trainer arguments", pdef=dict(), phelp=dict()):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('--use-wl', action='store_true', help="Data dir contains wl and sl data")
        parser.add_argument('--nrepeats', default=3, type=int, help="N repeat: wl pretrain -> sl train -> wl correct")
        parser.add_argument('--wl-train', type=str, help="weak labels (wl) dir")
        parser.add_argument('--sl-train', type=str, help="strong labels (sl) dir")
        parser.add_argument('--sl-tests', type=str, nargs='+', help="sl test dirs")

        parser.add_argument('--encrypted', action='store_true', help="Data is encrypted")
        parser.add_argument('--user-key', type=str, help="Data encryption key")

        parser.add_argument('--cross-val', action='store_true', help="Perform 5-fold cross validation on sl train set")
        parser.add_argument('--nfolds', default=5, type=int, help="Number of folds for cross val")
        parser.add_argument('--valid-size', default=.2, type=float, help='If no cv, splits train set with this %')

        parser.add_argument('-name', '--exp-name', required=True, help='Custom string to append to experiment log dir')
        parser.add_argument('--logdir', type=str, default=pdef.get('--logdir', os.path.join(str(Path.home()), 'logs')),
                            help=phelp.get('--logdir', "Root dir where logs will be saved, default to $HOME/logs"))
        parser.add_argument('--exp-logdir', type=str, help="Experiment logdir, will be created in root log dir")
        parser.add_argument('--test-figsize', type=float, nargs='+', default=pdef.get('--test-figsize', [7, 3.4]),
                            help=phelp.get('--test-figsize', "figsize of test performance plots"))

        parser.add_argument('--model', type=str, default=pdef.get('--model', None), help="Model name")
        parser.add_argument('--bs', default=pdef.get('--bs', 6), type=int, help="Batch size")
        parser.add_argument('--epochs', type=int, default=pdef.get('--epochs', 26), help='Number of epochs to run')

        parser.add_argument('--no-norm', action='store_true', help="Do not normalizes data")
        parser.add_argument('--full-precision', action='store_true', help="Train with full precision (more gpu memory)")
        parser.add_argument('--early-stop', action='store_true', help="Early stopping during training")

        parser.add_argument('--gpu', type=int, required=True, help="Id of gpu to be used by script")
        parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
        parser.add_argument("--num-machines", type=int, default=1, help="number of machines")

        parser.add_argument('--seed', type=int, default=pdef.get('--seed', 42), help="Random seed")

        return parser

    @staticmethod
    def get_exp_logdir(args, custom=""):
        d = f'{common.now()}_{args.model}_bs{args.bs}_epo{args.epochs}_seed{args.seed}' \
            f'_world{args.num_machines * args.num_gpus}'
        d += '' if args.no_norm else '_normed'
        d += '_fp32' if args.full_precision else '_fp16'
        d += f'_CV{args.nfolds}' if args.cross_val else f'_noCV_valid{args.valid_size}'
        d += '_WL' if args.use_wl else '_SL'
        d += f'_{custom}_{args.exp_name}'
        return d

    @staticmethod
    def prepare_training(args):
        common.set_seeds(args.seed)

        assert torch.cuda.is_available(), "Cannot run without CUDA device"
        fd.setup_distrib(args.gpu)

        if args.encrypted:
            args.user_key = os.environ.get('CRYPTO_KEY', "").encode()
            args.user_key = crypto.request_key(args.data, args.user_key)

        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(args.logdir, FastaiTrainer.get_exp_logdir(args))
        args.exp_logdir = fd.rank0_first(lambda: common.maybe_create(args.exp_logdir))
        print("Log directory: ", args.exp_logdir)

    def __init__(self, args, stratify):
        self.ALL_CATS = '__all__'
        self.args = args
        self.stratify = stratify
        self.cats_metrics = self.get_metrics()
        self.test_set_results = {test_name: defaultdict(list) for test_name in self.args.sl_tests}

    def get_metrics(self): raise NotImplementedError

    def load_data(self, path): raise NotImplementedError

    def get_test_sets_items(self): raise NotImplementedError

    def get_train_items(self): raise NotImplementedError

    def create_learner(self): raise NotImplementedError

    def create_dls(self): raise NotImplementedError

    def correct_wl(self, wl_items, preds): raise NotImplementedError

    def train_model(self): raise NotImplementedError

    def early_stop_cb(self): raise NotImplementedError

    def get_sorting_run_key(self, run_name): raise NotImplementedError

    def aggregate_test_performance(self, folds_res): raise NotImplementedError

    def plot_test_performance(self, test_path, agg): raise NotImplementedError

    def is_master_process(self): return self.args.gpu == 0

    def tensorboard_cb(self, run_name):
        return ImageTBCb(self.args.exp_logdir, run_name, self.cats_metrics.keys(), self.ALL_CATS)

    def split_data(self, items: np.ndarray, items_cls: np.ndarray):
        np.random.seed(self.args.seed)

        if self.stratify:
            cv_splitter, no_cv_splitter = StratifiedKFold, StratifiedShuffleSplit
        else:
            cv_splitter, no_cv_splitter = KFold, ShuffleSplit

        if self.args.cross_val:
            splitter = cv_splitter(n_splits=self.args.nfolds, shuffle=True, random_state=self.args.seed)
        else:
            splitter = no_cv_splitter(n_splits=1, test_size=self.args.valid_size, random_state=self.args.seed)

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(items, items_cls)):
            if self.args.cross_val:
                print("FOLD:", fold)
            yield fold, (items[train_idx], items_cls[train_idx]), (items[valid_idx], items_cls[valid_idx])

    def get_train_cbs(self, run):
        cbs = []
        cbs.append(self.tensorboard_cb(run))
        if self.args.early_stop:
            cbs.append(self.early_stop_cb())
        return cbs

    def prepare_learner(self, learn):
        if not self.args.full_precision:
            learn.to_fp16()
        return learn

    def basic_train(self, learn, run, dls):
        print("Training model:", run)
        learn.dls = dls
        with learn.distrib_ctx():
            learn.fine_tune(self.args.epochs, cbs=self.get_train_cbs(run))
        save_path = os.path.join(self.args.exp_logdir, f'{run}_model')
        save_learner(learn, is_fp16=(not self.args.full_precision), save_path=save_path)

    def evaluate_and_correct_wl(self, learn, wl_items, run):
        print("Evaluating WL data:", run)
        dl = learn.dls.test_dl(wl_items, with_labels=True)
        with learn.distrib_ctx():
            _, targs, decoded_preds = learn.get_preds(dl=dl, with_decoded=True)
        wl_items, changes = self.correct_wl(wl_items, decoded_preds)
        if self.is_master_process():
            with open(os.path.join(self.args.exp_logdir, f'{common.now()}_{run}__wl_changes.txt'), 'w') as changelog:
                changelog.write('file;old_label;new_label\n')
                changelog.write(changes)
        return wl_items

    def evaluate_on_test_sets(self, learn, run):
        print("Testing model:", run)
        for test_name, test_items_with_cls in self.get_test_sets_items():
            dl = learn.dls.test_dl(test_items_with_cls, with_labels=True)
            with learn.distrib_ctx():
                interp = fv.Interpretation.from_learner(learn, dl=dl)
            interp.metrics_res = {mn: m_fn(interp.preds, interp.targs) for mn, m_fn in self.cats_metrics.items()}
            self.test_set_results[test_name][self.get_sorting_run_key(run)].append(self.process_preds(interp))

    def process_preds(self, interp): return interp

    def generate_tests_reports(self):
        for test_name in self.args.sl_tests:
            test_path = common.maybe_create(self.args.exp_logdir, test_name)
            for run, folds_results in self.test_set_results[test_name].items():
                agg = self.aggregate_test_performance(folds_results)
                self.plot_test_performance(test_path, run, agg)
                with open(os.path.join(test_path, f'{run}_test_results.p'), 'wb') as f:
                    dill.dump(agg, f)


class ImageTrainer(FastaiTrainer):
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
        self.full_img_sep = full_img_sep
        self.BASIC_PERF_FNS = ['accuracy', 'precision', 'recall']
        super().__init__(args, stratify)

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

    def create_dls_from_lst(self, blocks, tr, val, get_y, bs, size):
        tfms = fv.aug_transforms(size=size)
        if not self.args.no_norm:
            tfms.append(fv.Normalize.from_stats(*fv.imagenet_stats))
        d = fv.DataBlock(blocks=blocks,
                         get_items=lambda x: tr + val,
                         get_x=lambda x: crypto.decrypt_img(x[0], self.args.user_key) if self.args.encrypted else x[0],
                         get_y=get_y,
                         splitter=fv.IndexSplitter(list(range(len(tr), len(tr) + len(val)))),
                         item_tfms=fv.Resize(self.args.input_size),
                         batch_tfms=tfms)
        return d.dataloaders(self.args.data, bs=bs)

    def split_data(self, items: np.ndarray, items_cls: np.ndarray):
        full_images_dict = img_utils.get_full_img_dict(items, self.full_img_sep)
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
            print(f"Iteration {it}: running {run}")
            yield it, run, self.create_dls(tr, val, bs, size)

    def progressive_resizing_train(self, tr, val, fold_suffix, run_prefix="", learn=None):
        for it, run, dls in self.progressive_resizing(tr, val, fold_suffix):
            if it == 0 and learn is None: learn = fd.rank0_first(lambda: self.create_learner(dls))
            self.basic_train(learn, f'{run_prefix}{run}', dls)
            self.evaluate_on_test_sets(learn, run)
        return learn, run

    def train_model(self):
        print("Running script with args:", self.args)
        sl_images, wl_images = self.get_train_items()
        for fold, tr, val in self.split_data(*sl_images):
            fold_suffix = f'__F{common.zero_pad(fold, self.args.nfolds)}__'
            if fold == 0 or not self.args.use_wl:
                learn, last_run = self.progressive_resizing_train(tr, val, f'{fold_suffix}sl_only')

            if self.args.use_wl:
                if fold == 0: wl_images = self.evaluate_and_correct_wl(learn, wl_images, last_run)
                for repeat in range(self.args.nrepeats):
                    repeat_prefix = f'__R{common.zero_pad(repeat, self.args.nrepeats)}__'
                    learn, _ = self.progressive_resizing_train(wl_images, val, f'{fold_suffix}wl_only', repeat_prefix)
                    learn, last_run = self.progressive_resizing_train(tr, val, f'{fold_suffix}_wl_sl', repeat_prefix, learn)
                    wl_images = self.evaluate_and_correct_wl(learn, wl_images, last_run)
        if self.is_master_process():
            self.generate_tests_reports()

    def get_sorting_run_key(self, run_name):
        regex = r"^(?:__R(?P<repeat>\d+)__)?__S(?P<progr_size>\d+)px_bs\d+____F(?P<fold>\d+)__.*$"
        m = re.match(regex, run_name)
        return run_name.replace(f'__F{m.group("fold")}__', '')

    def aggregate_test_performance(self, folds_res):
        """Returns a dict with perf_fn as keys and values a tuple of lsts of categories mean/std"""
        cats = self.args.cats + [self.ALL_CATS]
        res = {p: [[m.metrics_res[f'{c}_{p}'] for m in folds_res] for c in cats] for p in self.BASIC_PERF_FNS}
        res = {p: [tensors_mean_std(vals) for vals in cat_vals] for p, cat_vals in res.items()}
        return {p: tuple([torch.stack(s).numpy() for s in zip(*cat_vals)]) for p, cat_vals in res.items()}


