import os
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

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto


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


def split_model(model, splits):
    """Inspired from fastai 1, splits model on requested top level children"""
    top_children = list(model.children())
    idxs = [top_children.index(split) for split in splits]
    assert idxs == sorted(idxs), f"Provided splits ({splits}) are not sorted."
    assert len(idxs) > 0, f"Provided splits ({splits}) not found in top level children: {top_children}"
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(top_children): idxs.append(len(top_children))
    return [torch.nn.Sequential(*top_children[i:j]) for i, j in zip(idxs[:-1], idxs[1:])]


class CustomItemGetter(fv.ItemGetter):
    def __init__(self, i, fn): fv.store_attr()
    def encodes(self, x): return self.fn(x[self.i])

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

    def tensorboard_cb(self, run_name): raise NotImplementedError

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

    def save_learner(self, learn, run):
        if not self.args.full_precision:
            learn.to_fp32()
        learn.save(os.path.join(self.args.exp_logdir, f'{run}_model'))
        if not self.args.full_precision:
            learn.to_fp16()

    def basic_train(self, learn, run, dls):
        print("Training model:", run)
        learn.dls = dls
        with learn.distrib_ctx():
            learn.fine_tune(self.args.epochs, cbs=self.get_train_cbs(run))
        self.save_learner(learn, run)

    def evaluate_and_correct_wl(self, learn, wl_items, run):
        print("Evaluating WL data:", run)
        dl = learn.dls.test_dl(list(zip(*wl_items)), with_labels=True)
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
            dl = learn.dls.test_dl(list(zip(*test_items_with_cls)), with_labels=True)
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

