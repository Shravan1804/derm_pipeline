import os
import gc
import sys
import pickle
import argparse
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

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
    # Make sure classes are balanced
    # Proportion of both Positive and Negative that were correctly classified
    return (TP + TN) / (TP + TN + FP + FN + epsilon)


def precision(TP, TN, FP, FN, epsilon=1e-8):
    # Proportion of predicted Positives that are truly Positive
    return TP / (TP + FP + epsilon)


def recall(TP, TN, FP, FN, epsilon=1e-8):
    # Proportion of actual Positives (in ground truth) that are correctly classified
    return TP / (TP + FN + epsilon)


def F1(TP, TN, FP, FN, epsilon=1e-8):
    # Can be used to compare two classifiers BUT
    # F1-score gives a larger weight to lower numbers e.g. 100% pre and 0% rec => 0% F1
    # F1-score gives equal weight to pre/rec which may not what we seek depending on the problem
    pre, rec = precision(TP, TN, FP, FN, epsilon), recall(TP, TN, FP, FN, epsilon)
    return 2 * pre * rec / (pre + rec)


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


def load_custom_pretrained_weights(model, weights_path):
    new_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
    model_state_dict = model.state_dict()
    for name, param in model_state_dict.items():
        if name in new_state_dict:
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                print('Shape mismatch at:', name, 'skipping')
        else:
            print(f'{name} weight of the model not in pretrained weights')
    model.load_state_dict(model_state_dict)


def show_tensors_in_memory(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    # inspired from https://forums.fast.ai/t/gpu-memory-not-being-freed-after-training-is-over/10265/7
    def pretty_size(size): return " × ".join(map(str, size)) if isinstance(size, torch.Size) else size

    def print_tensor_like_obj_details(obj):
        if not gpu_only or obj.is_cuda:
            details = f'{type(obj).__name__} '
            if not torch.is_tensor(obj) and hasattr(obj, "data") and torch.is_tensor(obj.data):
                details += f'-> {type(obj.data).__name__} '
            details += f': {"GPU" if obj.is_cuda else "CPU"} {" pinned" if obj.data.is_pinned else ""} '
            details += f'{" grad" if obj.requires_grad else ""} {pretty_size(obj.data.size())}'
            print(details)

    total_size = 0
    for obj in gc.get_objects():
        try:
            with_data_tensor = hasattr(obj, "data") and torch.is_tensor(obj.data)
            if torch.is_tensor(obj) or with_data_tensor:
                print_tensor_like_obj_details(obj)
                total_size += obj.data.numel() if with_data_tensor else obj.numel()
        except Exception as err:
            print(err)
    print("Total size:", total_size)


class GPUManager:
    @staticmethod
    def clean_gpu_memory(*items):
        for item in items: del item
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def default_gpu_device_ids():
        assert torch.cuda.is_available(), "Cannot run without CUDA device"
        return list(range(torch.cuda.device_count()))

    @staticmethod
    def in_parallel_mode(): return not GPUManager.in_distributed_mode() and torch.cuda.device_count() > 1

    @staticmethod
    def in_distributed_mode(): return os.environ.get('RANK', None) is not None

    @staticmethod
    def distributed_rank(): return int(os.environ.get('RANK', 0))

    @staticmethod
    def is_master_process(): return GPUManager.distributed_rank() == 0 if GPUManager.in_distributed_mode() else True

    @staticmethod
    def init_distributed_process():
        if GPUManager.in_distributed_mode():
            rank = GPUManager.distributed_rank()
            fd.setup_distrib(rank)
            torch.cuda.set_device(rank)

    @staticmethod
    def sync_distributed_process():
        if GPUManager.in_distributed_mode(): fv.distrib_barrier()

    @staticmethod
    def running_context(learn, device_ids=None):
        device_ids = list(range(torch.cuda.device_count())) if device_ids is None else device_ids
        return learn.distrib_ctx() if GPUManager.in_distributed_mode() else learn.parallel_ctx(device_ids)


class FastaiTrainer:
    @staticmethod
    def get_argparser(desc="Fastai trainer arguments", pdef=dict(), phelp=dict()):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('--use-wl', action='store_true', help="Data dir contains wl and sl data")
        parser.add_argument('--nrepeats', default=3, type=int, help="N repeat: wl pretrain -> sl train -> wl correct")
        parser.add_argument('--wl-train', type=str, nargs='+', help="weak labels (wl) dirs")
        parser.add_argument('--sl-train', type=str, nargs='+', help="strong labels (sl) dirs")
        parser.add_argument('--sl-tests', type=str, nargs='+', help="sl test dirs")

        parser.add_argument('--encrypted', action='store_true', help="Data is encrypted")
        parser.add_argument('--ckey', type=str, help="Data encryption key")

        parser.add_argument('--inference', action='store_true', help="No train, only inference")
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
        parser.add_argument('--bs', default=pdef.get('--bs', 6), type=int, help="Batch size per GPU device")
        parser.add_argument('--lr', type=float, help='when None: uses auto_lr in parallel mode else .002')
        parser.add_argument('--fepochs', type=int, default=4, help='Epochs for frozen model')
        parser.add_argument('--epochs', type=int, default=pdef.get('--epochs', 12), help='Epochs for unfrozen model')

        parser.add_argument('--no-norm', action='store_true', help="Do not normalizes data")
        parser.add_argument('--full-precision', action='store_true', help="Train with full precision (more gpu memory)")
        parser.add_argument('--early-stop', action='store_true', help="Early stopping during training")

        parser.add_argument('--proc-gpu', type=int, default=0, help="Id of gpu to be used by process")
        parser.add_argument("--gpu-ids", type=int, nargs='+', default=GPUManager.default_gpu_device_ids(),
                            help="Ids of gpus to be used on each machine")
        parser.add_argument("--num-machines", type=int, default=1, help="number of machines")

        parser.add_argument('--seed', type=int, default=pdef.get('--seed', 42), help="Random seed")

        return parser

    @staticmethod
    def get_exp_logdir(args, custom=""):
        d = f'{common.now()}_{args.model}_bs{args.bs}_epo{args.epochs}_seed{args.seed}' \
            f'_world{args.num_machines * len(args.gpu_ids)}'
        d += '' if args.no_norm else '_normed'
        d += '_fp32' if args.full_precision else '_fp16'
        d += f'_CV{args.nfolds}' if args.cross_val else f'_noCV_valid{args.valid_size}'
        d += '_WL' if args.use_wl else '_SL'
        d += f'_{custom}_{args.exp_name}'
        return d

    @staticmethod
    def prepare_training(args):
        common.set_seeds(args.seed)
        common.check_dir_valid(args.data)

        if args.inference:
            common.check_dir_valid(args.exp_logdir)

        assert torch.cuda.is_available(), "Cannot run without CUDA device"
        args.bs = args.bs * len(args.gpu_ids) if GPUManager.in_parallel_mode() else args.bs
        # required for segmentation otherwise causes a NCCL error in inference distrib running context
        if GPUManager.in_distributed_mode(): GPUManager.init_distributed_process()

        if args.encrypted:
            args.ckey = os.environ.get('CRYPTO_KEY').encode() if GPUManager.in_distributed_mode() else args.ckey
            args.ckey = crypto.request_key(args.data, args.ckey)

        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(args.logdir, FastaiTrainer.get_exp_logdir(args))
        args.exp_logdir = fd.rank0_first(lambda: common.maybe_create(args.exp_logdir))
        print("Log directory: ", args.exp_logdir)

    def __init__(self, args, stratify):
        self.MODEL_SUFFIX = '_model'
        self.args = args
        self.stratify = stratify
        self.cust_metrics = self.prepare_custom_metrics()
        self.test_set_results = {test_name: defaultdict(list) for test_name in self.args.sl_tests}
        print("Training args:", self.args)

    def prepare_custom_metrics(self): raise NotImplementedError

    def load_items(self, path): raise NotImplementedError

    def get_test_sets_items(self): raise NotImplementedError

    def get_train_items(self): raise NotImplementedError

    def create_learner(self): raise NotImplementedError

    def load_learner_from_run_info(self, run_info, mpath, tr, val): raise NotImplementedError

    def create_dls(self): raise NotImplementedError

    def train_procedure(self, tr, val, fold_suffix, run_prefix="", learn=None): raise NotImplementedError

    def correct_wl(self, wl_items, preds): raise NotImplementedError

    def early_stop_cb(self): raise NotImplementedError

    def get_run_params(self, run_info): raise NotImplementedError

    def get_sorting_run_key(self, run_info): raise NotImplementedError

    def plot_test_performance(self, test_path, agg): raise NotImplementedError

    def tensorboard_cb(self, run_info): raise NotImplementedError

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
            if self.args.cross_val and not self.args.inference: print(f"Data fold {fold + 1}/{self.args.nfolds}")
            yield fold, (items[train_idx], items_cls[train_idx]), (items[valid_idx], items_cls[valid_idx])

    def get_train_cbs(self, run):
        cbs = []
        if GPUManager.is_master_process():  # otherwise creates deadlock
            cbs.append(self.tensorboard_cb(run))
        if self.args.early_stop:
            cbs.append(self.early_stop_cb())
        return cbs

    def prepare_learner(self, learn):
        if not self.args.full_precision:
            learn.to_fp16()
        learn.model.cuda()
        return learn

    def auto_lr_find(self, learn):
        if GPUManager.in_distributed_mode():
            def fixed_lr_find(self):
                self.learn.opt.zero_grad()  # Need to zero the gradients of the model before detaching the optimizer for future fits
                tmp_f = self.path / self.model_dir / '_tmp.pth'
                if tmp_f.exists():
                    self.learn.load('_tmp', with_opt=True)
                    if GPUManager.is_master_process(): os.remove(tmp_f)
                GPUManager.sync_distributed_process()
            fv.LRFinder.after_fit = fixed_lr_find
        lr_min, lr_steep = learn.lr_find(suggestions=True, show_plot=False)
        return lr_min/10

    def basic_train(self, run, learn, dls=None, save_model=True):
        GPUManager.sync_distributed_process()
        print("Training model:", run)
        if dls is not None: learn.dls = dls
        train_cbs = self.get_train_cbs(run)
        with GPUManager.running_context(learn, self.args.gpu_ids):
            lr = self.auto_lr_find(learn) if self.args.lr is None else self.args.lr
            #lr = .002 if self.args.lr is None else self.args.lr
            learn.fine_tune(self.args.epochs, base_lr=lr, freeze_epochs=self.args.fepochs, cbs=train_cbs)
        if save_model: learn.save(os.path.join(self.args.exp_logdir, f'{run}{self.MODEL_SUFFIX}_lr{lr:.2e}'))

    def evaluate_and_correct_wl(self, learn, wl_items, run):
        """Evaluate and correct weak labeled items, clears GPU memory (model and dls)"""
        GPUManager.sync_distributed_process()
        print("Evaluating WL data:", run)
        dl = learn.dls.test_dl(list(zip(*wl_items)), with_labels=True)
        with GPUManager.running_context(learn, self.args.gpu_ids):
            _, targs, decoded_preds = learn.get_preds(dl=dl, with_decoded=True)
        wl_items = self.correct_wl(wl_items, decoded_preds)
        GPUManager.clean_gpu_memory(dl, learn.dls, learn)
        return wl_items

    def evaluate_on_test_sets(self, learn, run):
        """Evaluate test sets, clears GPU memory held by test dl(s)"""
        for test_name, test_items_with_cls in self.get_test_sets_items():
            print("Testing model", run, "on", test_name)
            GPUManager.sync_distributed_process()
            dl = learn.dls.test_dl(list(zip(*test_items_with_cls)), with_labels=True)
            with GPUManager.running_context(learn, self.args.gpu_ids):
                interp = SimpleNamespace()
                interp.preds, interp.targs, interp.decoded = learn.get_preds(dl=dl, with_decoded=True)
                GPUManager.clean_gpu_memory(dl)
            interp = self.process_test_preds(interp)
            del interp.preds, interp.targs, interp.decoded
            self.test_set_results[test_name][self.get_sorting_run_key(run)].append(interp)

    def process_test_preds(self, interp):
        """Adds custom metrics results to interp object. Should return interp."""
        interp.metrics = {mn: mfn(interp.preds, interp.targs) for mn, mfn in self.cust_metrics.items()}
        return interp

    def aggregate_test_performance(self, folds_res):
        merged = [(mn, [fr.metrics[mn] for fr in folds_res]) for mn in folds_res[0].metrics.keys()]
        return {mn: tuple(s.numpy() for s in tensors_mean_std(mres)) for mn, mres in merged}

    def generate_tests_reports(self):
        print("Aggregating test predictions ...")
        if not GPUManager.is_master_process(): return
        for test_name in self.args.sl_tests:
            test_path = common.maybe_create(self.args.exp_logdir, test_name)
            for run, folds_results in self.test_set_results[test_name].items():
                agg = self.aggregate_test_performance(folds_results)
                self.plot_test_performance(test_path, run, agg)
                with open(os.path.join(test_path, f'{run}_test_results.p'), 'wb') as f:
                    pickle.dump(agg, f)

    def inference(self):
        if not self.args.inference:
            print("Inference mode not set, skipping inference.")
            return
        models = [m for m in common.list_files(self.args.exp_logdir, full_path=True) if m.endswith(".pth")]
        _, tr, val = next(self.split_data(*self.get_train_items()[0]))
        for mpath in models:
            run_info = os.path.basename(mpath).split(self.MODEL_SUFFIX)[0]
            learn = self.load_learner_from_run_info(run_info, tr, val, mpath)
            self.evaluate_on_test_sets(learn, run_info)
            GPUManager.clean_gpu_memory(learn.dls, learn)
        self.generate_tests_reports()

    def train_model(self):
        if self.args.inference:
            print("Inference only mode set, skipping train.")
            return
        sl_data, wl_data = self.get_train_items()
        for fold, tr, val in self.split_data(*sl_data):
            fold_suffix = f'__F{common.zero_pad(fold, self.args.nfolds)}__'
            if fold == 0 or not self.args.use_wl:
                learn, prev_run = self.train_procedure(tr, val, f'{fold_suffix}sl_only')

            if self.args.use_wl:
                if fold == 0:
                    wl_data = self.evaluate_and_correct_wl(learn, wl_data, prev_run)
                for repeat in range(self.args.nrepeats):
                    repeat_prefix = f'__R{common.zero_pad(repeat, self.args.nrepeats)}__'
                    print(f"WL-SL train procedure {repeat + 1}/{self.args.nrepeats}")
                    learn, _ = self.train_procedure(wl_data, val, f'{fold_suffix}wl_only', repeat_prefix)
                    learn, prev_run = self.train_procedure(tr, val, f'{fold_suffix}_wl_sl', repeat_prefix, learn)
                    wl_data = self.evaluate_and_correct_wl(learn, wl_data, prev_run)

            GPUManager.clean_gpu_memory(learn.dls, learn)
        self.generate_tests_reports()

