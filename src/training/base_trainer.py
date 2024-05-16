import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import fastai.distributed as fd  # needed for fastai multi gpu
import fastai.vision.all as fv
import numpy as np
import torch
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from tqdm import tqdm

from ..general import common, crypto
from ..training.train_utils import GPUManager, non_param_ci, tensors_mean_std


class FastaiTrainer:
    """Base training class. Should be extended to fulfill task specific requirements."""

    @staticmethod
    def get_argparser(desc="Fastai trainer arguments", pdef=dict(), phelp=dict()):
        """Create base trainer argparser.

        :param desc: str, argparser description
        :param pdef: dict, default arguments values
        :param phelp: dict, argument help strings
        :return: argparser
        """
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument(
            "--use-wl", action="store_true", help="Data dir contains wl and sl data"
        )
        parser.add_argument(
            "--nrepeats",
            default=3,
            type=int,
            help="N repeat: wl pretrain -> sl train -> wl correct",
        )
        parser.add_argument(
            "--wl-train", type=str, nargs="+", help="weak labels (wl) dirs"
        )
        parser.add_argument(
            "--sl-train",
            type=str,
            default=["train"],
            nargs="+",
            help="strong labels (sl) dirs",
        )
        parser.add_argument(
            "--sl-tests", type=str, default=["test"], nargs="+", help="sl test dirs"
        )

        parser.add_argument(
            "--encrypted", action="store_true", help="Data is encrypted"
        )
        parser.add_argument("--ckey", type=str, help="Data encryption key")

        parser.add_argument(
            "--inference", action="store_true", help="No train, only inference"
        )
        parser.add_argument(
            "--full-data", action="store_true", help="Use full train, no val split"
        )
        parser.add_argument(
            "--cross-val",
            action="store_true",
            help="Perform 5-fold cross validation on sl train set",
        )
        parser.add_argument(
            "--nfolds", default=5, type=int, help="Number of folds for cross val"
        )
        parser.add_argument(
            "--valid-size",
            default=0.2,
            type=float,
            help="If no cv, splits train set with this %",
        )

        parser.add_argument(
            "-name",
            "--exp-name",
            required=True,
            help="Custom string to append to experiment log dir",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            default=pdef.get("--logdir", os.path.join(str(Path.home()), "logs")),
            help=phelp.get(
                "--logdir", "Root dir where logs will be saved, default to $HOME/logs"
            ),
        )
        parser.add_argument(
            "--exp-logdir",
            type=str,
            help="Experiment logdir, will be created in root log dir",
        )
        parser.add_argument(
            "--test-figsize",
            type=float,
            default=pdef.get("--test-figsize", 3.4),
            help=phelp.get("--test-figsize", "figsize of test performance plots"),
        )

        parser.add_argument(
            "--model", type=str, default=pdef.get("--model", None), help="Model name"
        )
        parser.add_argument(
            "--bs",
            default=pdef.get("--bs", 6),
            type=int,
            help="Batch size per GPU device",
        )
        parser.add_argument(
            "--lr", type=float, default=pdef.get("--lr", 0.0001), help="learning rate"
        )
        parser.add_argument(
            "--auto-lr",
            action="store_true",
            help="Determine optimal lr (only in parallel train)",
        )
        parser.add_argument(
            "--fepochs",
            type=int,
            default=pdef.get("--fepochs", 4),
            help="Epochs for frozen model",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=pdef.get("--epochs", 12),
            help="Epochs for unfrozen model",
        )
        parser.add_argument(
            "--RMSProp", action="store_true", help="Use RMSProp optimizer"
        )
        parser.add_argument("--SGD", action="store_true", help="Use SGD optimizer")
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=None,
            help="Weight decay used for training.",
        )
        parser.add_argument(
            "--momentum", type=float, default=None, help="Momentum used for training."
        )

        parser.add_argument(
            "--no-norm", action="store_true", help="Do not normalizes data"
        )
        parser.add_argument(
            "--full-precision",
            action="store_true",
            help="Train with full precision (more gpu memory)",
        )
        parser.add_argument(
            "--early-stop", action="store_true", help="Early stopping during training"
        )
        parser.add_argument(
            "--reduce-lr-on-plateau",
            action="store_true",
            help="Reduce lr during training",
        )
        parser.add_argument(
            "--no-plot",
            action="store_true",
            help="Will not plot test results (e.g. too many classes)",
        )
        parser.add_argument(
            "--no-plot-val",
            action="store_true",
            help="Will not print vals inside plot area",
        )
        parser.add_argument(
            "--metrics-base-fns",
            type=str,
            nargs="+",
            default=["precision", "recall", "F1"],
            choices=[
                "accuracy",
                "precision",
                "recall",
                "F1",
                "specificity",
                "sensitivity",
                "ppv",
                "npv",
            ],
            help="metrics",
        )

        parser.add_argument(
            "--proc-gpu", type=int, default=0, help="Id of gpu to be used by process"
        )
        parser.add_argument(
            "--gpu-ids",
            type=int,
            nargs="+",
            default=GPUManager.default_gpu_device_ids(),
            help="Ids of gpus to be used on each machine",
        )
        parser.add_argument(
            "--num-machines", type=int, default=1, help="number of machines"
        )

        parser.add_argument(
            "--seed", type=int, default=pdef.get("--seed", 42), help="Random seed"
        )
        parser.add_argument(
            "--deterministic",
            action="store_true",
            help="Sets cudnn backends to deterministic",
        )
        parser.add_argument(
            "--debug-dls", action="store_true", help="Summarize dls then exits"
        )
        parser.add_argument(
            "--wandb",
            action="store_true",
            help="If the experiment should be logged to wandb",
        )

        return parser

    @staticmethod
    def get_exp_logdir(args, custom=""):
        """Create experiment log dir.

        :param args: command line arguments
        :param custom: custom string to be added in experiment log dirname
        :return: str, experiment log dir
        """
        d = f"{common.now()}_{args.model}_bs{args.bs}_epo{args.epochs}_lr{args.lr}_seed{args.seed}"
        d += f"_world{args.num_machines * len(args.gpu_ids)}"
        if not args.no_norm:
            d += "_normed"
        d += "_RMSProp" if args.RMSProp else "_Adam"
        d += "_fp32" if args.full_precision else "_fp16"
        if args.full_data:
            d += "_fullData"
        elif args.cross_val:
            d += f"_CV{args.nfolds}"
        else:
            d += f"_noCV_valid{args.valid_size}"
        d += "_WL" if args.use_wl else "_SL"
        if args.reduce_lr_on_plateau:
            d += "_reduce-lr"
        if args.deterministic:
            d += "_deterministic"
        d += f"_{custom}_{args.exp_name}"
        return d

    @staticmethod
    def prepare_training(args):
        """Set up training, checks if provided args valid, initiates distributed mode if needed.

        :param args: command line args
        """
        common.set_seeds(args.seed)
        fv.set_seed(args.seed, reproducible=args.deterministic)
        common.check_dir_valid(args.data)
        data_names = []
        if args.wl_train is not None:
            data_names.extend(args.wl_train)
        if args.sl_train is not None:
            data_names.extend(args.sl_train)
        if args.sl_tests is not None:
            data_names.extend(args.sl_tests)
        for d in data_names:
            os.path.exists(os.path.join(args.data, d))

        if args.valid_size < 0 or args.full_data:
            args.full_data, args.valid_size = True, -1
            assert not args.cross_val, "Both --full-data and --cross-val are set."
            assert not args.early_stop, "Both --full-data and --early-stop are set."
            assert (
                not args.reduce_lr_on_plateau
            ), "Both --full-data and --reduce-lr-on-plateau are set."

        assert torch.cuda.is_available(), "Cannot run without CUDA device"
        args.bs = (
            args.bs * len(args.gpu_ids) if GPUManager.in_parallel_mode() else args.bs
        )
        # required for segmentation otherwise causes a NCCL error in inference distrib running context
        if GPUManager.in_distributed_mode():
            GPUManager.init_distributed_process()

        if args.encrypted:
            args.ckey = (
                os.environ.get("CRYPTO_KEY").encode()
                if GPUManager.in_distributed_mode()
                else args.ckey
            )
            args.ckey = crypto.request_key(args.data, args.ckey)

        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(
                args.logdir, FastaiTrainer.get_exp_logdir(args)
            )
        args.exp_logdir = fd.rank0_first(lambda: common.maybe_create(args.exp_logdir))
        print("Log directory: ", args.exp_logdir)

    def __init__(self, args, stratify):
        """Create base trainer.

        :param args: command line args
        :param stratify: bool, whether to stratify data when splitting
        """
        self.PERF_CI = "__ci"
        self.ci_bootstrap_n = 100
        self.MODEL_SUFFIX = "_model"
        self.args = args
        self.stratify = stratify
        self.cust_metrics = self.prepare_custom_metrics()
        self.test_set_results = {
            test_name: defaultdict(list) for test_name in self.args.sl_tests
        }
        print("Inference" if self.args.inference else "Training", "args:", self.args)

    def prepare_custom_metrics(self):
        """Create task specific custom metrics."""
        raise NotImplementedError

    def get_test_items(self):
        """Get test items."""
        raise NotImplementedError  # returns tuple of items list and item labels list

    def get_train_items(self):
        """Get train items."""
        raise NotImplementedError

    def customize_learner(self):
        """Provide experiment specific kwargs for Learner.

        :return: kwargs dict
        """
        if self.args.RMSProp:
            opt_func = fv.RMSProp
        elif self.args.SGD:
            opt_func = fv.SGD
        else:
            opt_func = fv.Adam
        return {"opt_func": opt_func}

    def create_learner(self):
        """Create learner object."""
        raise NotImplementedError

    def load_learner_from_run_info(self, run_info, tr, val, mpath=None):
        """Load learner object.

        :param run_info: str, run information string e.g. bs, size, etc
        :param tr: tuple of train items and labels
        :param val: tuple of valid items and labels
        :param mpath: str, model weights path, optional
        """
        raise NotImplementedError

    def customize_datablock(self):
        """Provide experiment specific kwargs for DataBlock.

        :return: dict with argnames and argvalues
        """
        raise NotImplementedError

    def create_dls(self):
        """Create dataloader."""
        raise NotImplementedError

    def train_procedure(self, tr, val, fold_suffix, run_prefix="", learn=None):
        """
        Train learner. Creates learner if no learner provided.

        :param tr: tuple of train items and labels
        :param val: tuple of valid items and labels
        :param fold_suffix: str, suffix to be added to run info string
        :param run_prefix: str, prefix to be added to run info string
        :param learn: learner object, optional
        """
        raise NotImplementedError

    def correct_wl(self, wl_items, preds):
        """
        Correct weak labeled items with model predictions.

        :param wl_items: tuple of weak labeled items and labels
        :param preds: tensor, model predictions
        :return: corrrected weak label predictions
        """
        raise NotImplementedError

    def extract_run_params(self, run_info):
        """
        Extract run parameters from run info string.

        :param run_info: str, run info string
        """
        raise NotImplementedError

    def get_sorting_run_key(self, run_info):
        """
        Get key from run info string allowing to aggregate runs results.

        :param run_info: str, run info string
        """
        raise NotImplementedError

    def plot_test_performance(self, test_path, agg):
        """
        Plot test performance.

        :param test_path: str, enables performance identification, integrated in plot save path
        :param agg: dict, aggregated performance
        """
        raise NotImplementedError

    def tensorboard_cb(self, run_info):
        """Create tensorboard callback for logging."""
        raise NotImplementedError

    def split_data(self, items, items_cls):
        """
        Split train items and labels in fold.

        Yields split indices.
        :param items: list of training items
        :param items_cls: list of corresponding labels
        :yield tuple, fold id, train items and labels, valid items and labels
        """
        if self.stratify:
            cv_splitter, no_cv_splitter = StratifiedKFold, StratifiedShuffleSplit
        else:
            cv_splitter, no_cv_splitter = KFold, ShuffleSplit

        if self.args.cross_val:
            splitter = cv_splitter(
                n_splits=self.args.nfolds, shuffle=True, random_state=self.args.seed
            )
        elif self.args.full_data:
            print("WARNING: Full data set, will use merged test sets as validation set")
            yield 0, (items, items_cls), self.get_test_items(merged=True)
            return
        else:
            splitter = no_cv_splitter(
                n_splits=1, test_size=self.args.valid_size, random_state=self.args.seed
            )

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(items, items_cls)):
            if self.args.cross_val and not self.args.inference:
                print(f"Data fold {fold + 1}/{self.args.nfolds}")
            yield fold, (items[train_idx], items_cls[train_idx]), (
                items[valid_idx],
                items_cls[valid_idx],
            )

    def get_train_cbs(self, run):
        """
        Return list with custom training callbacks: tensorboard, ReduceLROnPlateau, early stopping.

        :param run: str, enable callback to know which run learner is performing
        :return: list of callbacks
        """
        cbs = []
        if GPUManager.is_master_process():  # otherwise creates deadlock
            cbs.append(self.tensorboard_cb(run))
        if self.args.reduce_lr_on_plateau:
            cbs.append(
                fv.ReduceLROnPlateau(monitor="valid_loss", min_delta=0.01, patience=3)
            )
        if self.args.early_stop:
            cbs.append(
                fv.EarlyStoppingCallback(
                    monitor="valid_loss", min_delta=0.01, patience=3
                )
            )
        return cbs

    def prepare_learner(self, learn):
        """
        Prepare learner after creation.

        Currently adds half precision, and brings model to gpu.
        :param learn: learner object
        :return: learner object
        """
        if not self.args.full_precision:
            learn.to_fp16()
        learn.model.cuda()
        return learn

    def set_lr(self, learn):
        """
        Set learning rate of learner.

        In ParallelMode can try to find optimal learning rate via lr_find.
        :param learn: learner object
        """
        if (
            self.args.auto_lr and not GPUManager.in_distributed_mode()
        ):  # auto_lr causes deadlock in distrib mode
            learn.freeze()
            with GPUManager.running_context(learn, self.args.gpu_ids):
                lr_min, lr_steep, lr_slide, lr_valley = learn.lr_find(
                    suggest_funcs=(fv.minimum, fv.steep, fv.slide, fv.valley),
                    show_plot=False,
                )
                lr = lr_valley
        else:
            lr = self.args.lr
        print("Learning rate is", lr)
        return lr

    def basic_train(self, run, learn, dls=None, save_model=True):
        """
        Train the learner, called by train_procedure method.

        :param run: str, run information string
        :param learn: learner object
        :param dls: train and valid dataloaders
        :param save_model: bool, whether to save the model weights after training
        """
        GPUManager.sync_distributed_process()
        if dls is not None:
            learn.dls = dls
        lr = self.set_lr(learn)
        train_cbs = self.get_train_cbs(run)
        print("Training model:", run)
        with GPUManager.running_context(learn, self.args.gpu_ids):
            if self.args.momentum is not None:
                mom = [self.args.momentum, self.args.momentum, self.args.momentum]
            else:
                mom = None
            learn.fine_tune(
                self.args.epochs,
                base_lr=lr,
                freeze_epochs=self.args.fepochs,
                wd=self.args.weight_decay,
                moms=mom,
                cbs=train_cbs,
            )
        if save_model:
            model_dir = fd.rank0_first(
                lambda: common.maybe_create(self.args.exp_logdir, learn.model_dir)
            )
            learn.save(os.path.join(model_dir, f"{run}{self.MODEL_SUFFIX}_lr{lr:.2e}"))

    def evaluate_and_correct_wl(self, learn, wl_items, run):
        """
        Evaluate and corrects weak labeled items, clears GPU memory (model and dls).

        :param learn: learner object
        :param wl_items: tuple of weak labeled items and labels
        :param run: str, run information string
        :return: corrected wl_items
        """
        GPUManager.sync_distributed_process()
        print("Evaluating WL data:", run)
        dl = learn.dls.test_dl(list(zip(*wl_items)), with_labels=True)
        with GPUManager.running_context(learn, self.args.gpu_ids):
            _, targs, decoded_preds = learn.get_preds(dl=dl, with_decoded=True)
        wl_items = self.correct_wl(wl_items, decoded_preds)
        GPUManager.clean_gpu_memory(dl, learn.dls, learn)
        return wl_items

    def evaluate_on_test_sets(self, learn, run):
        """
        Evaluate test sets, clears GPU memory held by test dl(s).

        Results are processed and stored in dict.
        :param learn: learner object
        :param run: str, run information string
        """
        for test_name, test_items_with_cls in self.get_test_items(merged=False):
            print("Testing model", run, "on", test_name)
            GPUManager.sync_distributed_process()
            dl = learn.dls.test_dl(list(zip(*test_items_with_cls)), with_labels=True)
            with GPUManager.running_context(learn, self.args.gpu_ids):
                interp = SimpleNamespace()
                interp.preds, interp.targs, interp.decoded = learn.get_preds(
                    dl=dl, with_decoded=True
                )
            interp.dl = dl
            interp = self.compute_metrics(interp, print_summary=True)
            if self.args.wandb:
                import wandb

                wandb.log({f"test_{k}": v for (k, v) in interp.metrics.items()})
            del interp.preds, interp.targs, interp.decoded, interp.dl
            GPUManager.clean_gpu_memory(dl)
            self.test_set_results[test_name][self.get_sorting_run_key(run)].append(
                interp
            )

    def print_metrics_summary(self):
        """Print summary of metrics."""
        raise NotImplementedError

    def precompute_metrics(self, interp):
        """
        Precompute values useful to speed up metrics calculations (e.g. class TP TN FP FN).

        :param interp: namespace with predictions, targets, decoded predictions
        :return: dict, with precomputed values
        """
        return {}

    def compute_metrics(self, interp, print_summary=False, with_ci=True):
        """
        Compute custom metrics and add results to interp object.

        Should return interp.
        :param interp: namespace with predictions, targets, decoded predictions
        :param with_ci: bool, compute confidence interval as well
        (used for confidence interval)
        :return: namespace with metrics results
        """
        prm = self.precompute_metrics(interp)
        interp.metrics = {
            mn: mfn(interp.preds, interp.targs, prm)
            for mn, mfn in self.cust_metrics.items()
        }
        if with_ci:
            with common.temporary_np_seed(self.args.seed):
                interp = self.compute_metrics_with_ci(interp)
        if print_summary:
            self.print_metrics_summary(interp.metrics)
        return interp

    def compute_metrics_with_ci(self, interp, ci_p=0.95):
        """
        Compute metrics with non-parametric confidence interval.

        :param interp: namespace with predictions, targets, decoded predictions
        :param ci_p: float, requested CI
        :param n: int, number of repetitions
        :return: tuple with list of repetition metrics results and resulting CI
        """
        if not hasattr(interp, "metrics"):
            interp = self.compute_metrics(interp, with_ci=False)
        all_metrics = [interp.metrics]
        bs = interp.targs.shape[0]
        for _ in tqdm(range(self.ci_bootstrap_n)):
            idxs = np.random.randint(0, bs, bs)
            s = SimpleNamespace()
            s.preds, s.targs, s.decoded, s.dl = (
                interp.preds[idxs],
                interp.targs[idxs],
                interp.decoded[idxs],
                interp.dl,
            )
            all_metrics.append(self.compute_metrics(s, with_ci=False).metrics)
        for mn in list(interp.metrics.keys()):
            interp.metrics[f"{mn}{self.PERF_CI}"] = non_param_ci(
                [am[mn] for am in all_metrics], ci_p
            )
        return interp

    def aggregate_test_performance(self, folds_res):
        """
        Merge metrics over different folds, computes mean and std.

        :param folds_res: list of metrics results over folds
        :return: dict with merged metrics results
        """
        merged = [
            (mn, [fr.metrics[mn] for fr in folds_res])
            for mn in folds_res[0].metrics.keys()
        ]
        return {
            mn: tuple(s.numpy() for s in tensors_mean_std(mres)) for mn, mres in merged
        }

    def generate_tests_reports(self):
        """Aggregate fold predictions, plots performance, save predictions results."""
        print("Aggregating test predictions ...")
        if not GPUManager.is_master_process():
            return
        for test_name in self.args.sl_tests:
            test_path = common.maybe_create(
                self.args.exp_logdir, f"{test_name}_{self.args.exp_name}"
            )
            for run, folds_results in self.test_set_results[test_name].items():
                agg = common.time_method(
                    self.aggregate_test_performance,
                    folds_results,
                    text="Test perfs aggregation",
                )
                if not self.args.no_plot:
                    self.plot_test_performance(test_path, run, agg)
                with open(os.path.join(test_path, f"{run}_test_results.p"), "wb") as f:
                    pickle.dump(agg, f)

    def train_model(self):
        """Perform training loop with cyclic weak labels refinement."""
        if self.args.inference:
            print("Inference only mode set, skipping train.")
            return
        sl_data, wl_data = self.get_train_items()
        for fold, tr, val in self.split_data(*sl_data):
            fold_suffix = f"__F{common.zero_pad(fold, self.args.nfolds)}__"
            learn, prev_run = self.train_procedure(tr, val, f"{fold_suffix}sl_only")

            if self.args.use_wl:
                for repeat in range(self.args.nrepeats):
                    wl_data = self.evaluate_and_correct_wl(learn, wl_data, prev_run)
                    repeat_prefix = (
                        f"__R{common.zero_pad(repeat, self.args.nrepeats)}__"
                    )
                    print(f"WL-SL train procedure {repeat + 1}/{self.args.nrepeats}")
                    learn, _ = self.train_procedure(
                        wl_data, val, f"{fold_suffix}wl_only", repeat_prefix
                    )
                    learn, prev_run = self.train_procedure(
                        tr, val, f"{fold_suffix}_wl_sl", repeat_prefix, learn
                    )
            GPUManager.clean_gpu_memory(learn.dls, learn)
        self.generate_tests_reports()
