#!/usr/bin/env python

"""train_utils.py: Helper functions to train model"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)


import os
import gc
import sys
from collections import defaultdict

import torch
import fastai.vision.all as fv
import fastai.distributed as fd
import fastai.callback.tensorboard as fc

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))


def tensors_mean_std(tensor_lst):
    """Takes mean and std of tensor list
    :param tensor_lst: list of tensors
    :return: tuple of tensors, (mean, std)
    """
    tensors = torch.cat([t.unsqueeze(0) for t in tensor_lst], dim=0)
    mean = tensors.mean(axis=0)
    std = tensors.std(axis=0) if len(tensor_lst) > 1 else torch.zeros_like(mean)
    return mean, std


def non_param_ci(tensor_lst, ci_p):
    """Computes the non-parametric confidence interval when predicted variable is not necessarily normally distributed
    :param tensor_lst: list of tensors, metrics results
    :param ci_p: float, e.g. .95 for CI95
    :return: tuple, lower and higher bound of CI
    """
    alpha = 1 - ci_p
    low, high = alpha/2, 1-alpha/2
    return torch.quantile(torch.stack(tensor_lst).float(), torch.tensor([low, high]), dim=0)


def split_model(model, splits):
    """Inspired from fastai 1, splits model on requested top level children
    Used for freezing the model and adaptive learning rates
    :param model: torch module
    :param splits: list of top level module children, where split should occur
    :return: list of children sequence based on requested splits
    """
    top_children = list(model.children())
    idxs = [top_children.index(split) for split in splits]
    assert idxs == sorted(idxs), f"Provided splits ({splits}) are not sorted."
    assert len(idxs) > 0, f"Provided splits ({splits}) not found in top level children: {top_children}"
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(top_children): idxs.append(len(top_children))
    return [
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            torch.nn.Sequential(*top_children[i:j]))
        for i, j in zip(idxs[:-1], idxs[1:])
    ]


def load_custom_pretrained_weights(model, weights_path):
    """Load model weights from weight file
    :param model: torch module
    :param weights_path: str, path to .pth weights
    """
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
    """Prints a list of the Tensors being tracked by the garbage collector.
    # inspired from https://forums.fast.ai/t/gpu-memory-not-being-freed-after-training-is-over/10265/7
    :param gpu_only: bool, show only tensors on gpu
    """
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
    """Helper class used to interact with gpus"""
    @staticmethod
    def clean_gpu_memory(*items):
        """Remove all provided items from gpu and call garbage collector
        :param items: list of item to be cleaned
        """
        for item in items: del item
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def default_gpu_device_ids():
        """Returns all gpu device idxs
        :return: list of gpus indexes
        """
        assert torch.cuda.is_available(), "Cannot run without CUDA device"
        return list(range(torch.cuda.device_count()))

    @staticmethod
    def in_parallel_mode():
        """Checks if running in parallel mode
        :return: bool, running in parallel mode
        """
        return not GPUManager.in_distributed_mode() and torch.cuda.device_count() > 1

    @staticmethod
    def in_distributed_mode():
        """Checks if running in distributed mode. True when script was launched with distributed_launch.py
        :return: bool, running in distributed mode
        """
        return os.environ.get('RANK', None) is not None

    @staticmethod
    def distributed_rank():
        """Gives the process gpu rank
        :return: int, gpu rank
        """
        return int(os.environ.get('RANK', 0))

    @staticmethod
    def is_master_process():
        """Checks if process is the master process (gpu rank is 0)
        :return: bool, is master process
        """
        return GPUManager.distributed_rank() == 0 if GPUManager.in_distributed_mode() else True

    @staticmethod
    def init_distributed_process():
        """Starts distributed mode. Should be called before beginning of training."""
        if GPUManager.in_distributed_mode():
            rank = GPUManager.distributed_rank()
            fd.setup_distrib(rank)
            torch.cuda.set_device(rank)

    @staticmethod
    def sync_distributed_process():
        """Create barrier so that all process sync"""
        if GPUManager.in_distributed_mode(): fv.distrib_barrier()

    @staticmethod
    def running_context(learn, device_ids=None):
        """Creates appropriate gpu running context (parallel or distributed)
        :param learn: learner object
        :param device_ids: available gpu device indices
        :return: running context manager
        """
        device_ids = list(range(torch.cuda.device_count())) if device_ids is None else device_ids
        return learn.distrib_ctx() if GPUManager.in_distributed_mode() else learn.parallel_ctx(device_ids)


class ImageTBCb(fc.TensorBoardBaseCallback):
    """Extension of tensorboard callback ensuring that custom category metrics are also plotted."""
    def __init__(self, log_dir, run_info, grouped_metrics, all_cats):
        """Creates tensorboard callback
        :param log_dir: str, directory path where TB logs can be saved
        :param run_info: str, run information to label logs
        :param grouped_metrics: list of metrics names to group together in same graph
        :param all_cats: list, categories including all cats code
        """
        super().__init__()
        self.log_dir = log_dir
        self.run_info = run_info
        self.grouped_metrics = grouped_metrics
        self.all_cats = all_cats

    def can_run(self):
        """Checks if callback can run. Must be master process"""
        return not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")\
               and GPUManager.is_master_process()

    def before_fit(self):
        """Sets up log writer"""
        self.run = self.can_run()
        if self.run: self._setup_writer()

    def after_batch(self):
        """Write batch loss metrics and hyperparameters"""
        if not self.run: return
        # if no self.smooth_loss then -1: when loss is nan, Recorder does not set smooth loss causing exception else
        self.writer.add_scalar(f'{self.run_info}_Loss/train_loss', getattr(self, "smooth_loss", -1), self.train_iter)
        for i, h in enumerate(self.opt.hypers):
            for k, v in h.items(): self.writer.add_scalar(f'{self.run_info}_Opt_hyper/{k}_{i}', v, self.train_iter)

    def after_epoch(self):
        """Writes epochs metrics"""
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
