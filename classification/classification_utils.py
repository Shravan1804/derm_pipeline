import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import train_utils


def get_image_cls(img_path):
    return os.path.basename(os.path.dirname(img_path)) if type(img_path) is str else img_path.parent.name


def cls_perf(perf, inp, targ, cls_idx, cats, axis=-1):
    if cls_idx is not None:
        if axis is not None:
            inp = inp.argmax(dim=axis)
        return torch.tensor(perf(*train_utils.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx))).float()
    else:
        cls_res = [cls_perf(perf, inp, targ, c, cats, axis) for c in range(len(cats))]
        return torch.stack(cls_res).mean()


def conf_mat(targs, preds, cats, normalize=True, epsilon=1e-8):
    # https://github.com/fastai/fastai/blob/master/fastai/interpret.py
    x = torch.arange(0, len(cats))
    cm = ((preds == x[:, None]) & (targs == x[:, None, None])).long().sum(2)
    if normalize: cm = cm.float() / (cm.sum(axis=1)[:, None] + epsilon)
    return cm


