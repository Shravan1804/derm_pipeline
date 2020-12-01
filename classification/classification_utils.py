import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import train_utils


def get_image_cls(img_path):
    return os.path.basename(os.path.dirname(img_path)) if type(img_path) is str else return img_path.parent.name


def cls_perf(perf, inp, targ, cls_idx, cats, axis=-1):
    if axis is not None:
        inp = inp.argmax(dim=axis)
    if cls_idx is None:
        res = [train_utils.get_cls_TP_TN_FP_FN(targ == c, inp == c) for c in range(len(cats))]
        res = torch.cat([torch.tensor(r).unsqueeze(0) for r in res], dim=0).sum(axis=0).tolist()
        return torch.tensor(perf(*res))
    else:
        return torch.tensor(perf(*train_utils.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx)))


def conf_mat(cats, preds, targs, normalize=True, epsilon=1e-8):
    # https://github.com/fastai/fastai/blob/master/fastai/interpret.py
    x = torch.arange(0, len(cats))
    cm = ((preds == x[:, None]) & (targs == x[:, None, None])).long().sum(2)
    if normalize: cm = cm.float() / (cm.sum(axis=1)[:, None] + epsilon)
    return cm


