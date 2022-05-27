#!/usr/bin/env python

"""classification_utils.py: Contains methods useful for image classification"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from training import metrics


def cls_perf(perf, inp, targ, cls_idx, cats, axis=-1, precomp={}):
    """Function used to compute classification performance
    :param perf: function to call on inp and targ
    :param inp: tensor, predictions
    :param targ: tensor, ground truth
    :param cls_idx: int, category id to compute performance for
    :param cats: list, categories
    :param axis: int, axis on which to perform argmax to decode prediction
    :param precomp: dict, precomputed values to speed-up metrics computation
    :return: tensor, performance results
    """
    if cls_idx is not None:
        if axis is not None:
            inp = inp.argmax(dim=axis)
            TP_TN_FP_FN = precomp[cls_idx] if precomp else metrics.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx)
        return torch.tensor(perf(*TP_TN_FP_FN)).float()
    else:
        cls_res = [cls_perf(perf, inp, targ, c, cats, axis, precomp) for c in range(len(cats))]
        return torch.stack(cls_res).mean()


def conf_mat(targs, preds, cats, normalize=True, epsilon=1e-8):
    """Computes confusion matrix from prediction and targs
    source https://github.com/fastai/fastai/blob/master/fastai/interpret.py
    :param targs: tensor, ground truth size, B
    :param preds: tensor, model decoded predictions, size B
    :param cats: list, categories, size N
    :param normalize: bool, whether to normalize confusion matrix
    :param epsilon: float, against division by zero
    :return: tensor, confusion matrix, size N x N
    """
    x = torch.arange(0, len(cats))
    cm = ((preds == x[:, None]) & (targs == x[:, None, None])).long().sum(2)
    if normalize: cm = cm.float() / (cm.sum(axis=1)[:, None] + epsilon)
    return cm


class LinearClassifier(torch.nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels

        self.dropout = torch.nn.Dropout(0.3)

        self.linear = torch.nn.Linear(dim, 128)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.relu = torch.nn.ReLU()

        self.dropout2 = torch.nn.Dropout(0.3)

        self.linear2 = torch.nn.Linear(128, num_labels)
        self.linear2.weight.data.normal_(mean=0.0, std=0.01)
        self.linear2.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # Dropout
        x = self.dropout(x)
        # 1. linear layer
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout2(x)
        # 2. linear layer
        x = self.linear2(x)
        return x
