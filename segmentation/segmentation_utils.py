import os
import sys

import cv2

import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils
from classification.classification_utils import conf_mat


def get_mask_path(img_path, img_dir, mask_dir, mext):
    if type(img_path) is str:
        return img_path.replace(img_dir, mask_dir).replace(os.path.splitext(img_path)[1], mext)
    else:
        return img_path.parent.parent/mask_dir/(img_path.stem + mext)


def load_img_and_mask(img_path, mask_path):
    return common.load_rgb_img(img_path), cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


def cls_perf(perf, inp, targ, cls_idx, cats, bg=None, axis=1):
    """If bg sets then computes perf without background"""
    assert bg != cls_idx or cls_idx is None, f"Cannot compute class {cls_idx} perf as bg = {bg}"
    if axis is not None:
        inp = inp.argmax(dim=axis)
    if bg is not None:
        mask = targ != bg
        inp, targ = inp[mask], targ[mask]
    if cls_idx is None:
        res = [train_utils.get_cls_TP_TN_FP_FN(targ == c, inp == c) for c in range(0 if bg is None else 1, len(cats))]
        res = torch.cat([torch.tensor(r).unsqueeze(0) for r in res], dim=0).sum(axis=0).tolist()
        return torch.tensor(perf(*res))
    else:
        return torch.tensor(perf(*train_utils.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx)))


def pixel_conf_mat(cats, preds, targs, normalize=True, epsilon=1e-8):
    return conf_mat(cats, preds.flatten(), targs.flatten(), normalize, epsilon)

