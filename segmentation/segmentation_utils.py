import os
import sys
from pathlib import Path

import cv2

import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils
from classification.classification_utils import conf_mat


def common_segm_args(parser, pdef=dict(), phelp=dict()):
    parser.add_argument('--img-dir', type=str, default=pdef.get('--img-dir', "images"),
                        help=phelp.get('--img-dir', "Images dir"))
    parser.add_argument('--mask-dir', type=str, default=pdef.get('--mask-dir', "masks"),
                        help=phelp.get('--mask-dir', "Masks dir"))
    parser.add_argument('--mext', type=str, default=pdef.get('--mext', ".png"),
                        help=phelp.get('--mext', "Masks file extension"))
    parser.add_argument('--bg', type=int, default=pdef.get('--bg', 0),
                        help=phelp.get('--bg', "Background mask code, also index of bg cat"))
    return parser


def get_mask_path(img_path, img_dir, mask_dir, mext):
    # return img_path.replace(img_dir, mask_dir).replace(os.path.splitext(img_path)[1], mext)
    if type(img_path) is str: img_path = Path(img_path)
    return img_path.parent.parent/mask_dir/(img_path.stem + mext)


def load_img_and_mask(img_path, mask_path):
    return common.load_rgb_img(img_path), cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


def cls_perf(perf, inp, targ, cls_idx, cats, bg=None, axis=1):
    """If bg sets then computes perf without background"""
    if cls_idx == bg and cls_idx is not None: return torch.tensor(0).float()
    if axis is not None: inp = inp.argmax(dim=axis)
    if bg is not None:
        mask = targ != bg
        inp, targ = inp[mask], targ[mask]
    if cls_idx is None:
        res = [train_utils.get_cls_TP_TN_FP_FN(targ == c, inp == c) for c in range(0 if bg is None else 1, len(cats))]
        res = torch.cat([torch.tensor(r).float().unsqueeze(0) for r in res], dim=0).sum(axis=0).tolist()
        return torch.tensor(perf(*res)).float()
    else:
        return torch.tensor(perf(*train_utils.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx))).float()


def pixel_conf_mat(targs, preds, cats, normalize=True, epsilon=1e-8):
    return conf_mat(targs.flatten(), preds.flatten(), cats, normalize, epsilon)

