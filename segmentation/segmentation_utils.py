import os
import sys

import cv2

import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import train_utils
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


def get_mask_path(img_path, img_dir='images', mask_dir='masks', mext='.png'):
    if type(img_path) is str:
        file, ext = os.path.splitext(os.path.basename(img_path))
        return img_path.replace(f'{img_dir}/{file}{ext}', f'{mask_dir}/{file}{mext}')
    else:
        return img_path.parent.parent/mask_dir/(img_path.stem + mext)


def load_img_and_mask(img_path, mask_path):
    return common.load_rgb_img(img_path), common.load_img(mask_path)


def cls_perf(perf, inp, targ, cls_idx, cats, bg=None, axis=1):
    """If bg sets then computes perf without background"""
    targ = targ.as_subclass(torch.Tensor)
    if cls_idx is not None:
        if cls_idx == bg: return torch.tensor(0).float()
        if axis is not None: inp = inp.argmax(dim=axis)
        if bg is not None:
            mask = targ != bg
            inp, targ = inp[mask], targ[mask]
        return torch.tensor(perf(*train_utils.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx))).float()
    else:
        cls_res = [cls_perf(perf, inp, targ, c, cats, bg, axis) for c in range(0 if bg is None else 1, len(cats))]
        return torch.stack(cls_res).mean()


def pixel_conf_mat(targs, preds, cats, normalize=True, epsilon=1e-8):
    return conf_mat(targs.flatten(), preds.flatten(), cats, normalize, epsilon)

