import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ..classification.classification_utils import conf_mat
from ..general import common_img as cimg
from ..training import metrics


def common_segm_args(parser, pdef=dict(), phelp=dict()):
    """Add usual segmentation arguments.

    :param parser: argparser
    :param pdef: dict, default arguments values
    :param phelp: dict, argument help strings
    :return: argparser
    """
    parser.add_argument(
        "--img-dir",
        type=str,
        default=pdef.get("--img-dir", "images"),
        help=phelp.get("--img-dir", "Images dir"),
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=pdef.get("--mask-dir", "masks"),
        help=phelp.get("--mask-dir", "Masks dir"),
    )
    parser.add_argument(
        "--mext",
        type=str,
        default=pdef.get("--mext", ".png"),
        help=phelp.get("--mext", "Masks file extension"),
    )
    parser.add_argument(
        "--bg",
        type=int,
        default=pdef.get("--bg", 0),
        help=phelp.get("--bg", "Background mask code, also index of bg cat"),
    )
    return parser


def get_mask_path(img_path, img_dir="images", mask_dir="masks", mext=".png"):
    """Get mask path from image path.

    :param img_path: str, image path
    :param img_dir: str, image directory
    :param mask_dir: str, mask directory
    :param mext: str, mask extension
    :return: str, mask path if file exists else None
    """
    if not os.path.exists(img_path) or f"{img_dir}/" not in str(img_path):
        return None
    elif type(img_path) is str:
        file, ext = os.path.splitext(os.path.basename(img_path))
        mask_path = img_path.replace(
            f"{img_dir}/{file}{ext}", f"{mask_dir}/{file}{mext}"
        )
    else:
        mask_path = img_path.parent.parent / mask_dir / (img_path.stem + mext)
    return mask_path if os.path.exists(mask_path) else None


def load_img_and_mask(img_path, mask_path):
    """Load image and mask from their paths.

    :param img_path: str, image path
    :param mask_path: str, mask path
    :return: tuple of image and mask arrays
    """
    return cimg.load_img(img_path), cimg.load_img(mask_path)


def cls_perf(perf, inp, targ, cidx, cats, bg=None, axis=1, precomp={}):
    """Compute classification performance.

    If bg not None then computes perf without background
    :param perf: function to call on inp and targ
    :param inp: tensor, predictions
    :param targ: tensor, ground truth
    :param cidx: int, category id to compute performance for
    :param cats: list, categories
    :param bg: int, index of background cat. Will be masked if set
    :param axis: int, axis on which to perform argmax to decode prediction
    :param precomp: dict, precomputed values to speed-up metrics computation
    :return: tensor, performance results
    """
    targ = targ.as_subclass(torch.Tensor)
    if cidx is not None:
        if cidx == bg:
            return torch.tensor(0).float()
        if axis is not None:
            inp = inp.argmax(dim=axis)
        if bg is not None:
            mask = targ != bg
            inp, targ = inp[mask], targ[mask]
        TP_TN_FP_FN = (
            precomp[(cidx, bg)]
            if precomp
            else metrics.get_cls_TP_TN_FP_FN(targ == cidx, inp == cidx)
        )
        return torch.tensor(perf(*TP_TN_FP_FN)).float()
    else:
        cidxs = [c for c in range(len(cats)) if bg is None or c != bg]
        cls_res = [cls_perf(perf, inp, targ, c, cats, bg, axis, precomp) for c in cidxs]
        return torch.stack(cls_res).mean()


def pixel_conf_mat(targs, preds, cats, normalize=True, epsilon=1e-8):
    """Compute pixel-wise confusion matrix from prediction and targs.

    :param targs: tensor, ground truth size, B
    :param preds: tensor, model decoded predictions, size B
    :param cats: list, categories, size N
    :param normalize: bool, whether to normalize confusion matrix
    :param epsilon: float, against division by zero
    :return: tensor, confusion matrix, size N x N
    """
    return conf_mat(targs.flatten(), preds.flatten(), cats, normalize, epsilon)


def create_dummy_segm_dataset(
    path,
    n_classes,
    dataset_name="dummy_segm_dset",
    img_size=(64, 64),
    n_samples=10,
    train_ratio=0.8,
):
    path = Path(path) / dataset_name
    train_size = int(n_samples * train_ratio)
    splits = {"train": train_size, "test": n_samples - train_size}

    for split, num_samples in splits.items():
        img_dir = path / split / "images"
        mask_dir = path / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            img = np.random.randint(0, 256, (*img_size, 3), dtype=np.uint8)
            mask = np.random.randint(0, n_classes, img_size, dtype=np.uint8)

            # Save image and mask
            img_path = img_dir / f"image_{i:03}.jpg"
            mask_path = mask_dir / f"image_{i:03}.png"

            cv2.imwrite(str(img_path), img)
            cv2.imwrite(str(mask_path), mask)

    print(f"Dummy segmentation dataset created at {path}")
