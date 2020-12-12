import os
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO

import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils
from object_detection import coco_format
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


def segm_dataset_to_coco_format(segm_masks, cats, scores=False, bg=0, ret_json=False):
    segm_masks = segm_masks if type(segm_masks) is np.ndarray else segm_masks.numpy()
    dataset = coco_format.get_default_dataset()
    cats = [(c, i) for i, c in enumerate(cats) if i != bg]
    dataset['categories'] = coco_format.get_categories(*zip(*cats))
    ann_id = 1  # MUST start at 1 since pycocotools.cocoeval uses detId to track matches and checks with > 0
    for img_id, non_binary_mask in enumerate(segm_masks):
        img_id += 1  # to be on the safe side (same idea as ann_id)
        dataset['images'].append(coco_format.get_img_record(img_id, f'{img_id}.jpg', non_binary_mask.shape))
        obj_cats = np.array([t for t in np.unique(non_binary_mask) if t != bg])
        if not obj_cats.size: continue
        cat_masks = (non_binary_mask == obj_cats[:, None, None]).astype(np.uint8)
        obj_cats_masks = tuple(cv2.connectedComponents(cmsk) for cmsk in cat_masks)
        ann_id, img_annos = coco_format.get_annos_from_objs_mask(img_id, ann_id, obj_cats, obj_cats_masks, scores)
        dataset['annotations'].extend(img_annos)
    if ret_json:
        return dataset
    else:
        coco_ds = COCO()
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds


