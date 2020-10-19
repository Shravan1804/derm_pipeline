import os
import math

import cv2
import numpy as np

import torch
import fastai.vision as fv


class DrawHelper:
    """Helper class to draw patches delimitations on images"""

    def __init__(self, thickness=1, style='dotted', gap=10):
        self.thickness = thickness
        self.style = style
        self.gap = gap

    def drawline(self, im, pt1, pt2, color):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5  # pythagoras hypotenuse
        pts = []
        for i in np.arange(0, dist, self.gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)

        if self.style == 'dotted':
            for p in pts:
                cv2.circle(im, p, self.thickness, color, -1)
        else:
            e = pts[0]
            for i, p in enumerate(pts):
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(im, s, e, color, self.thickness)

    def drawpoly(self, im, pts, color):
        e = pts[0]
        pts.append(pts.pop(0))
        for p in pts:
            s = e
            e = p
            self.drawline(im, s, e, color)

    def drawrect(self, im, pt1, pt2, color):
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        self.drawpoly(im, pts, color)

    def draw_patches(self, img_arr, h_w_pos, ps):
        for h, w in h_w_pos:
            s = (w, h)
            e = (w + ps, h + ps)
            self.drawrect(img_arr, s, e, (255, 255, 255))


def compute_side_overlap(n, ps):
    """Computes minimum overlap between patches, n is the total size (img height or width), ps is the patch size"""
    remainder = n % ps
    quotient = max(1, n // ps)
    overlap = math.ceil((ps - remainder) / quotient)
    return 0 if overlap == n else overlap


def maybe_resize(im_arr, ps):
    """Resize img only if dim smaller than the max patch size otherwise returns img unchanged"""
    h, w = im_arr.shape[:2]
    smallest = min(h, w)
    if smallest < ps:
        ratio = ps / smallest
        return cv2.resize(im_arr, (max(ps, int(w * ratio)), max(ps, int(h * ratio))))
    else:
        return im_arr


def patch_img(im_arr, ps=512):
    """Converts img_arr into a grid of patches"""
    im_arr = maybe_resize(im_arr, ps)
    im_h, im_w = im_arr.shape[:2]
    oh, ow = compute_side_overlap(im_h, ps), compute_side_overlap(im_w, ps)
    step_h, step_w = ps - oh, ps - ow
    grid_h = np.arange(start=0, stop=1 + im_h - ps, step=step_h)
    grid_w = np.arange(start=0, stop=1 + im_w - ps, step=step_w)
    grid_idx = [(h, w) for h in grid_h for w in grid_w]
    # if lst empty then image fits a single patch
    grid_idx = grid_idx if grid_idx else [(0, 0)]
    return ps, oh, ow, grid_idx, [im_arr[h:h + ps, w:w + ps] for h, w in grid_idx]


def batch_list(lst, bs):
    return [lst[i:min(len(lst), i + bs)] for i in range(0, len(lst), bs)]


def prepare_batch_for_inference(learn, batch):
    batch = [fv.Image(fv.pil2tensor(b, np.float32).div_(255)) for b in batch]
    return torch.cat([learn.data.one_item(b)[0] for b in batch], dim=0)


def get_learner(model_path='/home/navarinilab/models/20201019_body_loc.pkl'):
    assert os.path.exists(model_path) and model_path.endswith('.pkl'), f"Error with learner path: {model_path}"
    return fv.load_learner(os.path.dirname(model_path), os.path.basename(model_path))


def write_preds(im_arr, pos, preds_idx, preds_prob, labels):
    trad = {'arme': 'Arm', 'beine': 'Leg', 'fusse': 'Feet', 'hande': 'Hand', 'kopf': 'Head', 'other': 'Other',
            'stamm': 'Trunk', 'mean': 'Mean'}
    for (h, w), idxs, probs in zip(pos, preds_idx, preds_prob):
        for i, (idx, prob) in enumerate(zip(idxs, probs)):
            cv2.putText(im_arr, f'{trad[labels[idx]]}: {prob:.{3}f}', (50 + w, 25 + (i + 1) * 75 + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness=3)


def create_body_region_mapping(learn, im_arr, patch_size=512, batch_size=12, topk=3):
    patch_size, oh, ow, pos, patches = patch_img(im_arr, ps=512)
    batches = batch_list(patches, batch_size)

    preds = []
    for batch in batches:
        batch = prepare_batch_for_inference(learn, batch)
        preds.append(learn.pred_batch(batch=[batch, -1]).cpu().detach())
    preds = torch.cat(preds, dim=0)
    topk_p, topk_idx = preds.topk(topk, axis=1)

    out_img = im_arr.copy()
    write_preds(out_img, pos, topk_idx, topk_p, learn.data.classes)
    DrawHelper().draw_patches(out_img, pos, patch_size)
    return out_img
