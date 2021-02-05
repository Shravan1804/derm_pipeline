import os
import sys
import math

import cv2
import numpy as np
from pycocotools import mask as pycoco_mask

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common


def non_binary_mask_to_rles(mask):
    return [(k, pycoco_mask.encode(np.asfortranarray(mask == k, dtype=np.uint8))) for k in np.unique(mask)]


def rles_to_non_binary_mask(rles):
    return sum([k*pycoco_mask.decode(rle) for k, rle in rles])


def rm_small_objs_from_non_bin_mask(non_binary_mask, min_size, cats_with_bg, bg=0):
    res = np.zeros(non_binary_mask.shape, dtype=np.uint8)
    for c, _ in enumerate(cats_with_bg):
        if c == bg: continue  # background
        binary_mask = rm_small_objs_from_bin_mask(non_binary_mask == c, min_size)
        res[binary_mask > 0] = c
    return res


def rm_small_objs_from_bin_mask(binary_mask, min_size):
    nb_obj, obj_labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
    if nb_obj < 2: return binary_mask  # only background
    obj_ids, inverse, sizes = np.unique(obj_labels, return_inverse=True, return_counts=True)
    for i, size in enumerate(sizes):
        if size < min_size: obj_ids[i] = 0  # set this component to background

    mask_cleaned = np.reshape(obj_ids[inverse], binary_mask.shape)
    mask_cleaned[mask_cleaned > 0] = 1
    return mask_cleaned.astype(np.uint8)


def load_mask_array(mask_path): return cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)


def resize_mask(mask, new_size):
    return cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)


def crop_im(im, bbox):
    wmin, hmin, wmax, hmax = bbox
    return im[hmin:hmax+1, wmin:wmax+1]


def get_obj_proportion(mask, bg=0):
    u, uc = np.unique(mask, return_counts=True)
    return uc[u != bg].sum() / uc.sum(), u, uc


def get_bbox(cond):
    pos = cond.nonzero()
    if pos[0].size == 0:  # no objects
        return None
    hmin, hmax = np.min(pos[0]), np.max(pos[0])
    wmin, wmax = np.min(pos[1]), np.max(pos[1])
    return np.array((wmin, hmin, wmax, hmax))


def bbox_side_size(bbox):
    """Returns w, h"""
    return np.array((bbox[2] - bbox[0], bbox[3] - bbox[1]))


def bbox_area(bbox):
    w, h = bbox_side_size(bbox)
    return w*h


def bbox_centroid(bbox):
    wmin, hmin, wmax, hmax = bbox
    w, h = bbox_side_size(bbox)
    cX = wmin + math.floor(w/2)
    cY = hmin + math.floor(h/2)
    return np.array((cX, cY))


def ensure_bbox_side_min_size(smin, smax, side_range, min_size):
    if smax - smin < min_size:
        diff = min_size - (smax - smin)
        bbox = (smin, 0, smax, 0)
        bbox = grow_bbox(bbox, side_range, (0, 0), diff, rand=False)
        return np.array((bbox[0], bbox[2]))
    else:
        return np.array((smin, smax))


def ensure_bbox_min_size(im_shape, bbox, min_size):
    wmin, hmin, wmax, hmax = bbox
    wmin, wmax = ensure_bbox_side_min_size(wmin, wmax, (0, im_shape[1]), min_size)
    hmin, hmax = ensure_bbox_side_min_size(hmin, hmax, (0, im_shape[0]), min_size)
    return np.array((wmin, hmin, wmax, hmax))


def bboxes_have_intersect(bbox1, bbox2):
    # https://gamedev.stackexchange.com/questions/586/what-is-the-fastest-way-to-work-out-2d-bounding-box-intersection
    # left/right for w (x axis) wmin/wmax, top/bottom for h (y axis) hmin/hmax
    # return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom || r2.bottom < r1.top);
    wmin1, hmin1, wmax1, hmax1 = bbox1
    wmin2, hmin2, wmax2, hmax2 = bbox2
    return not (wmin2 > wmax1 or wmax2 < wmin1 or hmin2 > hmax1 or hmax2 < hmin1)


def bboxes_intersection(bbox1, bbox2):
    if not bboxes_have_intersect(bbox1, bbox2):
        return None
    wmin1, hmin1, wmax1, hmax1 = bbox1
    wmin2, hmin2, wmax2, hmax2 = bbox2
    return np.array((max(wmin1, wmin2), max(hmin1, hmin2), min(wmax1, wmax2), min(hmax1, hmax2)))


def bboxes_union(bbox1, bbox2):
    wmin1, hmin1, wmax1, hmax1 = bbox1
    wmin2, hmin2, wmax2, hmax2 = bbox2
    return np.array((min(wmin1, wmin2), min(hmin1, hmin2), max(wmax1, wmax2), max(hmax1, hmax2)))


def bboxes_overlap(bbox1, bbox2):
    intersect = bboxes_intersection(bbox1, bbox2)
    if intersect is None:
        return 0
    area1, area2, intersect_area = bbox_area(bbox1), bbox_area(bbox2), bbox_area(intersect)
    return intersect_area/(area1 + area2 - intersect_area)


def merge_bboxes_based_on_overlap(bboxes, max_overlap):
    merged = common.merge(lst=bboxes,
                          cond_fn=lambda a, b: bboxes_overlap(a, b) >= max_overlap,
                          merge_fn=lambda a, b: bboxes_union(a, b))
    return merged


def merge_bboxes_based_on_centroids_dist(centroids, bboxes, min_dist):
    merged = common.merge(lst=list(zip(centroids, bboxes)),
                          cond_fn=lambda a, b: np.square(a[0] - b[0]).sum() <= min_dist ** 2,
                          merge_fn=lambda a, b: ((a[0] + b[0]) // 2, bboxes_union(a[1], b[1])))
    return merged


def grow_bbox(start_bbox, wrange, hrange, total_growth, rd=2, rand=True):
    """
    (Randomly) increase bbox on either sides by up to total_growth px in the im_dim
    rd is the coords reducing/growing split index, first group is reducing, second is growing
    """
    assert wrange[0] <= start_bbox[0] <= start_bbox[2] <= wrange[1], f"BBOX w ({start_bbox}) does not fit {wrange}"
    assert hrange[0] <= start_bbox[1] <= start_bbox[3] <= hrange[1], f"BBOX h ({start_bbox}) does not fit {hrange}"
    coords = np.array(start_bbox)
    # start_bbox is wmin, hmin, wmax, hmax
    low = np.resize(np.array([wrange[0], hrange[0]]), coords.size)
    high = np.resize(np.array([wrange[1], hrange[1]]), coords.size)
    can_grow = (coords > low) & (coords < high)

    changed = True
    while can_grow.any() and changed and total_growth > 0:
        growth = np.zeros_like(coords)
        growth[can_grow] = common.int_to_bins(total_growth, can_grow.sum(), rand=rand)
        growth[:rd] *= -1

        new_coords = coords + growth
        new_coords = new_coords.clip(low, high)

        rest = (growth - np.abs(coords - new_coords)).sum()
        changed = rest != total_growth
        total_growth = rest

        coords[can_grow] = new_coords[can_grow]
        can_grow = (coords > low) & (coords < high)

    return coords


def get_centroids_with_bboxes(mask, kern=(5, 5), dilate_it=10, bg=0):
    """Returns list of tuples (cX, cY) of centroids coords with the contours bounding boxes"""
    m = cv2.dilate((mask != bg).astype(np.uint8), np.ones(kern, np.uint8), iterations=dilate_it)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroids, bboxes = [], []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            print("WARNING, error while computing centroid, skipping blob.")
            continue
        centroids.append(np.array((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))))    #(cX, cY)
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append(np.array([x, y, x+w, y+h]))
    return centroids, bboxes


def grow_crop_bbox_to_thresh(mask, bbox, thresh, rand, bg=0):
    """Increases the bbox size if obj proportion above thresh (adds more background, thus reduces obj proportion)"""
    cropped_mask = crop_im(mask, bbox)
    obj_prop, u, uc = get_obj_proportion(cropped_mask, bg=bg)
    if obj_prop > thresh:
        # alpha is the background area to be added to have equality with thresh
        alpha = uc[u != 0].sum()/thresh - uc.sum()
        N = bbox[3] - bbox[1] + bbox[2] - bbox[0]
        # base margin is the margin to be added on all 4 sides to reach alpha
        base_margin = max(0, math.floor((-N+math.sqrt(N**2+4*alpha))/4))
        bbox = grow_bbox(bbox, (0, mask.shape[1]), (0, mask.shape[0]), base_margin*4, rand=rand)
    return bbox


def extract_bboxes_from_img_mask(im, mask, bboxes):
    cropped_imgs, cropped_masks = [], []
    for bbox in bboxes:
        cropped_imgs.append(crop_im(im, bbox))
        cropped_masks.append(crop_im(mask, bbox))
    return cropped_imgs, cropped_masks


def crop_img_and_mask_to_objs(im, mask, thresh=.01, rand=True, single=True, only_bboxes=False, bg=0,
                              min_obj_area=36, min_crop_side_size=128, max_crop_overlap=.6):
    """If rand is true will randomly grow margins to try fitting threshold object proportion
    If single is false will return many masks of single nearby objects"""
    if not (mask > 0).any():
        raise Exception("Provided mask does not contain any objects.")

    if single:
        crop_bbox = grow_crop_bbox_to_thresh(mask, get_bbox(mask > 0), thresh, rand, bg=bg)
        if only_bboxes:
            return crop_bbox,
        else:
            return (crop_im(im, crop_bbox), ), (crop_im(mask, crop_bbox), ), (crop_bbox, )

    centroids, bboxes = get_centroids_with_bboxes(mask)
    cleaned_cropped_bboxes = []
    for centroid, bbox in zip(centroids, bboxes):
        min_bbox = get_bbox(crop_im(mask, bbox) != bg)
        bbox = bbox if min_bbox is None else np.tile(bbox[:2], 2) + np.array(min_bbox)
        if bbox_area(bbox) < min_obj_area:
            continue
        bbox = grow_crop_bbox_to_thresh(mask, bbox, thresh, rand, bg=bg)
        cleaned_cropped_bboxes.append(ensure_bbox_min_size(mask.shape, bbox, min_crop_side_size))
    cleaned_cropped_bboxes = merge_bboxes_based_on_overlap(cleaned_cropped_bboxes, max_crop_overlap)

    if only_bboxes:
        return cleaned_cropped_bboxes
    else:
        im_crops, mask_crops = extract_bboxes_from_img_mask(im, mask, cleaned_cropped_bboxes)
        return im_crops, mask_crops, cleaned_cropped_bboxes


def im_mask_on_ax(ax, im, mask, title=None):
    common.img_on_ax(im, ax, title=title)
    ax.imshow(mask, cmap='jet', alpha=0.4)


def show_im_with_masks(im, mask, cats):
    _, axs = common.prepare_img_axs(im.shape[0]/im.shape[1], 1, len(cats) + 1)
    common.img_on_ax(im, axs[0], title='Original image')
    for cls_idx, cat in enumerate(cats):
        m = (mask == cls_idx).astype(np.uint8)
        obj_prop = get_obj_proportion(m)[0]
        common.img_on_ax(m, axs[cls_idx + 1], title=f'{cat} mask ({obj_prop:.{3}f}%)')


def show_im_with_mask_overlaid(im, mask, cats, bg=0, show_bg=False):
    cls_idxs = np.unique(mask)
    if not show_bg:
        cls_idxs = cls_idxs[cls_idxs != bg]
    cls_labels = np.array(cats)[cls_idxs]
    ncols = cls_labels.size + 1
    _, axs = common.prepare_img_axs(im.shape[0]/im.shape[1], 1, ncols)
    common.img_on_ax(im, axs[0], title='Original image')
    for ax, cls_idx, cat in zip(axs[1:], cls_idxs, cls_labels):
        m = (mask == cls_idx).astype(np.uint8)
        obj_prop = get_obj_proportion(m)[0]
        im_mask_on_ax(ax, im, m, title=f'{cat} segmentation ({obj_prop:.{3}f}%)')


def show_im_with_crops_bboxes(im, mask, obj_threshs=[.01], ncols=4, bg=0):
    nrows = math.ceil(len(obj_threshs)/ncols) * 2
    _, axs = common.prepare_img_axs(im.shape[0]/im.shape[1], nrows, ncols)
    for r, rand in enumerate([False, True]):
        for i, t in enumerate(obj_threshs):
            im_arr = im.copy()
            m = mask.copy()
            crop_bboxes = crop_img_and_mask_to_objs(im_arr, m, t, rand, single=False, only_bboxes=True, bg=bg)
            for (wmin, hmin, wmax, hmax) in crop_bboxes:
                cv2.rectangle(im_arr, (wmin, hmin), (wmax, hmax), 255, 5)
            im_mask_on_ax(axs[r*ncols + i], im_arr, m, title=f' {"Rand bbox" if rand else "Bbox"} thresh {t}')

