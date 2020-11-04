import math

import cv2
import numpy as np

import common
import img_utils


def get_obj_proportion(mask, bg=0):
    u, uc = np.unique(mask, return_counts=True)
    return uc[u != bg].sum() / uc.sum(), u, uc


def get_bbox(cond):
    pos = cond.nonzero()
    if pos[0].size == 0:  # no objects
        return None
    hmin, hmax = np.min(pos[0]), np.max(pos[0])
    wmin, wmax = np.min(pos[1]), np.max(pos[1])
    return wmin, hmin, wmax, hmax


def merge_bboxes(centroids, bboxes, min_dist):
    merged = common.merge(list(zip(centroids, bboxes)),
                          lambda a, b: np.square(a[0] - b[0]).sum() <= min_dist ** 2,
                          lambda a, b: ((a[0] + b[0]) // 2, np.array([min(a[1][0], b[1][0]), min(a[1][1], b[1][1]),
                                                                      max(a[1][2], b[1][2]), max(a[1][3], b[1][3])])))
    return merged


def grow_bbox(start_bbox, im_dim, total_growth, rd=2, rand=True):
    """
    Randomly increase bbox on either sides by up to total_growth px in the im_dim
    rd is the coords reducing/growing split index, first group is reducing, second is growing
    """
    assert 0 <= start_bbox[0] <= start_bbox[2] <= im_dim[1], f"Start_bbox ({start_bbox}) does not fit in image {im_dim}"
    assert 0 <= start_bbox[1] <= start_bbox[3] <= im_dim[0], f"Start_bbox ({start_bbox}) does not fit in image {im_dim}"
    coords = np.array(start_bbox)
    # start_bbox is wmin, hmin, wmax, hmax
    low, high = np.zeros_like(coords), np.resize(np.array([im_dim[1], im_dim[0]]), coords.size)
    can_grow = (coords >= low) & (coords <= high)

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
        can_grow = (coords >= low) & (coords <= high)

    return coords.tolist()


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
        centroids.append(np.array((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))))
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append(np.array([x, y, x+w, y+h]))
    return centroids, bboxes


def fit_crop_bbox_to_thresh(mask, bbox, thresh, rand, bg=0):
    cropped_mask = img_utils.crop_im(mask, bbox)
    obj_prop, u, uc = get_obj_proportion(cropped_mask, bg=bg)
    if obj_prop > thresh:
        # alpha is the background area to be added to have equality with thresh
        alpha = uc[u != 0].sum()/thresh - uc.sum()
        N = bbox[3] - bbox[1] + bbox[2] - bbox[0]
        # base margin is the margin to be added on all 4 sides to reach alpha
        base_margin = max(0, math.floor((-N+math.sqrt(N**2+4*alpha))/4))
        bbox = grow_bbox(bbox, mask.shape, base_margin*4, rand=rand)
    return bbox


def crop_img_and_mask_to_objs(im, mask, thresh=.01, rand=True, single=True, only_bboxes=False, bg=0):
    """If rand is true will randomly grow margins to try fitting threshold object proportion
    If single is false will return many masks of single nearby objects"""
    if not (mask > 0).any():
        raise Exception("Provided mask does not contain any objects.")

    if single:
        crop_bbox = fit_crop_bbox_to_thresh(mask, get_bbox(mask > 0), thresh, rand, bg=bg)
        if only_bboxes:
            return [crop_bbox]
        else:
            return [img_utils.crop_im(im, crop_bbox)], [img_utils.crop_im(mask, crop_bbox)], [crop_bbox]

    centroids, bboxes = get_centroids_with_bboxes(mask)
    cropped_imgs, cropped_masks, crop_bboxes = [], [], []
    for centroid, bbox in zip(centroids, bboxes):
        min_bbox = get_bbox(img_utils.crop_im(mask, bbox) != bg)
        bbox = bbox if min_bbox is None else np.tile(bbox[:2], 2) + np.array(min_bbox)
        bbox = fit_crop_bbox_to_thresh(mask, bbox, thresh, rand, bg=bg)
        if not only_bboxes:
            cropped_imgs.append(img_utils.crop_im(im, bbox))
            cropped_masks.append(img_utils.crop_im(mask, bbox))
        crop_bboxes.append(bbox)
    return crop_bboxes if only_bboxes else cropped_imgs, cropped_masks, crop_bboxes


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
        common.img_on_ax(im, ax, title=f'{cat} segmentation ({obj_prop:.{3}f}%)')
        ax.imshow(m, cmap='jet', alpha=0.4)


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
            common.img_on_ax(im_arr, axs[r*ncols + i], title=f' {"Rand bbox" if rand else "Bbox"} thresh {t}')
            axs[r*ncols + i].imshow(m, cmap='jet', alpha=0.4)


