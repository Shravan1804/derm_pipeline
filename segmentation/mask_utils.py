#!/usr/bin/env python

"""mask_utils.py: Helpers function to work with masks where categories are represented by unique numbers"""

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
import math

import cv2
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
from pycocotools import mask as pycoco_mask

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, common_plot as cplot


def binary_mask_to_rle(bmask):
    """RLE encoding of binary mask
    :param bmask: array, binary mask
    :return: dict, rle encoded mask
    """
    bmask = bmask.astype(np.uint8)
    return pycoco_mask.encode(bmask if bmask.flags['F_CONTIGUOUS'] else np.asfortranarray(bmask))


def non_binary_mask_to_rles(mask):
    """Category-wise RLE encoding of non binary mask
    :param mask: array, non binary mask
    :return: list of tuples (cat id, rle of cat binary mask)
    """
    return [(k, binary_mask_to_rle(mask == k)) for k in np.unique(mask)]


def rles_to_non_binary_mask(rles):
    """Decodes category-wise RLE encoded to non binary mask
    :param rles: list of tuples (cat id, rle of cat binary mask)
    :return: array, non binary mask
    """
    return sum([k*pycoco_mask.decode(rle) for k, rle in rles])


def maybe_simplify_poly(poly):
    """Generated poly can be overly compley, tries to simplify it
    :param poly: Polygon object
    :return simplified polygon
    """
    simpler_poly = poly.simplify(1.0, preserve_topology=False)
    if type(simpler_poly) is not Polygon:   # Multipolygon case
        return poly
    segmentation = np.array(simpler_poly.exterior.coords).ravel().tolist()
    # CVAT requirements
    return simpler_poly if len(segmentation) % 2 == 0 and 3 <= len(segmentation) // 2 else poly


def convert_binary_mask_to_polys(mask):
    """Converts binary mask in polygons
    # https://www.immersivelimit.com/create-coco-annotations-from-scratch
    :param mask: array, binary mask
    :return: list of polygons
    """
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')
    polys = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        poly = Polygon(contour)
        poly = maybe_simplify_poly(poly)
        polys.append(poly)
    return polys


def merge_obj_masks(obj_masks, obj_labels):
    """Merge individual object masks into single mask
    :param obj_masks: list of single object masks
    :param obj_labels: list of int labels
    :return: array of merged mask
    """
    mres = np.zeros_like(obj_masks[0])
    for ia_mask, label in zip(obj_masks, obj_labels): mres[ia_mask > 0] = label
    return mres


def separate_objs_in_mask(mask, bg=0):
    """Split mask array in individual object masks
    :param mask: array, mask
    :param bg: int, background code
    :return: tuple of labels and object masks
    """
    obj_cats = np.array([t for t in np.unique(mask) if t != bg])
    if obj_cats.size == 1 and obj_cats[0] == bg: return None
    cat_masks = (mask == obj_cats[:, None, None]).astype(np.uint8)
    return obj_cats, tuple(cv2.connectedComponents(cmsk) for cmsk in cat_masks)


def rm_small_objs_from_non_bin_mask(non_binary_mask, min_size, cats_idxs=None, bg=0):
    """Remove objects smaller than threshold from non binary mask
    :param non_binary_mask: array, mask
    :param min_size: int, object size threshold
    :param cats_idxs: list, categories indices for which to remove small objects
    :param bg: int, background code
    :return: array, mask without small objects
    """
    res = np.ones(non_binary_mask.shape, dtype=np.uint8)*bg
    if cats_idxs is None: cats_idxs = np.unique(non_binary_mask)
    for c in cats_idxs:
        if c == bg: continue  # background
        binary_mask = rm_small_objs_from_bin_mask(non_binary_mask == c, min_size)
        res[binary_mask > 0] = c
    return res


def rm_small_objs_from_bin_mask(binary_mask, min_size):
    """Remove objects smaller than threshold from binary mask
    :param non_binary_mask: array, mask
    :param min_size: int, object size threshold
    :return: array, binary mask without small objects
    """
    nb_obj, obj_labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
    if nb_obj < 2: return binary_mask  # only background
    obj_ids, inverse, sizes = np.unique(obj_labels, return_inverse=True, return_counts=True)
    for i, size in enumerate(sizes):
        if size < min_size: obj_ids[i] = 0  # set this component to background

    mask_cleaned = np.reshape(obj_ids[inverse], binary_mask.shape)
    mask_cleaned[mask_cleaned > 0] = 1
    return mask_cleaned.astype(np.uint8)


def nb_obj_in_binary_mask(binary_mask):
    """Counts objects in binary mask
    # -1 for background. Even if the mask contains only 1s, there will be 2 objects
    # even if there are separated 0s zones, they will count only as 1 background object
    # only the separated 1s zones are counted as separated objects
    :param binary_mask:
    :return: int, number of objects
    """
    return cv2.connectedComponents(binary_mask.astype(np.uint8))[0] - 1


def nb_objs(non_binary_mask, cls_id, bg=0):
    """Counts objects in non binary mask. Can count category specific objects or all objects.
    :param non_binary_mask: array, mask
    :param cls_id: int, category index, -1 for any categories
    :param bg: int, background code
    :return: int, number of objects
    """
    bm = non_binary_mask != bg if cls_id == -1 else non_binary_mask == cls_id
    return nb_obj_in_binary_mask(bm.astype(np.uint8))


def area_objs(nbm, cls_id, bg=0):
    """Evaluates number of object pixels. Category specific objects or all objects.
    :param nbm: array, non binary mask
    :param cls_id: int, category index, -1 for any categories
    :param bg: int, background code
    :return: int, area of objects
    """
    u, c = np.unique(nbm, return_counts=True)
    area = c[u != bg].sum(keepdims=True) if cls_id == -1 else c[u == cls_id]
    return area[0] if area.size > 0 else 0


def get_obj_proportion(mask, bg=0):
    """Computes proportion of objects over background
    :param mask: array, non binary mask
    :param bg: int, background code
    :return: tuple with proportion of objects, unique category codes and unique category counts
    """
    u, uc = np.unique(mask, return_counts=True)
    return uc[u != bg].sum() / uc.sum(), u, uc


def load_mask_array(mask_path):
    """Loads mask array
    :param mask_path: str, mask path
    :return: array, mask
    """
    return cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)


def resize_mask(mask, new_size_h_w):
    """Resize mask with nearest neighbor interpolation
    :param mask: array, mask
    :param new_size_h_w: tuple of ints, (h,w)
    :return: array of resized mask
    """
    h, w = new_size_h_w
    # cv2 resize new dim takes first w then h!
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)


def crop_im(im, bbox):
    """Crop image to bounding box
    :param im: array, image
    :param bbox: tuple of 4 ints, wmin, hmin, wmax, hmax
    :return: array of cropped image
    """
    wmin, hmin, wmax, hmax = bbox
    return im[hmin:hmax+1, wmin:wmax+1]


def get_bbox(cond):
    """Get bounding box englobing boolean mask truth values
    :param cond: array, boolean mask satisfying condition
    :return: array of 4 elements, wmin, hmin, wmax, hmax
    """
    pos = cond.nonzero()
    if pos[0].size == 0:  # no objects
        return None
    hmin, hmax = np.min(pos[0]), np.max(pos[0])
    wmin, wmax = np.min(pos[1]), np.max(pos[1])
    return np.array((wmin, hmin, wmax, hmax))


def bbox_side_size(bbox):
    """Computes width and height of bounding box
    :param bbox: array, wmin, hmin, wmax, hmax
    :return: tuple of width and height
    """
    return np.array((bbox[2] - bbox[0], bbox[3] - bbox[1]))


def bbox_area(bbox):
    """Computes bounding box area
    :param bbox: array, wmin, hmin, wmax, hmax
    :return: int, area of bbox
    """
    w, h = bbox_side_size(bbox)
    return w*h


def bbox_centroid(bbox):
    """Finds centroid of bounding box
    :param bbox: array, wmin, hmin, wmax, hmax
    :return: array with centroid coordinates (x, y)
    """
    wmin, hmin, wmax, hmax = bbox
    w, h = bbox_side_size(bbox)
    cX = wmin + math.floor(w/2)
    cY = hmin + math.floor(h/2)
    return np.array((cX, cY))


def bboxes_have_intersect(bbox1, bbox2):
    """Checks if bounding boxes do intersect
    # https://gamedev.stackexchange.com/questions/586/what-is-the-fastest-way-to-work-out-2d-bounding-box-intersection
    # left/right for w (x axis) wmin/wmax, top/bottom for h (y axis) hmin/hmax
    # return !(r2.left > r1.right || r2.right < r1.left || r2.top > r1.bottom || r2.bottom < r1.top);
    :param bbox1: array, wmin, hmin, wmax, hmax
    :param bbox2: array, wmin, hmin, wmax, hmax
    :return: bool, if bbox have non-empty intersection
    """
    wmin1, hmin1, wmax1, hmax1 = bbox1
    wmin2, hmin2, wmax2, hmax2 = bbox2
    return not (wmin2 > wmax1 or wmax2 < wmin1 or hmin2 > hmax1 or hmax2 < hmin1)


def bboxes_intersection(bbox1, bbox2):
    """Computes intersection of bounding boxes
    :param bbox1: array, wmin, hmin, wmax, hmax
    :param bbox2: array, wmin, hmin, wmax, hmax
    :return: array, bounding box of intersection
    """
    if not bboxes_have_intersect(bbox1, bbox2):
        return None
    wmin1, hmin1, wmax1, hmax1 = bbox1
    wmin2, hmin2, wmax2, hmax2 = bbox2
    return np.array((max(wmin1, wmin2), max(hmin1, hmin2), min(wmax1, wmax2), min(hmax1, hmax2)))


def bboxes_union(bbox1, bbox2):
    """Computes union of bounding boxes
    :param bbox1: array, wmin, hmin, wmax, hmax
    :param bbox2: array, wmin, hmin, wmax, hmax
    :return: array, bounding box of union
    """
    wmin1, hmin1, wmax1, hmax1 = bbox1
    wmin2, hmin2, wmax2, hmax2 = bbox2
    return np.array((min(wmin1, wmin2), min(hmin1, hmin2), max(wmax1, wmax2), max(hmax1, hmax2)))


def bboxes_overlap(bbox1, bbox2):
    """Computes overlap area ratio of bounding boxes
    :param bbox1: array, wmin, hmin, wmax, hmax
    :param bbox2: array, wmin, hmin, wmax, hmax
    :return: float, ratio of intersect over union
    """
    intersect = bboxes_intersection(bbox1, bbox2)
    if intersect is None:
        return 0
    area1, area2, intersect_area = bbox_area(bbox1), bbox_area(bbox2), bbox_area(intersect)
    return intersect_area/(area1 + area2 - intersect_area)


def merge_bboxes_based_on_overlap(bboxes, max_overlap):
    """Merge bounding boxes if their overlap ratio is greater than threshold
    :param bboxes: list of bounding boxes
    :param max_overlap: float, overlap ratio threshold
    :return: list of merged bounding boxes
    """
    merged = common.merge(lst=bboxes,
                          cond_fn=lambda a, b: bboxes_overlap(a, b) >= max_overlap,
                          merge_fn=lambda a, b: bboxes_union(a, b))
    return merged


def merge_bboxes_based_on_centroids_dist(centroids, bboxes, min_dist):
    """Merge bounding boxes if their respective centroids distance is below a threshold
    :param centroids: list of bounding boxes centroids
    :param bboxes: list of bounding boxes
    :param min_dist: int, minimum distance ratio threshold
    :return: list of merged bounding boxes
    """
    merged = common.merge(lst=list(zip(centroids, bboxes)),
                          cond_fn=lambda a, b: np.square(a[0] - b[0]).sum() <= min_dist ** 2,
                          merge_fn=lambda a, b: ((a[0] + b[0]) // 2, bboxes_union(a[1], b[1])))
    return merged


def ensure_bbox_side_min_size(smin, smax, side_range, min_size):
    """Grows bounding box side within range until minimum size is reached (or max range reached)
    :param smin: int, side min coordinate
    :param smax: int, side max coordinate
    :param side_range: tuple max side size
    :param min_size: int, minimum size
    :return: array, new side coordinates (min, max)
    """
    if smax - smin < min_size:
        diff = min_size - (smax - smin)
        bbox = (smin, 0, smax, 0)
        bbox = grow_bbox(bbox, side_range, (0, 0), diff, rand=False)
        return np.array((bbox[0], bbox[2]))
    else:
        return np.array((smin, smax))


def ensure_bbox_min_size(im_shape, bbox, min_size):
    """Grows bounding box to minimum side size within image shape
    :param im_shape: tuple, image shape (h,w)
    :param bbox: array, wmin, hmin, wmax, hmax
    :param min_size: int, minimum side size
    :return: array, resized bbox
    """
    wmin, hmin, wmax, hmax = bbox
    wmin, wmax = ensure_bbox_side_min_size(wmin, wmax, (0, im_shape[1]), min_size)
    hmin, hmax = ensure_bbox_side_min_size(hmin, hmax, (0, im_shape[0]), min_size)
    return np.array((wmin, hmin, wmax, hmax))


def grow_bbox(start_bbox, wrange, hrange, total_growth, rd=2, rand=True):
    """Grow bounding box within range on either sides up to a maximum threshold
    rd is the coords reducing/growing split index, first group is reducing, second is growing
    :param start_bbox: array, wmin, hmin, wmax, hmax
    :param wrange: tuple, (min max) x coordinates
    :param hrange: tuple, (min max) y coordinates
    :param total_growth: int, max side size to be gained by growth
    :param rd:int, coords reducing/growing split index, first group is reducing, second is growing
    :param rand: bool, uniform or random growth
    :return: array, bbox after growth
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
    """First dilates mask to remove noise then extracts objects centroids with bounding boxes
    :param mask: array, mask
    :param kern: array, kernel for dilate operations
    :param dilate_it: int, number of dilate iterations
    :param bg: int, code for background
    :return: tuple with list of centroids (cX, cY) and list of bounding boxes
    """
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
    """Increases the bbox size if obj proportion above thresh (adds more background, thus reduces obj proportion)
    :param mask: array, mask
    :param bbox: array, bounding box
    :param thresh: float, object proportion threshold
    :param rand: bool, grow bbox randomly
    :param bg: int, background code
    :return: array, bbox after growth
    """
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
    """Extract all bboxes from image and mask
    :param im: array, image
    :param mask: array, mask
    :param bboxes: list of bboxes
    :return: tuple of cropped images and cropped masks
    """
    cropped_imgs, cropped_masks = [], []
    for bbox in bboxes:
        cropped_imgs.append(crop_im(im, bbox))
        cropped_masks.append(crop_im(mask, bbox))
    return cropped_imgs, cropped_masks


def crop_img_and_mask_to_objs(im, mask, thresh=.01, rand=True, single=True, only_bboxes=False, bg=0,
                              min_obj_area=36, min_crop_side_size=128, max_crop_overlap=.6):
    """Crops image and mask around objects such that object proportion fits threshold
    Parameters min_obj_area, min_crop_side_size,max_crop_overlap may result in object proportion no fitting threshold
    :param im: array, image
    :param mask: array, mask
    :param thresh: float, object proportion threshold
    :param rand: bool, if True random object bbox else square object centered bbox
    :param single: bool, return only one mask per object
    :param only_bboxes: bool, do not return image crops, only bboxes
    :param bg: int, background code
    :param min_obj_area: int, minimum object size
    :param min_crop_side_size: int, minimum crop side size
    :param max_crop_overlap: int, maximum crops overlap
    :return: either list of bboxes or tuple with image crops, mask crops and bboxes
    """
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


def apply_color_map_to_mask(mask, cmap=cv2.COLORMAP_JET, normalize=True, normin=None, normax=None):
    """RGB version of mask following specified color map. Some mask may not have all classes p present in which case,
    use normalize, normin and normax args to account for all classes. Otherwise colors will be different between masks
    :param mask: array, mask
    :param cmap: cv2 color map
    :param normalize: bool, normalize mask between normin and normax
    :param normin: int, min category id
    :param normax:int, max category id
    :return: array, colored mask
    """
    assert len(mask.shape) == 2, "Image should be grayscale"
    if normalize:
        mask = mask.astype(np.float)
        mask -= mask.min() if normin is None else normin
        mask /= mask.max() if normax is None else normax
        mask *= 255
    mask = mask.astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(mask, cmap), cv2.COLOR_BGR2RGB)


def blend_im_mask(im, mask, alpha=.4):
    """Blend image and mask together
    :param im: array, image
    :param mask: array, mask
    :param alpha: float, blend weight, alpha applied to mask, 1-alpha to image
    :return: array, blended image
    """
    return cv2.addWeighted(im, 1-alpha, mask, alpha, 0)


def compute_dice(cls, pred, targ):
    """Calculate dice score for specified class for the given masks
    :param cls: int, category id
    :param pred: tensor, mask
    :param targ: tensor, mask
    :return: float, dice score
    """
    pred = (pred == cls).type(pred.dtype)
    targ = (targ == cls).type(targ.dtype)
    inter = (pred * targ).float().sum().item()
    union = (pred + targ).float().sum().item()
    return 2. * inter / union if union > 0 else None


def im_mask_on_ax(ax, im, mask, title=None):
    """Plot image with mask blended
    :param ax: axis
    :param im: array, image
    :param mask: array, mask
    :param title: str, plot title
    """
    cplot.img_on_ax(im, ax, title=title)
    ax.imshow(mask, cmap='jet', alpha=0.4)


def show_im_with_masks(im, mask, cats):
    """Plot image along with category specific blended masks
    :param im: array, image
    :param mask: array, mask
    :param cats: list, categories
    """
    _, axs = cplot.prepare_img_axs(im.shape[0]/im.shape[1], 1, len(cats) + 1)
    cplot.img_on_ax(im, axs[0], title='Original image')
    for cls_idx, cat in enumerate(cats):
        m = (mask == cls_idx).astype(np.uint8)
        obj_prop = get_obj_proportion(m)[0]
        cplot.img_on_ax(m, axs[cls_idx + 1], title=f'{cat} mask ({obj_prop:.{3}f}%)')


def show_im_with_crops_bboxes(im, mask, obj_threshs=[.01], ncols=4, bg=0):
    """Plot image with object bounding boxes crops
    :param im: array, image
    :param mask: array, mask
    :param obj_threshs: float, object proportion thresholds that bounding boxes should satisfy
    :param ncols: int, number of columns to show
    :param bg: int, background code
    """
    nrows = math.ceil(len(obj_threshs)/ncols) * 2
    _, axs = cplot.prepare_img_axs(im.shape[0]/im.shape[1], nrows, ncols)
    for r, rand in enumerate([False, True]):
        for i, t in enumerate(obj_threshs):
            im_arr = im.copy()
            m = mask.copy()
            crop_bboxes = crop_img_and_mask_to_objs(im_arr, m, t, rand, single=False, only_bboxes=True, bg=bg)
            for (wmin, hmin, wmax, hmax) in crop_bboxes:
                cv2.rectangle(im_arr, (wmin, hmin), (wmax, hmax), 255, 5)
            im_mask_on_ax(axs[r*ncols + i], im_arr, m, title=f' {"Rand bbox" if rand else "Bbox"} thresh {t}')

