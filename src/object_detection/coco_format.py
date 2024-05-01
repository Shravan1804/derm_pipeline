import os
import sys
import json
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from ..general import common_plot as cplot, common_img as cimg, common
from ..segmentation import mask_utils


def get_default_dataset(description='coco format data'):
    """Prepares default description dict
    :param description: str, custom description
    :return: default description dict
    """
    return {'info': {'year': 2020, 'description': description},
            'licenses': [{'id': 1, 'name': 'confidential, cannot be shared'}],
            'images': [], 'categories': [], 'annotations': []}


def get_categories(cat_names, cat_ids):
    """Prepares list of category dicts
    Categories ids (cat_ids param) start from 1
    :param cat_names: list of categories' names
    :param cat_ids: list of int, indices of categories
    :return: list of category dicts
    """
    return [{'id': i, 'name': cat_names[i - 1], 'supercategory': cat_names[i - 1]} for i in sorted(cat_ids)]


def get_img_record(idx, img_path, im_shape=None, license_id=1):
    """Prepares image dict
    :param idx: int, image id
    :param img_path: str, image path
    :param im_shape: tuple, image shape (h,w)
    :param license_id: int, license id
    :return: image dict
    """
    h, w = cimg.quick_img_size(img_path) if im_shape is None else im_shape
    return {
        'id': idx,
        'file_name': os.path.basename(img_path),
        'license': license_id,
        'height': h,
        'width': w
    }


def convert_obj_mask_to_rle(mask):
    """Converts binary array mask to rle dict and converts counts information to utf-8
    :param mask: array, binary mask
    :return: mask rle dict
    """
    """Converts input to fortran contiguous array if needed"""
    rle = mask_utils.binary_mask_to_rle(mask)
    rle['counts'] = str(rle['counts'], 'utf-8')
    return rle


def convert_obj_mask_to_poly(mask):
    """Converts array binary mask to polygon coordinates
    :param mask: array, binary mask
    :return: list of polygon coordinates
    """
    segmentations = []
    for poly in mask_utils.convert_binary_mask_to_polys(mask):
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        assert len(segmentation) % 2 == 0 and 3 <= len(segmentation) // 2, "Wrong polygon points: %s" % segmentation
        segmentations.append(segmentation)
    return segmentations


def convert_torch_masks_to_rle(masks):
    """Conversion of torch category masks to rle format
    :param masks: tensor, category binary masks
    :return: list of rle dict for each category
    """
    masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    return [convert_obj_mask_to_rle(mask.numpy()) for mask in masks]


def get_obj_anno(img_id, ann_id, cat_id, bbox, area, seg, is_crowd=0, scores=False):
    """Creates object annotation dict
    :param img_id: int, image id
    :param ann_id: int, annotation id
    :param cat_id: int, object category id
    :param bbox: list/array of bounding boxes
    :param area: int, object area in pixels
    :param seg: rle dict or list of polygon coordinates
    :param is_crowd: int, does image shows crowd of object? (1 for True, 0 for False)
    :param scores: bool, should a dummy score be added to annotations? (needed for segmentation preds conversion)
    :return: object annotation dict
    """
    anno = {
        'image_id': int(img_id),
        'bbox': list(bbox),
        'category_id': int(cat_id),
        'area': int(area),
        'iscrowd': int(is_crowd),
        'id': int(ann_id),
        'segmentation': seg
    }
    if scores:
        anno['score'] = 1.0
    return anno


def get_annos_from_objs_mask(img_id, start_ann_id, obj_cats, obj_cats_masks, scores=False, to_poly=False):
    """Extracts all object annotations from array mask
    :param img_id: int, image id
    :param start_ann_id: int, first annotation index (there are multiple images)
    :param obj_cats: list of object categories
    :param obj_cats_masks: list of object masks
    :param scores: bool, should a dummy score be added to annotations? (needed for segmentation preds conversion)
    :param to_poly: bool, converts masks to polygons
    :return: tuple with next object annotation index and list of object annotations
    """
    annos = []
    ann_id = start_ann_id
    for oc, (nb_objs, oc_mask) in zip(obj_cats, obj_cats_masks):
        for obj_id in range(1, nb_objs):    # first object 0 is the background, last obj id is nb_objs-1
            binary_mask = (oc_mask == obj_id).astype(np.uint8)
            try:
                rle = convert_obj_mask_to_rle(binary_mask)
                seg = convert_obj_mask_to_poly(binary_mask) if to_poly else rle
            except Exception as err:
                print(f'Image {img_id} has an issue with obj {obj_id}: {err}')
                print(f'Mask info: {np.unique(binary_mask, return_counts=True)}')
                continue
            annos.append(get_obj_anno(img_id, ann_id, oc, coco_mask.toBbox(rle),
                                      coco_mask.area(rle), seg, scores=scores))
            ann_id += 1
    return ann_id, annos


def get_img_annotations(img_id, start_ann_id, targets):
    """Get object annotations from image
    :param img_id: int, image id
    :param start_ann_id: int, first annotation id
    :param targets: dict with objects information
    :return: tuple with next object annotation index, list of object annotations and object categories
    """
    assert start_ann_id > 0, "ann_id MUST start at 1 since pycocotools.cocoeval uses detId to track matches" \
                             "and checks with > 0"
    annos = []
    categories = set()
    ann_id = start_ann_id
    targets["boxes"][:, 2:] -= targets["boxes"][:, :2]
    for k in targets.keys():
        if hasattr(k, 'tolist'):
            targets[k] = targets[k].tolist()
    for i in range(len(targets['labels'])):
        annos.append(get_obj_anno(img_id, ann_id, int(targets['labels'][i]), [float(b) for b in targets["boxes"][i]],
                                  int(targets['areas'][i]), targets['masks'][i]))
        categories.add(int(targets['labels'][i]))
        ann_id += 1
    return ann_id, annos, categories


def visualize_dataset(img_dir, anno_json, dest):
    """Plot image with objects overlayed and save plots to disk.
    :param img_dir: str, path of directory containing images
    :param anno_json: str, path of coco json file
    :param dest: str, path of destination directory
    """
    common.check_dir_valid(img_dir)
    common.check_dir_valid(dest)
    common.check_file_valid(anno_json)

    coco = COCO(anno_json)
    imgs = coco.loadImgs(coco.getImgIds())
    for img in tqdm(imgs):
        im = cimg.load_img(os.path.join(img_dir, img['file_name']))
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(im)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))
        coco.showAnns(anns)
        cplot.plt_save_fig(os.path.join(dest, str(img['id']).zfill(4)+'_'+img['file_name']), dpi=150)


def save_sample_coco_data(coco_json_path, sample_size=100, save_path=None, seed=42):
    """Create sample of coco dataset
    :param coco_json_path: str, full dataset annotation file path
    :param sample_size: int, number of images in sample dataset
    :param save_path: str, optional path where to save sample json file
    """
    random.seed(seed)
    with open(coco_json_path, 'r') as f:
        coco_json = json.load(f)
        sample = get_default_dataset()
        for k in ['info', 'licenses', 'categories']:
            sample[k] = coco_json[k]
        sample['images'] = random.sample(coco_json['images'], sample_size)
        for img in sample['images']:
            sample['annotations'].extend([a for a in coco_json['annotations'] if a['image_id'] == img['id']])
        save_path = coco_json_path.replace('.json', '_sample.json') if save_path is None else save_path
        json.dump(sample, open(save_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    path = '/home/shravan/deep-learning/data/ppp_grading/full_img_segm_test_set'
    visualize_dataset(path+'/images', path+'/full_img_segm_test_set.json', path+'/results')
