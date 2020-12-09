import os
import sys
import json
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage import measure
from shapely.geometry import Polygon
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common


def get_default_dataset(description='coco format data'):
    return {'info': {'year': 2020, 'description': description},
            'licenses': [{'id': 1, 'name': 'confidential, cannot be shared'}],
            'images': [], 'categories': [], 'annotations': []}


def get_categories(cat_names, cat_ids):
    """Categories ids (cat_ids param) start from 1"""
    return [{'id': i, 'name': cat_names[i - 1], 'supercategory': cat_names[i - 1]} for i in sorted(cat_ids)]


def get_img_record(idx, img_path, im_shape=None, license_id=1):
    w, h = Image.open(img_path).size if im_shape is None else (im_shape[1], im_shape[0])
    return {
        'id': idx,
        'file_name': os.path.basename(img_path),
        'license': license_id,
        'height': h,
        'width': w
    }


def convert_obj_mask_to_rle(mask):
    """Converts input to fortran contiguous array if needed"""
    rle = coco_mask.encode(mask if mask.flags['F_CONTIGUOUS'] else np.asfortranarray(mask))
    rle['counts'] = str(rle['counts'], 'utf-8')
    return rle


def convert_torch_masks_to_rle(masks):
    masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    return [convert_obj_mask_to_rle(mask.numpy()) for mask in masks]


def maybe_simplify_poly(poly):
    simpler_poly = poly.simplify(1.0, preserve_topology=False)
    if type(simpler_poly) is not Polygon:   # Multipolygon case
        return poly
    segmentation = np.array(simpler_poly.exterior.coords).ravel().tolist()
    # CVAT requirements
    return simpler_poly if len(segmentation) % 2 == 0 and 3 <= len(segmentation) // 2 else poly


def convert_obj_mask_to_poly(mask):
    # https://www.immersivelimit.com/create-coco-annotations-from-scratch
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        poly = Polygon(contour)
        poly = maybe_simplify_poly(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        assert len(segmentation) % 2 == 0 and 3 <= len(segmentation) // 2, "Wrong polygon points: %s" % segmentation
        segmentations.append(segmentation)
    return segmentations


def convert_masks_to_poly(masks):
    return [convert_obj_mask_to_poly(mask.numpy()) for mask in masks]


def get_obj_anno(img_id, ann_id, cat_id, bbox, area, seg, is_crowd=0, with_score=False):
    anno = {
        'image_id': img_id,
        'bbox': bbox,
        'category_id': cat_id,
        'area': area,
        'iscrowd': is_crowd,
        'id': ann_id,
        'segmentation': seg
    }
    if with_score:
        anno['score'] = 1.0
    return anno


def get_annos_from_objs_mask(img_id, start_ann_id, obj_cats, obj_cats_masks, with_score=False):
    annos = []
    ann_id = start_ann_id
    for oc, (nb_objs, oc_mask) in zip(obj_cats, obj_cats_masks):
        for obj_id in range(1, nb_objs):    # first object 0 is the background, last obj id is nb_objs-1
            rle = convert_obj_mask_to_rle((oc_mask == obj_id).astype(np.uint8))
            annos.append(get_obj_anno(img_id, ann_id, oc, coco_mask.toBbox(rle), coco_mask.area(rle),
                                      rle, with_score=with_score))
            ann_id += 1
    return ann_id, annos


def get_img_annotations(img_id, start_ann_id, targets):
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
    common.check_dir_valid(img_dir)
    common.check_dir_valid(dest)
    common.check_file_valid(anno_json)

    coco = COCO(anno_json)
    imgs = coco.loadImgs(coco.getImgIds())
    for img in tqdm(imgs):
        im = common.load_rgb_img(os.path.join(img_dir, img['file_name']))
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(im)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))
        coco.showAnns(anns)
        common.plt_save_fig(os.path.join(dest, str(img['id']).zfill(4)+'_'+img['file_name']), dpi=150)


def save_sample_coco_data(coco_json_path, sample_size=100, save_path=None):
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
    path = '/home/shravan/Downloads/temp/'
    d = '00_patched1024'
    visualize_dataset(path+'images/'+d, path+'annotations/14_PPP_study_goldberg_00_1024px.json', path+'results')
