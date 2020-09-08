import os
import cv2
import json
import random
import numpy as np

from skimage import measure
from shapely.geometry import Polygon
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from tqdm import tqdm

import common

def get_default_dataset(description='coco format_data'):
    return {'info': {'year': 2020, 'description': description},
            'licenses': [{'id': 1, 'name': 'confidential, cannot be shared'}],
            'images': [], 'categories': [], 'annotations': []}


def get_categories(classes, categories):
    return [{'id': i, 'name': classes[i - 1], 'supercategory': classes[i - 1]} for i in sorted(categories)]


def get_img_record(idx, img_path, license_id=1):
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return {
        'id': idx,
        'file_name': os.path.basename(img_path),
        'license': license_id,
        'height': im.shape[-2],
        'width': im.shape[-3]
    }


def convert_obj_mask_to_rle(mask):
    rle = coco_mask.encode(mask)
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


def get_img_annotations(img_id, start_ann_id, targets):
    annos = []
    categories = set()
    ann_id = start_ann_id
    targets["boxes"][:, 2:] -= targets["boxes"][:, :2]
    for k in targets.keys():
        if hasattr(k, 'tolist'):
            targets[k] = targets[k].tolist()
    for i in range(len(targets['labels'])):
        annos.append({
            'image_id': img_id,
            'bbox': [float(b) for b in targets["boxes"][i]],
            'category_id': int(targets['labels'][i]),
            'area': int(targets['areas'][i]),
            'iscrowd': int(targets['iscrowd'][i]),
            'id': ann_id,
            'segmentation': targets['masks'][i]
        })
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
