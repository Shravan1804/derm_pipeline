import os
import cv2
from pycocotools import mask as coco_mask

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
    """Assumes mask is fortran contiguous numpy array
    with tensor: masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    """
    rle = coco_mask.encode(mask)
    rle['counts'] = str(rle['counts'], 'utf-8')
    return rle


def convert_masks_to_rle(masks):
    masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    return [convert_obj_mask_to_rle(mask.numpy()) for mask in masks]


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
            'bbox': targets["boxes"][i],
            'category_id': targets['labels'][i],
            'area': targets['areas'][i],
            'iscrowd': targets['iscrowd'][i],
            'id': ann_id,
            'segmentation': targets['masks'][i]
        })
        categories.add(targets['labels'][i])
        ann_id += 1
    return ann_id, annos, categories

