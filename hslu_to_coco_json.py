import os
import cv2
import json
import common
import random
import argparse
import numpy as np
from skimage import measure
import multiprocessing as mp
from shapely.geometry import Polygon
from pycocotools import mask as coco_mask
import concurrency
from ObjDetecPatchSamplerDataset import ObjDetecPatchSamplerDataset
import torch

def maybe_simplify_poly(poly):
    simpler_poly = poly.simplify(1.0, preserve_topology=False)
    if type(simpler_poly) is not Polygon:   # Multipolygon case
        return poly
    segmentation = np.array(simpler_poly.exterior.coords).ravel().tolist()
    # CVAT requirements
    return simpler_poly if len(segmentation) % 2 == 0 and 3 <= len(segmentation) // 2 else poly

def mask_to_polygon(mask):
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

def load_img_from_disk(img_path):
    return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

def extract_annos(proc_id, q_annos, data, dirs):
    for dirname in dirs:
        print(f"Process {proc_id}: extracting {dirname} objects")
        for idx, mask_file in enumerate(sorted(os.listdir(os.path.join(data, dirname)))):
            mask = load_img_from_disk(os.path.join(data, dirname, mask_file))
            mask = ObjDetecPatchSamplerDataset.rm_small_objs_and_sep_instance(mask, 30, check_bbox=True)
            q_annos.put([{mask_file: {dirname: ObjDetecPatchSamplerDataset.extract_mask_objs(mask)}}])

def combine_class_annos(img_objs, mask_dirs):
    targets = ObjDetecPatchSamplerDataset.merge_all_masks_objs([img_objs[mdir] for mdir in mask_dirs])
    if targets:
        return {'labels': targets['classes'], 'area': targets['bbox_areas'], 'boxes': targets['boxes'],
                'masks': targets['obj_masks'], 'iscrowd': targets['iscrowd']}
    else:
        return {}

def to_coco_format(data, img_dir, annos, mext, classes, to_polygon):
    dataset = {'info': {'year': 2020, 'description': os.path.basename(data)},
               'licenses': [{'id': 1, 'name': 'confidential, cannot be shared'}],
               'images': [], 'categories': [], 'annotations': []}
    categories = set()
    ann_id = 0
    for idx, img in enumerate(sorted(os.listdir(os.path.join(data, img_dir)))):
        im = load_img_from_disk(os.path.join(data, img_dir, img))
        dataset['images'].append({
            'id': idx,
            'file_name': img,
            'license': dataset['licenses'][0]['id'],
            'height': im.shape[-2],
            'width': im.shape[-3]
        })
        targets = annos[os.path.splitext(img)[0] + mext]
        if not targets:  # empty dict
            continue
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        masks = torch.as_tensor(targets['masks'], dtype=torch.uint8)
        # make masks Fortran contiguous for coco_mask
        masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        num_objs = len(bboxes)
        for i in range(num_objs):
            try:
                anno = {
                    'image_id': idx,
                    'bbox': bboxes[i],
                    'category_id': labels[i],
                    'area': areas[i],
                    'iscrowd': iscrowd[i],
                    'id': ann_id
                }
                if to_polygon:
                    anno['segmentation'] = mask_to_polygon(masks[i].numpy())
                else:
                    rle = coco_mask.encode(masks[i].numpy())
                    rle['counts'] = str(rle['counts'], 'utf-8')
                    anno['segmentation'] = rle
                dataset['annotations'].append(anno)

                categories.add(labels[i])
                ann_id += 1
            except Exception as err:
                print(f"Image {img} is causing a problem with obj {i} of category {classes[labels[i] - 1]}: {err}")
    dataset['categories'] = [{'id': i, 'name': classes[i - 1], 'supercategory': classes[i - 1]}
                             for i in sorted(categories)]
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Converts dataset to a patch dataset without data augmentation")
    parser.add_argument('--data', type=str, required=True, help="dataset root directory absolute path")
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--to_polygon', action='store_true', help="converts bitmask to polygon")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")
    parser.add_argument('--mdir-prefix', type=str, default='masks_', help="prefix to rm from mask dirs to get mask class")
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    common.set_seeds(args.seed)

    json_path = os.path.join(args.data, f"{os.path.basename(args.data)}_coco_format.json")

    if args.data is None or not os.path.exists(args.data):
        raise Exception("Error, --data invalid")

    mask_dirs = [d for d in sorted(os.listdir(args.data)) if d != args.img_dir and os.path.isdir(os.path.join(args.data, d))]
    classes = [m.replace(args.mdir_prefix, '') for m in mask_dirs]

    workers, batch_size, batched_dirs = concurrency.batch_lst(mask_dirs)
    q_annos = mp.Queue()
    jobs = []
    for i, dirs in zip(range(workers), batched_dirs):
        jobs.append(mp.Process(target=extract_annos, args=(i, q_annos, args.data, dirs)))
        jobs[i].start()
    annos = concurrency.unload_mpqueue(q_annos, jobs)
    print("Processes completed work, merging results ...")
    merged_annos = {k: {dirname: None for dirname in mask_dirs} for anno in annos for k in anno.keys()}
    for anno in annos:
        for mask_file, v in anno.items():
            for dirname, objs in v.items():
                merged_annos[mask_file][dirname] = objs

    merged_annos = {mask_file: combine_class_annos(merged_annos[mask_file], mask_dirs) for mask_file in merged_annos.keys()}

    for j in jobs:
        j.join()
    print("Converting and saving as coco json")
    json.dump(to_coco_format(args.data, args.img_dir, merged_annos, args.mext, classes, args.to_polygon),
              open(json_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
    print("done")


if __name__ == '__main__':
    main()
