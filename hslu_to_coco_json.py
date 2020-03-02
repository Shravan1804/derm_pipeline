import os
import cv2
import json
import random
import numpy as np
from radam import *
import argparse
import multiprocessing as mp

import concurrency

def extract_annos(proc_id, q_annos, data, dirs):
    for dir in dirs:
        for idx, mask_path in enumerate(sorted(os.path.listdir(os.path.join(data, dir)))):
        mask = cv2.imread(os.path.join(data, dir, mask_path), cv2.IMREAD_UNCHANGED)

def create_json_dict(data, img_dir, annos):
    dataset = {'info': {'year': 2020, 'description': os.path.basename(data)},
               'licenses': [{'id': 1, 'name': 'confidential, cannot be shared'}],
               'images': [], 'categories': [], 'annotations': []}
    """categories = set()
    ann_id = 0
    for img_idx in range(len(self)):
        img, targets = self[img_idx]
        image_id = targets["image_id"].item()
        dataset['images'].append({
            'id': image_id,
            'file_name': os.path.basename(self.get_patch_list()[img_idx]['patch_path']),
            'license': dataset['licenses'][0]['id'],
            'height': img.shape[-2],
            'width': img.shape[-3]
        })
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        num_objs = len(bboxes)
        for i in range(num_objs):
            rle = coco_mask.encode(masks[i].numpy())
            rle['counts'] = str(rle['counts'], 'utf-8')
            categories.add(labels[i])
            dataset['annotations'].append({
                'image_id': image_id,
                'bbox': bboxes[i],
                'category_id': labels[i],
                'area': areas[i],
                'iscrowd': iscrowd[i],
                'id': ann_id,
                'segmentation': rle
            })
            ann_id += 1
    classes = [v.replace('masks_', '').upper() for v in self.masks_dirs]
    dataset['categories'] = [{'id': i, 'name': classes[i - 1], 'supercategory': classes[i - 1]}
                             for i in sorted(categories)]"""
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Converts dataset to a patch dataset without data augmentation")
    parser.add_argument('--data', type=str, required=True, help="dataset root directory absolute path")
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    json_path = os.path.join(args.data, f"{os.path.basename(args.data)}_coco_format.json")

    if args.data is None or not os.path.exists(args.data):
        raise Exception("Error, --data invalid")

    mask_dirs = [d for d in sorted(os.listdir(args.data)) if d != args.img_dir]

    workers, batch_size, batched_dirs = concurrency.batch_dirs(mask_dirs)
    q_annos = mp.Queue()
    jobs, annos = [], []
    for i, dirs in zip(range(workers), batched_dirs):
        jobs.append(mp.Process(target=extract_annos, args=(i, q_annos, args.data, dirs)))
        jobs[i].start()
    q_annos.extend(concurrency.unload_mpqueue(q_annos, jobs))
    for j in jobs:
        j.join()
    json.dump(create_json_dict(args.data, args.img_dir, annos), open(json_path, 'w'))
    print("done")


if __name__ == '__main__':
    main()
