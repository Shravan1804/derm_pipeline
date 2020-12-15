import os
import sys
import argparse
import multiprocessing as mp
from types import SimpleNamespace

import cv2
import numpy as np

from pycocotools.coco import COCO

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, concurrency
import segmentation.segmentation_utils as segm_utils


def create_segm_masks(args, proc_id, images, coco):
    print(f"Process {proc_id}: creating {len(images)} masks")
    for img in images:
        file, _ = os.path.splitext(os.path.basename(img.file_name))
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        for catId in sorted([c['id'] for c in coco.cats.values()]):
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img.id, catIds=catId)):
                mask[coco.annToMask(ann).astype(np.bool)] = catId
        cv2.imwrite(os.path.join(args.mask_dir, f'{file}{args.mext}', mask))
    print(f"Process {proc_id}: task completed")


def get_images_without_labels(args, img_with_labels):
    has_labels = [os.path.basename(img.file_name) for img in img_with_labels]
    no_labels = []
    for f in common.list_files(os.path.join(os.path.basename(args.json), args.img_dir), full_path=True):
        filename = os.path.basename(f)
        if filename not in has_labels:
            h, w = common.quick_img_size(f)
            no_labels.append(SimpleNamespace(id=-1337, file_name=filename, height=h, width=w))
    return no_labels


def main(args):
    coco = COCO(args.json)
    all_images = [SimpleNamespace(**img) for img in coco.imgs.values()]
    all_images.extend(get_images_without_labels(all_images))
    np.random.shuffle(all_images)   # otherwise last process has all the images without annos
    workers, batch_size, batched_images = concurrency.batch_lst(all_images, workers=args.workers)
    jobs = []
    for proc_id, images in zip(range(workers), batched_images):
        jobs.append(mp.Process(target=create_segm_masks, args=(args, proc_id, images, coco)))
        jobs[proc_id].start()
    for proc_id in jobs:
        proc_id.join()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates masks from coco json file")
    parser.add_argument('--json', type=str, required=True, help="COCO json labels to be converted")
    parser.add_argument('--workers', type=int, default=8, help="Number of parallel workers")
    segm_utils.common_segm_args(parser)
    args = parser.parse_args()

    common.check_file_valid(args.json)

    if os.path.dirname(args.mask_dir) == '':
        args.mask_dir = common.maybe_create(os.path.dirname(args.json), args.mask_dir)
    else:
        args.mask_dir = common.maybe_create(args.mask_dir)

    common.time_method(main, args)

