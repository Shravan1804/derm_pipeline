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


def create_segm_masks(args, proc_id, images, coco):
    print(f"Process {proc_id}: creating {len(images)} masks")
    for img in images:
        file, _ = os.path.splitext(os.path.basename(img.file_name))
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        for catId in sorted([c['id'] for c in coco.cats.values()]):
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img.id, catIds=catId)):
                mask[coco.annToMask(ann).astype(np.bool)] = catId
        cv2.imwrite(os.path.join(args.mask_dir, f'{file}.png', mask))
    print(f"Process {proc_id}: task completed")


def main(args):
    coco = COCO(args.json)
    all_images = [SimpleNamespace(**img) for img in coco.imgs.values()]
    workers, batch_size, batched_images = concurrency.batch_lst(all_images)
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
    parser.add_argument('--mask-dir', default="masks", type=str, help="mask directory")
    args = parser.parse_args()

    common.check_file_valid(args.json)

    if os.path.dirname(args.mask_dir) == '':
        args.mask_dir = common.maybe_create(os.path.dirname(args.json), args.mask_dir)
    else:
        args.mask_dir = common.maybe_create(args.mask_dir)

    common.time_method(main, args)

