#!/usr/bin/env python

"""crop_to_thresh.py: Creates cropped around objects version of dataset. The idea is to rebalance object proportion. """

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
import argparse
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, common_img as cimg, concurrency
import segmentation.segmentation_utils as seg_utils
from segmentation.mask_utils import crop_img_and_mask_to_objs, crop_im


SEP = '__CROP__'    # used to separate full image filename from crop info


def save_crops(args, thresh, img_path, im, mask, crop_bboxes):
    """Used to save images crops to disk
    :param args: command line args
    :param thresh: float, object proportion threshold
    :param img_path: str, image path
    :param im: array, image
    :param mask: array, mask
    :param crop_bboxes: list of bounding boxes around objects
    """
    file, ext = os.path.splitext(os.path.basename(img_path))
    imdir = os.path.dirname(img_path).replace(args.data, args.dest)
    mdir = imdir.replace(args.img_dir, args.mask_dir)
    suf = f'_{thresh}_{"rand" if args.rand_margins else "notRand"}'
    for i, bbox in enumerate(crop_bboxes):
        cropped_imgs, cropped_masks = crop_im(im, bbox), crop_im(mask, bbox)
        crop_id = common.zero_pad(i, len(cropped_imgs))
        crop_fname = f'{file}{SEP}{suf}_{"_".join([str(s) for s in bbox])}__{crop_id}{ext}'
        cimg.save_img(cropped_imgs, os.path.join(imdir, crop_fname))
        cimg.save_img(cropped_masks, os.path.join(mdir, crop_fname.replace(ext, args.mext)))


def multiprocess_cropping(args, proc_id, images):
    """Method called by process to perform image cropping around objects
    :param args: command line arguments
    :param proc_id: int, process id
    :param images: list of image paths
    """
    print(f"Process {proc_id}: cropping {len(images)} patches")
    for img_path in images:
        mask_path = seg_utils.get_mask_path(img_path, args.img_dir, args.mask_dir, args.mext)
        im, mask = seg_utils.load_img_and_mask(img_path, mask_path)
        if args.cats_ids is not None: mask[~np.isin(mask, args.cats_ids)] = args.bg
        try:
            for thresh in args.threshs:
                _, _, bboxes = crop_img_and_mask_to_objs(im, mask, thresh, args.rand_margins, single=False, bg=args.bg)
                save_crops(args, thresh, img_path, im, mask, bboxes)
        except Exception:
            print(f"Process {proc_id}: error with img {img_path}, skipping.")
    print(f"Process {proc_id}: task completed")


def main(args, all_dirs):
    """Runs multi process image cropping to objects
    :param args: command line arguments
    :param all_dirs: list of directory paths where images should be cropped around objects
    """
    all_images = [f for d in all_dirs for f in common.list_files(os.path.join(d, args.img_dir), full_path=True)]
    workers, batch_size, batched_images = concurrency.batch_lst(all_images)
    jobs = []
    for proc_id, images in zip(range(workers), batched_images):
        jobs.append(mp.Process(target=multiprocess_cropping, args=(args, proc_id, images)))
        jobs[proc_id].start()
    for proc_id in jobs:
        proc_id.join()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates crops satisfying objects proportion threshold")
    parser.add_argument('--data', type=str, required=True, help="source data root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the crops should be saved")
    parser.add_argument('--cats-ids', type=int, nargs='+', help="Obj categories to crop on")
    seg_utils.common_segm_args(parser)
    parser.add_argument('--threshs', nargs='+', default=[.01], type=float, help="Object proportion thresholds")
    parser.add_argument('--rand-margins', action='store_true', help="Grow crops margins randomly")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--splitted', action='store_true', help="args.data contains multiple datasets")
    parser.add_argument('--ignore', nargs='+', type=str, help="dirs to be ignored in cropping (e.g. weak labels)")
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    args.data = args.data.rstrip('/')

    common.set_seeds(args.seed)

    args.threshs = sorted(args.threshs)

    all_dirs = common.list_dirs(args.data, full_path=True) if args.splitted else [args.data]
    if args.ignore is not None: all_dirs = [d for d in all_dirs if os.path.basename(d) not in args.ignore]
    if args.dest is None:
        args.dest = common.maybe_create(f'{args.data}_cropped_{"_".join(map(str, args.threshs))}')
    for d in all_dirs:
        dest_dir = d.replace(args.data, args.dest)
        common.maybe_create(dest_dir, args.img_dir)
        common.maybe_create(dest_dir, args.mask_dir)

    common.time_method(main, args, all_dirs)

