import os
import argparse

import cv2

import common
import concurrency
import multiprocessing as mp
import segmentation.segmentation_utils as seg_utils
from segmentation.mask_utils import crop_img_and_mask_to_objs


SEP = '__CROP__'


def save_crops(args, thresh, img_path, cropped_imgs, cropped_masks):
    file, ext = os.path.splitext(os.path.basename(img_path))
    imdir = os.path.dirname(img_path).replace(args.data, args.dest)
    mdir = imdir.replace(args.img_dir, args.mask_dir)
    suf = f'_{thresh}_{"rand" if args.rand_margins else "notRand"}'
    for i, (im, mask) in enumerate(zip(cropped_imgs, cropped_masks)):
        crop_fname = f'{file}{SEP}{suf}_{common.zero_pad(i, len(cropped_imgs))}{ext}'
        cv2.imwrite(os.path.join(imdir, crop_fname), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(mdir, crop_fname.replace(ext, args.mext)), mask)


def multiprocess_cropping(args, proc_id, images):
    print(f"Process {proc_id}: cropping {len(images)} patches")
    for img_path in images:
        mask_path = seg_utils.get_mask_path(img_path, args.img_dir, args.mask_dir, args.mext)
        im, mask = seg_utils.load_img_and_mask(img_path, mask_path)
        try:
            for thresh in args.threshs:
                crops = crop_img_and_mask_to_objs(im, mask, thresh, args.rand_margins, single=False, bg=args.bg)
                save_crops(args, thresh, img_path, *crops[:2])
        except Exception:
            print(f"Process {proc_id}: error with img {img_path}, skipping.")
    print(f"Process {proc_id}: task completed")


def main(args):
    all_images = common.list_files(os.path.join(args.data, args.img_dir), full_path=True)
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
    seg_utils.segmentation_args(parser, pdef={'--cats': ["other", "pustules", "spots"]})
    parser.add_argument('--threshs', nargs='+', default=[.01], type=float, help="Object proportion thresholds")
    parser.add_argument('--rand-margins', action='store_true', help="Grow crops margins randomly")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    args.data = args.data.rstrip('/')

    common.set_seeds(args.seed)

    args.threshs = sorted(args.threshs)

    if args.dest is None:
        args.dest = common.maybe_create(f'{args.data}_cropped_{"_".join(map(str, args.threshs))}')
        common.maybe_create(args.dest, args.img_dir)
        common.maybe_create(args.dest, args.mask_dir)

    common.time_method(main, args)

