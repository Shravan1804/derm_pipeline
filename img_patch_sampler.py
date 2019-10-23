import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import cv2
from scipy import ndimage
from tqdm import tqdm


def prepare_dest_dir(root_input_dir, root_suffix='_patches'):
    if root_input_dir.endswith('/'):
        root_input_dir = args.root_img_dir[:-1]
    # 1 check path
    if not os.path.exists(root_input_dir):
        raise Exception("Error with provided path:", root_input_dir)
    root_dest_dir = os.path.join(os.path.dirname(root_input_dir), os.path.basename(root_input_dir) + root_suffix)
    if not os.path.exists(root_dest_dir):
        os.makedirs(root_dest_dir)
    _, dirs, _ = next(os.walk(root_input_dir))
    for dir in dirs:
        dest = os.path.join(root_dest_dir, dir)
        if not os.path.exists(dest):
            os.makedirs(dest)
    return root_dest_dir


def maybe_resize(img, min_size):
    w, h, c = img.shape
    smallest = min(h, w)
    if smallest < min_size:
        ratio = min_size / smallest
        return cv2.resize(img, (max(min_size, int(w * ratio)), max(min_size, int(h * ratio))))
    else:
        return img


def get_overlap(n, div):
    remainder = n % div
    quotient = max(1, int(n / div))
    overlap = math.ceil((div - remainder) / quotient)
    return 0 if overlap == n else overlap


def save_img_as_grid_of_patches(im_arr, patch_size, leading_zeros, patch_path):
    im_h, im_w, im_c = np.shape(im_arr)
    step_h = patch_size - get_overlap(im_h, patch_size)
    grid_h = np.arange(start=0, stop=im_h - patch_size, step=step_h)
    step_w = patch_size - get_overlap(im_w, patch_size)
    grid_w = np.arange(start=0, stop=im_w - patch_size, step=step_w)
    grid_idx = [(a, b) for a in grid_h for b in grid_w]
    if not grid_idx:
        grid_idx = [(0, 0)]
    for i, idx in enumerate(grid_idx):
        patch = get_img_patch(im_arr, idx[0], idx[1], patch_size)
        cv2.imwrite(patch_path.format(str(i).zfill(leading_zeros)), patch)
    return len(grid_idx)


def get_img_patch(im, id_h, id_w, patch_size):
    return im[id_h:id_h + patch_size, id_w:id_w + patch_size]


def main():
    parser = argparse.ArgumentParser(description="Sample image patches from images present in provided path "
                                                 "subdirectories")
    parser.add_argument('--root_img_dir', type=str, required=True, help="source image root directory")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--patch_size', type=int, default=400, help="size of patches")
    parser.add_argument('--sample_n_patch', type=int, default=-1,
                        help="nb patches to sample in addition to patches of img as grid,"
                             " defaults to im_h * im_w / args.patch_size_squared")
    parser.add_argument('--rotation_padding', type=str, default='constant', help="mode how rotated image is padded")
    args = parser.parse_args()

    root_dest_dir = prepare_dest_dir(args.root_img_dir, root_suffix=
    '_patch-{}_rotation-{}_nsample-{}'.format(args.patch_size, args.rotation_padding,
                                              args.sample_n_patch if args.sample_n_patch != -1
                                              else 'default'))
    np.random.seed(args.seed)
    default_sample_n_patch = args.sample_n_patch == -1
    # 2 create patches
    _, dirs, _ = next(os.walk(args.root_img_dir))
    for dir in dirs:
        dest = os.path.join(root_dest_dir, dir)
        print("Info: saving", dir, "patches in", dest)
        root, _, files = next(os.walk(os.path.join(args.root_img_dir, dir)))
        for file in tqdm(files):
            im = maybe_resize(cv2.imread(os.path.join(root, file)), args.patch_size)
            im_w, im_h, im_c = im.shape
            file, ext = os.path.splitext(file)
            patch_path = os.path.join(dest, file + '_{}' + ext)
            if default_sample_n_patch:
                args.sample_n_patch = int(im_h * im_w / args.patch_size / args.patch_size)
            im_arr = np.asarray(im)
            leading_zeros = len(str(int(im_h * im_w / args.patch_size / args.patch_size) + args.sample_n_patch)) + 1
            first_rand_patch_index = save_img_as_grid_of_patches(im_arr, args.patch_size, leading_zeros, patch_path)

            rotations = np.random.randint(low=1, high=359, size=args.sample_n_patch).tolist()
            for i, rotation in enumerate(rotations):
                im_arr_rotated = ndimage.rotate(im_arr.astype(np.int16), rotation, reshape=True,
                                                mode=args.rotation_padding, cval=-1)
                h, w, c = np.shape(im_arr_rotated)
                patch = [-1]
                loop_count = 0
                while -1 in patch and loop_count < 1000:
                    idx = (np.random.randint(low=0, high=h - args.patch_size, size=1)[0],
                           np.random.randint(low=0, high=w - args.patch_size, size=1)[0])
                    patch = get_img_patch(im_arr_rotated, idx[0], idx[1], args.patch_size)
                    loop_count += 1
                if loop_count >= 1000:
                    continue
                rand_patch_path = patch_path.format(str(first_rand_patch_index + i).zfill(leading_zeros))
                cv2.imwrite(rand_patch_path, patch.astype(np.uint8))


if __name__ == '__main__':
    main()
