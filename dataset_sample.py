import os
import random
import argparse
import numpy as np
from shutil import copy
from common import maybe_create, add_obj_detec_args

def sample_imgs(args, directory):
    files = [os.path.join(args.data, directory, f) for f in os.listdir(os.path.join(args.data, directory))]
    n = min(len(files), args.sample)
    return np.random.choice(files, n, replace=False).tolist()

def copy_imgs(args, cat, samples, mask_dirs=None):
    for f in samples:
        copy(f, os.path.join(maybe_create(args.dest, cat), os.path.basename(f)))
        if mask_dirs:
            for mdir in mask_dirs:
                m = os.path.splitext(os.path.basename(f))[0] + args.mext
                copy(os.path.join(args.data, mdir, m), os.path.join(maybe_create(args.dest, mdir), m))


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    if args.obj_detec:
        mask_dirs = [d for d in dirs if d != args.img_dir and os.path.isdir(os.path.join(args.data, d))]
        copy_imgs(args, args.img_dir, sample_imgs(args, args.img_dir), mask_dirs)
    else:
        for d in dirs:
            copy_imgs(args, d, sample_imgs(args, d))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a sample dataset")
    parser.add_argument('--data', type=str, required=True, help="source dataset root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the patches should be saved")
    parser.add_argument('--sample', default=5, type=int, help="number of images to retrieve from each categories")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    add_obj_detec_args(parser)
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    assert os.path.exists(args.data) and os.path.isdir(args.data), f"Provided dataset dir {args.data} invalid."

    if args.dest is None:
        args.dest = maybe_create(os.path.dirname(args.data), os.path.basename(args.data) + '_sample')
    args.dest = args.dest.rstrip('/')

    main(args)
