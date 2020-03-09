import os
import random
import argparse
import numpy as np
from shutil import copy

def dataset_split(files, cross_val, nfolds):
    files = sorted(files)
    #TODO
    return files[:10], files[10:]

def maybe_create(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def copy_imgs(data, dest, splits, mask_dirs=None, mext=None):
    cat = os.path.basename(os.path.dirname(splits['train']))
    print(f"Copying {cat if not mask_dirs else ''} images:")
    for ds, files in splits.items():
        for f in files:
            copy(f, os.path.join(maybe_create(os.path.join(dest, ds, cat), os.path.basename(f))))
            if mask_dirs:
                for mdir in mask_dirs:
                    mname = os.path.splitext(os.path.basename(f)) + mext
                    copy(os.path.join(data, mdir, mname), os.path.join(maybe_create(os.path.join(dest, ds, mdir), mname))

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = sorted(os.listdir(args.data))
    if args.obj_detec:
        mask_dirs = [d for d in dirs if d != args.img_dir and os.path.isdir(os.path.join(args.data, d))]
        splits = dataset_split(os.listdir(os.path.join(args.data, args.img_dir)))
        copy_imgs(args.data, args.dest, splits, mask_dirs, args.mext)
    else:
        for d in dirs:
            copy_imgs(args.data, args.dest, dataset_split(os.listdir(os.path.join(args.data, d))))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset in train/test dataset")
    parser.add_argument('--data', type=str, required=True, help="source dataset root directory absolute path")
    parser.add_argument('--dest', type=str, required=True, help="directory where the patches should be saved")
    parser.add_argument('--test', default=.2, type=float, help="Proportion of test set")
    parser.add_argument('--cross-val', action='store_true', help='apply cross validation')
    parser.add_argument('--nfolds', default=5, type=int, help='Number of folds')
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--obj-detec', action='store_true', help='if not set, will consider dataset to be classif')
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images if --obj-detec")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")
    args = parser.parse_args()

    assert os.path.exists(args.data) and os.path.isdir(args.data), f"Provided dataset dir {args.data} invalid."

    main(args)
