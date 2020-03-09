import os
import random
import argparse
import numpy as np
from shutil import copy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def dataset_split(args, directory):
    files = [os.path.join(args.data, directory, f) for f in os.listdir(os.path.join(args.data, directory))]
    train, test = train_test_split(files, test_size=args.test_size, random_state=args.seed, shuffle=True)
    if not args.cross_val:
        return {'train': train, 'test': test}
    else:
        assert len(train) > args.nfolds, "Too many folds."
        splits = {'test': test}
        kf = KFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)
        data = np.array(train)
        for i, idx in enumerate(kf.split(data)):
            splits[f'train{i}'] = data[idx[0]].tolist()
            splits[f'val{i}'] = data[idx[1]].tolist()
        return splits


def maybe_create(*d):
    path = os.path.join(*d)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def copy_imgs(data, dest, splits, mask_dirs=None, mext=None):
    cat = os.path.basename(os.path.dirname(splits['test'][0]))
    print(f"Copying {cat} images ...")
    for ds, files in splits.items():
        for f in files:
            copy(f, os.path.join(maybe_create(dest, ds, cat), os.path.basename(f)))
            if mask_dirs:
                for mdir in mask_dirs:
                    m = os.path.splitext(os.path.basename(f))[0] + mext
                    copy(os.path.join(data, mdir, m), os.path.join(maybe_create(dest, ds, mdir), m))


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    if args.obj_detec:
        mask_dirs = [d for d in dirs if d != args.img_dir and os.path.isdir(os.path.join(args.data, d))]
        splits = dataset_split(args, args.img_dir)
        copy_imgs(args.data, args.dest, splits, mask_dirs, args.mext)
    else:
        for d in dirs:
            copy_imgs(args.data, args.dest, dataset_split(args, d))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset in train/test dataset")
    parser.add_argument('--data', type=str, required=True, help="source dataset root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the patches should be saved")
    parser.add_argument('--test-size', default=.2, type=float, help="Proportion of test set")
    parser.add_argument('--cross-val', action='store_true', help='apply cross validation')
    parser.add_argument('--val-size', default=.2, type=float, help="Proportion of test set")
    parser.add_argument('--nfolds', default=5, type=int, help='Number of folds')
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--obj-detec', action='store_true', help='if not set, will consider dataset to be classif')
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images if --obj-detec")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    assert os.path.exists(args.data) and os.path.isdir(args.data), f"Provided dataset dir {args.data} invalid."

    if args.dest is None:
        args.dest = maybe_create(os.path.dirname(args.data), os.path.basename(args.data) + '_splitted')
    args.dest = args.dest.rstrip('/')

    main(args)
