import os
import argparse
import numpy as np
from shutil import copy

from tqdm import tqdm

import common


def main(args):
    np.random.seed(args.seed)
    common.reproduce_dir_structure(args.data, args.dest)
    dirs = [args.data] + common.list_dirs(args.data, full_path=True, recursion=True)
    for d in tqdm(dirs):
        fd = common.list_files(d, full_path=True)
        if fd:
            fs = np.random.choice(fd, min(len(fd), args.sample), replace=False)
            for f in fs:
                copy(f, f.replace(args.data, args.dest))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a sample dataset")
    parser.add_argument('--data', type=str, required=True, help="source dataset root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the patches should be saved")
    parser.add_argument('--sample', default=5, type=int, help="number of images to retrieve from each categories")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)

    if args.dest is None:
        args.dest = common.maybe_create(os.path.dirname(args.data), os.path.basename(args.data)
                                        + f'_sample{args.sample}')
    else:
        args.dest = args.dest.rstrip('/')
        common.check_dir_valid(args.dest)

    common.time_method(main, args)
