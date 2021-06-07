import os
import json
import random
import argparse
import numpy as np
from shutil import copy
from sklearn.model_selection import train_test_split

import common


def dataset_split(args, directory):
    files = common.list_files(os.path.join(args.data, directory), full_path=True)
    train, test = train_test_split(files, test_size=args.test_size, random_state=args.seed, shuffle=True)
    return {'train': train, 'test': test}


def copy_imgs(args, cat, splits):
    print(f"Copying {cat} images ...")
    for ds, files in splits.items():
        dest_dir = os.path.join(args.dest, ds, cat)
        common.maybe_create(dest_dir)
        for file in files:
            filename, ext = os.path.splitext(os.path.basename(file))
            copy(file, os.path.join(dest_dir, filename + ext))
            for ld in args.label_dirs:
                lf = filename + args.lext
                lab_dest_dir = common.maybe_create(dest_dir.replace(cat, ld))
                copy(os.path.join(args.data, ld, lf), os.path.join(lab_dest_dir, lf))


def split_coco_labels(coco_json, files):
    with open(coco_json, 'r') as f:
        labels = json.load(f)
        files = [os.path.basename(f) for f in files]
        labels['images'] = [l for l in labels['images'] if os.path.basename(l['file_name']) in files]
        image_ids = [i['id'] for i in labels['images']]
        labels['annotations'] = [l for l in labels['annotations'] if l['image_id'] in image_ids]
        return labels


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = [d for d in common.list_dirs(args.data) if d not in args.label_dirs and d not in args.ignore]
    if args.coco_json:
        # with coco format, images are in single dir
        splits = dataset_split(args, dirs[0])
        json_dir, json_file = args.label_dirs[0], os.path.basename(args.coco_json)
        # remove json dir since from args.label_dirs since no images are there to copy
        args.label_dirs = []
        copy_imgs(args, dirs[0], splits)
        for ds, files in splits.items():
            split_labels = split_coco_labels(args.coco_json, files)
            json_path = os.path.join(json_dir, json_file.replace('.json', f'_{ds}.json'))
            json.dump(split_labels, open(json_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
    else:
        for d in dirs:
            copy_imgs(args, d, dataset_split(args, d))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset in train/test dataset")
    parser.add_argument('--data', type=str, required=True, help="source dataset root directory")
    parser.add_argument('--dest', type=str, help="directory where the slits should be saved")
    parser.add_argument('--test-size', default=.2, type=float, help="Proportion of test set")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--label-dirs', nargs='+', default=[], type=str, help="dirs with labels")
    parser.add_argument('--ignore', nargs='+', default=[], type=str, help="ignore dirs")
    parser.add_argument('--lext', type=str, default='.png', help="label file extension")
    parser.add_argument('--coco-json', type=str, help="if given will not look for masks")
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    args.data = args.data.rstrip('/')

    if args.dest is None:
        args.dest = common.maybe_create(f'{args.data}_splitted')
    else:
        common.check_dir_valid(args.dest)
        args.dest = args.dest.rstrip('/')

    if args.coco_json is not None:
        common.check_file_valid(args.coco_json)
        args.label_dirs = [os.path.basename(os.path.dirname(args.coco_json))]

    common.time_method(main, args)
