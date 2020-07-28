import os
import json
import random
import argparse
import numpy as np
from shutil import copy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import common


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


def copy_imgs(args, cat, splits, mask_dirs=None):
    print(f"Copying {cat} images ...")
    for ds, files in splits.items():
        for f in files:
            copy(f, os.path.join(common.maybe_create(args.dest, ds, cat), os.path.basename(f)))
            if mask_dirs:
                for mdir in mask_dirs:
                    m = os.path.splitext(os.path.basename(f))[0] + args.mext
                    copy(os.path.join(args.data, mdir, m), os.path.join(common.maybe_create(args.dest, ds, mdir), m))


def split_coco_labels(coco_json, files):
    with open(coco_json, 'r') as f:
        labels = json.lead(f)
        files = [os.path.basename(f) for f in files]
        labels['images'] = [l for l in labels['images'] if os.path.basename(l['file_name']) in files]
        image_ids = [i['id'] for i in labels['images']]
        labels['annotations'] = [l for l in labels['annotations'] if l['image_id'] in image_ids]
        return labels


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    if args.obj_detec:
        if args.coco_json:
            splits = dataset_split(args, '')
            copy_imgs(args, '', splits)
            json_dir, json_file = os.path.dirname(args.coco_json), os.path.basename(args.coco_json)
            for ds, files in splits.items():
                split_labels = split_coco_labels(args.coco_json, files)
                json_path = os.path.join(json_dir, json_file.replace('.json', f'_{ds}.json'))
                json.dump(split_labels, open(json_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
        else:
            mask_dirs = [d for d in dirs if d != args.img_dir and os.path.isdir(os.path.join(args.data, d))]
            assert len(mask_dirs) > 0, "Error, no masks dirs in provided directory, please provide coco json"
            splits = dataset_split(args, args.img_dir)
            copy_imgs(args, args.img_dir, splits, mask_dirs)
    else:
        for d in dirs:
            copy_imgs(args, d, dataset_split(args, d))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset in train/test dataset")
    parser.add_argument('--data', type=str, required=True, help="source dataset root directory")
    parser.add_argument('--dest', type=str, help="directory where the slits should be saved")
    parser.add_argument('--test-size', default=.2, type=float, help="Proportion of test set")
    parser.add_argument('--cross-val', action='store_true', help="apply cross validation")
    parser.add_argument('--val-size', default=.2, type=float, help="Proportion of validation set")
    parser.add_argument('--nfolds', default=5, type=int, help="Number of folds")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    common.add_obj_detec_args(parser)
    parser.add_argument('--coco-json', type=str, help="if given will not look for masks")
    args = parser.parse_args()

    common.check_dir_valid(args.data)

    if args.coco_json:
        common.check_file_valid(args.coco_json)

    if args.dest is None:
        args.dest = common.maybe_create(os.path.dirname(args.data), os.path.basename(args.data) + '_splitted')
    args.dest = args.dest.rstrip('/')

    main(args)
