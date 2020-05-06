import os
import json
import argparse
import itertools

import cv2
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPoint

import common
import PatchExtractor


def main(args):
    with open(args.labels, 'r') as f:
        labels = json.load(f)

    print("CONVERTING LABELS TO NEW PATCH SIZE")
    patcher = PatchExtractor(args.patch_size)
    pms = patcher.imgs_to_patches(args.full_imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts patch coco labels to full img labels")
    parser.add_argument('--full-imgs', type=str, required=True, help="Path of dir containing full imgs")
    parser.add_argument('--labels', type=str, help="JSON file containing patch labels in coco format")
    parser.add_argument('-p', '--patch-size', default=1024, type=int, help="patch size to convert labels to")
    args = parser.parse_args()

    common.check_dir_valid(args.full_imgs)
    common.check_file_valid(args.labels)

    common.time_method(main, args)

