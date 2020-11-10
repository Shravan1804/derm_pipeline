import math
import argparse

from tqdm import tqdm
from PIL import Image

import common

def main(args):
    files = common.list_files_in_dirs(args.data, full_path=True) if args.classif else common.list_files(args.data, full_path=True)
    resolutions = {}
    for img_path in tqdm(files):
        try:
            size = tuple([math.ceil(n / args.bins) * args.bins for n in sorted(Image.open(img_path).size)])
            if size in resolutions:
                resolutions[size] += 1
            else:
                resolutions[size] = 1
        except Exception as err:
            print(img_path, "caused an error:", err, "skipping ...")
    print(f"There are {len(resolutions.keys())} different resolutions for {len(files)} images (bins={args.bins}):")
    for s in sorted(resolutions.keys()):
        print(f"{s[0]}x{s[1]} images: {resolutions[s]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a sample dataset")
    parser.add_argument('--data', type=str, required=True, help="source dataset root directory absolute path")
    parser.add_argument('--bins', type=int, default=1, help="group resolutions in bins")
    common.add_classif_args(parser)
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)

    common.time_method(main, args)
