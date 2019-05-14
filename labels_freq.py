import os, sys, argparse, cv2

from tqdm import tqdm

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create image histogram")
    parser.add_argument('--label_dir', type=str, default='/home/shravan/basel-server/data/oliver-auto/run13_man_seg/processed_img/_resize1.0', help="dir containing labels")
    arg = parser.parse_args()

    total_counts = {}
    img_list = [os.path.join(arg.label_dir, f) for f in os.listdir(arg.label_dir) if os.path.isfile(os.path.join(arg.label_dir, f)) and f.startswith('label') and f.endswith('.png')]
    for img in tqdm(img_list):
        im = cv2.imread(img)
        unique, counts = np.unique(im, return_counts=True)
        for i, val in enumerate(unique):
            if(val in total_counts):
                total_counts[val] += counts[i]
            else:
                total_counts[val] = counts[i]
    print(total_counts)

if __name__ == '__main__':
    main()

