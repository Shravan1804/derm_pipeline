#FULLSCREEN
#important to be the FIRST import
import matplotlib
matplotlib.use('TkAgg')
#else can run first in bash before script
# export MPLBACKEND=TkAgg

import cv2
import os, sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MASK_EXT = '.png'
COLORS = [(238, 130, 238),(139,69,19)]
IMG_DIR = 'images'


def plt_set_fullscreen():
    backend = str(plt.get_backend())
    mgr = plt.get_current_fig_manager()
    if backend == 'TkAgg':
        if os.name == 'nt':
            mgr.window.state('zoomed')
        else:
            mgr.resize(*mgr.window.maxsize())
    elif backend == 'wxAgg':
        mgr.frame.Maximize(True)
    elif backend == 'Qt4Agg':
        mgr.window.showMaximized()

def read_img(img_path):
    return cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

def show_overlayed_img(img_path, masks, classes):
    img_orig = read_img(img_path)
    img = img_orig.copy()
    legend=[]
    for i, mask in enumerate(masks):
        img[mask!=0] = COLORS[i]
        legend.append(Line2D([0], [0], marker='o', color=tuple(map(lambda x: x/255, COLORS[i])), label=classes[i], markersize=10))

    fig, ax = plt.subplots()
    plt.title(os.path.basename(img_path))
    ax1=fig.add_subplot(1, 2, 1)
    plt.imshow(img_orig)
    ax2=fig.add_subplot(1, 2, 2)
    plt.imshow(img)
    ax.legend(handles=legend, loc='upper right')
    ax.set_axis_off()
    for axi in [ax1,ax2]:
        axi.set_axis_off()
    plt_set_fullscreen()
    plt.draw()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)


def get_masks(img, masks_dirs):
    masks = []
    for mask_dir in masks_dirs:
        mask_fname = os.path.splitext(os.path.basename(img))[0] + MASK_EXT
        mask = cv2.imread(os.path.join(mask_dir, mask_fname), cv2.IMREAD_UNCHANGED)
        masks.append(mask)
    return masks


def main():
    parser = argparse.ArgumentParser(description="Loop over files in given dir with masks overlayed")
    parser.add_argument('--root', default='/home/shravan/deep-learning/data/PPP_orig_cleaned_sample', type=str, help="source image root directory absolute path")
    parser.add_argument('--img', type=str, help="absolute path, show only specific image")
    args = parser.parse_args()

    if not os.path.exists(args.root):
        raise Exception("No correct image specified but there is an error with provided path:", args.root)

    masks_dirs = [os.path.join(args.root, m) for m in sorted(os.listdir(args.root))
                  if os.path.isdir(os.path.join(args.root, m)) and m.startswith('masks_')]

    classes = [m.replace(args.root, '').replace('/', '').replace('masks_', '') for m in masks_dirs]

    if args.img is not None:
        show_overlayed_img(args.img, get_masks(args.img, masks_dirs), classes)
    else:
        img_list = list(sorted(os.listdir(os.path.join(args.root, IMG_DIR))))
        for img in img_list:
            img_path = os.path.join(args.root, IMG_DIR, img)
            show_overlayed_img(img_path, get_masks(img_path, masks_dirs), classes)


if __name__ == '__main__':
    main()
