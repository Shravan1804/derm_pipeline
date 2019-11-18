#FULLSCREEN
#important to be the FIRST import
import matplotlib
matplotlib.use('TkAgg')
#else can run first in bash before script
# export MPLBACKEND=TkAgg

import cv2
import os, sys
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ObjDetecPatchSamplerDataset import ObjDetecPatchSamplerDataset

import numpy as np

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

def show_overlayed_img(id, img_path, masks, classes, transform_mask=False, bbox=False):
    img_orig = read_img(img_path)
    img = img_orig.copy()
    img_mask_cleaned = img_orig.copy()
    legend=[]
    for i, mask in enumerate(masks):
        img[mask!=0] = COLORS[i]
        if transform_mask:
            mask_cleaned = ObjDetecPatchSamplerDataset.clean_mask(mask, 30, 10)
            #mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            #mask_cleaned = cv2.dilate(mask,np.ones((3, 3), np.uint8),iterations=1)
            #mask_cleaned = cv2.erode(mask_cleaned, np.ones((2, 2), np.uint8), iterations=2)
            img_mask_cleaned[mask_cleaned!=0] = COLORS[i]
        if bbox:
            draw_bbox(img, mask)
            if transform_mask: draw_bbox(img_mask_cleaned, mask_cleaned)
        legend.append(Line2D([0], [0], marker='o', color=tuple(map(lambda x: x/255, COLORS[i])), label=classes[i], markersize=10))

    fig, ax = plt.subplots()
    plt.title(f'Image {id}: {os.path.basename(img_path)}')
    ax1=fig.add_subplot(1, 2, 1)
    if transform_mask:
        plt.imshow(img)
        ax1.title.set_text('Image with original mask')
    else:
        plt.imshow(img_orig)
        ax1.title.set_text('Original image')
    ax2=fig.add_subplot(1, 2, 2)
    if transform_mask:
        plt.imshow(img_mask_cleaned)
        ax2.title.set_text('Image with transformed mask')
    else:
        plt.imshow(img)
        ax2.title.set_text('Image with mask')
    ax.legend(handles=legend, loc='lower center')
    ax.set_axis_off()
    for axi in [ax1,ax2]:
        axi.set_axis_off()
    plt_set_fullscreen()
    plt.draw()
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)

def draw_bbox(img, mask):
    objs = ObjDetecPatchSamplerDataset.process_mask(mask)
    if objs is None: return
    _, bbox, _, _ = objs
    for b in bbox:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
    return img

def get_masks(img, masks_dirs):
    masks = []
    for mask_dir in masks_dirs:
        mask_fname = os.path.splitext(os.path.basename(img))[0] + MASK_EXT
        mask_path = os.path.join(mask_dir, mask_fname)
        if not os.path.exists(mask_path):
            raise Exception("Error:", mask_path, "doesn't exist.")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        masks.append(mask)
    return masks


def main():
    parser = argparse.ArgumentParser(description="Loop over files in given dir with masks overlayed")
    parser.add_argument('--root', default='/home/shravan/deep-learning/data/PPP_orig_cleaned_sample', type=str, help="source image root directory absolute path")
    parser.add_argument('--img', type=str, help="absolute path, show only specific image")
    parser.add_argument('--clean_mask', action='store_true', help="Should clean mask. App will compare masks instead of with/wihout mask pics")
    parser.add_argument('--bbox', action='store_true', help="Draws bbox")
    parser.add_argument('--from_sampler', action='store_true', help="show images from ObjDetecPatchSamplerDataset")
    parser.add_argument('--img_ids', type=int, nargs='*', help="show only img with specified id")

    args = parser.parse_args()

    if args.img is not None:
        args.root = os.path.dirname(os.path.dirname(args.img))

    if not os.path.exists(args.root):
        raise Exception("No correct image specified but there is an error with provided path:", args.root)

    masks_dirs = [os.path.join(args.root, m) for m in sorted(os.listdir(args.root))
                  if os.path.isdir(os.path.join(args.root, m)) and m.startswith('masks_')]

    classes = [m.replace(args.root, '').replace('/', '').replace('masks_', '') for m in masks_dirs]

    if args.img is not None:
        show_overlayed_img(args.img, get_masks(args.img, masks_dirs), classes, args.clean_mask, args.bbox)
    else:
        if args.from_sampler:
            datasets = pickle.load(open(os.path.join(args.root, os.path.basename(args.root) + '.p'), "rb"))
            img_list = [os.path.basename(p['patch_path']) for patch_maps in datasets for p in patch_maps]
        else:
            img_list = list(sorted(os.listdir(os.path.join(args.root, IMG_DIR))))
        img_index = range(len(img_list))
        if args.img_ids is not None:
            img_list = [img_list[i] for i in args.img_ids]
            img_index = args.img_ids
        for i, img in enumerate(img_list):
            img_path = os.path.join(args.root, IMG_DIR, img)
            try:
                show_overlayed_img(img_index[i], img_path, get_masks(img_path, masks_dirs), classes, args.clean_mask, args.bbox)
            except:
                print(img_path, 'created an error, exiting ...')
                raise


if __name__ == '__main__':
    main()
