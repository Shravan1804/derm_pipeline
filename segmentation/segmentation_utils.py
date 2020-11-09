import os

import cv2

import common


def get_mask_path(img_path, img_dir, mask_dir, mext):
    return img_path.replace(img_dir, mask_dir).replace(os.path.splitext(img_path)[1], mext)


def load_img_and_mask(img_path, mask_path):
    return common.load_rgb_img(img_path), cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)


def common_segm_args(parser, pdef=dict(), phelp=dict()):
    parser.add_argument('--img-dir', type=str, default=pdef.get('--img-dir', "images"),
                        help=phelp.get('--img-dir', "Images dir"))
    parser.add_argument('--mask-dir', type=str, default=pdef.get('--mask-dir', "masks"),
                        help=phelp.get('--mask-dir', "Masks dir"))
    parser.add_argument('--mext', type=str, default=pdef.get('--mext', ".png"),
                        help=phelp.get('--mext', "Masks file extension"))
    parser.add_argument('--cats', type=str, nargs='+', default=pdef.get('--cats', None),
                        help=phelp.get('--mext', "Segmentation categories"))
    parser.add_argument('--bg', type=int, default=pdef.get('--bg', 0),
                        help=phelp.get('--bg', "Background mask code"))


