import cv2
import os
import argparse

MASK_EXT = '.png'



def main():
    parser = argparse.ArgumentParser(description="Loop over files in given dir with masks overlayed")
    parser.add_argument('--root', type=str, help="source image root directory absolute path")
    parser.add_argument('--img',
                        default='/home/shravan/deep-learning/data/PPP_orig_cleaned_sample/images/run12_00030.jpg' type=str, help="absolute path, show only specific image")
    args = parser.parse_args()

    if not os.path.exists(args.root) and not (args.img is not None and os.path.exists(args.img)):
        raise Exception("No correct image specified but there is an error with provided path:", args.root)

    masks_dirs = [os.path.join(args.root, m) for m in sorted(os.listdir(args.root))
                  if os.path.isdir(os.path.join(args.root, m)) and m.startswith('masks_')]

    if args.img is not None:
        show_overlayed_img(args.img, get_masks(args.img, masks_dirs))
    else:
        img_list = list(sorted(os.listdir(args.root)))
        for img in img_list:
            show_overlayed_img(img, get_masks(img, masks_dirs))


if __name__ == '__main__':
    main()


def show_overlayed_img(img_path, masks):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    for mask in masks:
        img[mask] = (0, 0, 255)
    cv2.namedWindow("img-overlayed", cv2.WINDOW_AUTOSIZE)
    # or cv.namedWindow("window",cv.CV_WINDOW_AUTOSIZE)
    cv2.imshow("img-overlayed", img)


def get_masks(img, masks_dirs):
    masks = []
    for mask_dir in masks_dirs:
        mask_fname = os.path.splitext(os.path.basename(img))[0] + MASK_EXT
        masks.append(cv2.imread(os.path.join(mask_dir, mask_fname), cv2.IMREAD_UNCHANGED))
    return masks
