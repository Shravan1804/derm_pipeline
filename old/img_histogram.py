import os, sys, argparse, cv2

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create image histogram")
    parser.add_argument('--img', type=str, default='/home/shravan/basel-server/data/oliver-auto/run13_man_seg/processed_img/_resize1.0/label00000.png', help="source image")
    arg = parser.parse_args()

    img_list = [arg.img]
    for img in img_list:
        im = cv2.imread(img)
        h = np.zeros((300,256,3))

        bins = np.arange(256).reshape(256,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ]

        for ch, col in enumerate(color):
            hist_item = cv2.calcHist([im],[ch],None,[256],[0,255])
            cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
            hist=np.int32(np.around(hist_item))
            pts = np.column_stack((bins,hist))
            cv2.polylines(h,[pts],False,col)

        h=np.flipud(h)

        cv2.imshow('colorhist',h)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()

