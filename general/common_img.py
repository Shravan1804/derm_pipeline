#!/usr/bin/env python

"""common_img.py: File regrouping useful methods for images"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

import os
import sys

import cv2
import numpy as np
import PIL.Image as PImage

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common

def quick_img_size(img_path):
    """Returns height, width of img, quicker than cv2 since it does not load the image in memory
    :param img_path: str, image path
    :return: tuple, (height, width)
    """
    width, height = PImage.open(img_path).size
    return height, width


def im_sharpness_score(im):
    """Computes a sharpness score of image
    im is a numpy array of rgb image
    Returns sharpness score (float): the lower the score the blurrier the image,
    the higher the score, the sharper the image

    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    Laplacian is used to measure the 2nd derivative of an image,
    it highlights regions of an image containing rapid intensity changes (edges)

    Assumption is that if an image contains high variance then there is a wide spread of responses,
    both edge-like and non-edge like, representative of a normal, in-focus image.
    But if there is very low variance, then there is a tiny spread of responses,
    indicating there are very little edges in the image.
    As we know, the more an image is blurred, the less edges there are.
    :param im: array, image array
    :return: float, sharpness score
    """
    return cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


def img_bgr_to_rgb(im):
    """Helper for bgr to rgb colorspace conversion
    :param im: array, image
    :return: array, converted image
    """
    if len(im.shape) != 3:
        raise Exception(f"Error cannot convert from bgr to rgb, im shape is {im.shape}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def resize(im, new_h, new_w):
    """Helper to resize function (h, w in correct order, cv2 uses w, h)
    :param im: array, image
    :param new_h: int, new height
    :param new_w: int, new width
    :return: array, resized image
    """
    return cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST_EXACT if len(im.shape) < 3 else cv2.INTER_LINEAR)


def resize_keep_aspect_ratio(im, maxh, maxw=None):
    """Helper to resize function while keeping aspect ratio
    :param im: array, image
    :param maxh: int, max possible height
    :param maxw: int, optional, max possible width, if None same as maxh
    :return: array, resized image
    """
    if maxw is None: maxw = maxh
    h, w = im.shape[:2]
    f = min(maxh/h, maxw/w)
    return resize(im, int(h * f), int(w * f))


def load_img(path, resize=None):
    """Load image with cv2, converts to rgb if needed, eventually resize
    :param path: str, image path
    :param resize: tuple, (new_width, new_heigh)
    :return: array, image
    """
    if common.is_path(path) and type(path) is not str: path = str(path)
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = img_bgr_to_rgb(im) if len(im.shape) > 2 else im
    if resize is not None: im = cv2.resize(im, resize)
    return im


def h_w_ratio(impath, im=None):
    """Calculates image heigh/width ratio
    :param impath: str, image path
    :param im: array, image optional, if None will look for img shape from impath
    :return: float, heigh/width ratio
    """
    h, w = quick_img_size(impath) if im is None else im.shape[:2]
    return h/w


def save_img(im, path):
    """Saves images to specified path
    If im has more than two channels, assumes it is an rgb image thus will convert it to bgr before using imwrite
    :param im: array, image
    :param path: str, where to save image
    """
    cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if len(im.shape) > 2 else im)


class DrawHelper:
    """Class used to draw patches delimitations on images"""
    def __init__(self, thickness=1, style='dotted', gap=10):
        self.thickness = thickness
        self.style = style
        self.gap = gap

    def drawline(self, im, pt1, pt2, color):
        """Draws line according to args
        :param im: array, image
        :param pt1: tuple, x, y
        :param pt2: tuple, x, y
        :param color: tuple, cv2 color
        """
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5  # pythagoras hypotenuse
        pts = []
        for i in np.arange(0, dist, self.gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)

        if self.style == 'dotted':
            for p in pts:
                cv2.circle(im, p, self.thickness, color, -1)
        else:
            e = pts[0]
            for i, p in enumerate(pts):
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(im, s, e, color, self.thickness)

    def drawpoly(self, im, pts, color):
        """Draws polygon on image
        :param im: array, image
        :param pts: list of tuples, each tuple x, y
        :param color: tuple, cv2 color
        """
        e = pts[0]
        pts.append(pts.pop(0))
        for p in pts:
            s = e
            e = p
            self.drawline(im, s, e, color)

    def drawrect(self, im, pt1, pt2, color):
        """Draw rectangle on image
        :param im: array, image
        :param pt1: tuple, x, y
        :param pt2: tuple, x, y
        :param color: tuple, cv2 color
        """
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        self.drawpoly(im, pts, color)

    def draw_patches(self, img_arr, pm, patch_size):
        """Draw patches on image
        :param img_arr: array, image
        :param pm: dict, patch map with height and width
        :param patch_size: int, side-size of patch
        :return: array, modified image
        """
        for p in pm:
            s = (p['w'], p['h'])
            e = (p['w'] + patch_size, p['h'] + patch_size)
            self.drawrect(img_arr, s, e, (255, 255, 255))
        return img_arr

