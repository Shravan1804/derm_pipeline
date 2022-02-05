#!/usr/bin/env python

"""PatchExtractor.py: Patch images in whole directory structure, also contains methods used to independantly patch images"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

import os
import re
import sys
import math
import pickle
import argparse
import multiprocessing as mp

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, common_img as cimg, concurrency

class PatchExtractor:
    """Extracts patches from images either by random sampling or splitting the image as a grid
    Each patch is represented by a dict, for details check create_pm method"""
    # SEP separates original image name before file ext with patch information
    SEP = '__SEP__'
    # RAND separates original image name before file ext with patch information if patch is randomly sampled
    RAND = '__RAND__'

    def __init__(self, patch_sizes, seed=42, dynamic_res=False, root_dir=None):
        """Creation of a PatchExtractor object
        :param patch_sizes: list of int
        :param seed: int, random seed
        :param dynamic_res: bool, should the patch sizes be adjusted to the median image resolution?
        :param root_dir: str, path of dataset, used for dynamic_res
        """
        self.patch_sizes = sorted(patch_sizes) if type(patch_sizes) == list else [patch_sizes]
        np.random.seed(seed)
        self.base_resolution = self.compute_base_resolution(root_dir) if dynamic_res else None

    def compute_base_resolution(self, rdir):
        """Calculates median resolution from dir
        :param rdir: str, dataset directory path
        :return: int, median total pixel size h*w
        """
        common.check_dir_valid(rdir)
        impaths = common.list_images(rdir, full_path=True, recursion=True)
        resolutions = [h * w for ip in impaths for h, w in [cimg.quick_img_size(ip)]]
        return int(np.median(np.array(resolutions)))

    def adjust_ps(self, ps, im_res):
        """Adjust patch size if dynamic resolution was set
        :param ps: int, patch size
        :param im_res: int, image total pixel size h*w
        :return: int, adjusted patch size
        """
        return ps if self.base_resolution is None else int(pow(im_res * ps * ps / self.base_resolution, .5))

    def dir_images_to_patches(self, dirname):
        """Converts images in directory to patches
        :param dirname: str, directory path
        :return: dict, full image basename as keys and list of patches as values
        """
        files = common.list_files(dirname, full_path=True)
        exclude = [p for p in files if not(p.endswith('.png') or p.endswith('.jpg'))]
        if exclude:
            print("Warning: In", dirname, "these files are excluded:", exclude)
        return {os.path.basename(f): self.patch_grid(f) for f in files if f not in exclude}

    def sample_patches(self, img_path, nb, im_arr=None):
        """ Randomly samples patches from image
        :param img_path: str, image path
        :param nb: int, number of patches to sample
        :param im_arr: array, image optional, default will load img_path
        :return: list of patch maps
        """
        assert nb > 1, f"Cannot sample nb={nb} patches from {img_path}."
        im_arr = self.load_image(img_path) if im_arr is None else self.maybe_resize(im_arr)
        im_h, im_w = im_arr.shape[:2]
        patches = []
        for ps in self.patch_sizes:
            ps = self.adjust_ps(ps, im_h * im_w)
            idx_h, idx_w = (np.random.randint(low=0, high=1 + im_h - ps, size=nb),
                            np.random.randint(low=0, high=1 + im_w - ps, size=nb))
            patches.extend([PatchExtractor.create_pm(img_path, ps, 0, 0, h, w, True) for h, w in zip(idx_h, idx_w)])
        return patches

    def patch_grid(self, img_path, im_arr=None):
        """Converts image into a grid of patches
        :param img_path: str, image path
        :param im_arr:  array, image optional, default will load img_path
        :return: list of patch maps
        """
        im_arr = self.load_image(img_path) if im_arr is None else self.maybe_resize(im_arr)
        im_h, im_w = im_arr.shape[:2]
        patches = []
        for ps in self.patch_sizes:
            ps = self.adjust_ps(ps, im_h * im_w)
            oh, ow = PatchExtractor.compute_side_overlap(im_h, ps), PatchExtractor.compute_side_overlap(im_w, ps)
            step_h, step_w = ps - oh, ps - ow
            # Don't forget + 1 in the stop argument otherwise the upper bound won't be included
            grid_h = np.arange(start=0, stop=1 + im_h - ps, step=step_h)
            grid_w = np.arange(start=0, stop=1 + im_w - ps, step=step_w)
            grid_idx = [PatchExtractor.create_pm(img_path, ps, oh, ow, a, b) for a in grid_h for b in grid_w]
            # if lst empty then image fits a single patch
            patches.extend(grid_idx if grid_idx else [PatchExtractor.create_pm(img_path, ps, 0, 0, 0, 0)])
        return patches

    def maybe_resize(self, im_arr):
        """Resize img only if dim smaller than the max patch size otherwise returns img unchanged
        :param im_arr: array, image
        :return: array, maybe resized image
        """
        h, w = im_arr.shape[:2]
        ps = self.adjust_ps(self.patch_sizes[-1], h * w)
        smallest = min(h, w)
        if smallest < ps:
            ratio = ps / smallest
            # resize new dim takes first w then h!
            return cv2.resize(im_arr, (max(ps, int(w * ratio)), max(ps, int(h * ratio))))
        else:
            return im_arr

    def load_image(self, img_path):
        """Loads and resize image if needed
        :param img_path: str, image path
        :return: array, (resized) image array
        :raise AssertionError if file invalid
        """
        if common.is_path(img_path) and type(img_path) is not str: img_path = str(img_path)
        common.check_file_valid(img_path)
        return self.maybe_resize(cimg.load_img(img_path))

    def save_patches(self, source, dest, dirname, grids):
        """Extracts and writes patches to disk
        :param source: str, root path of source images
        :param dest: str, destination root folder path
        :param dirname: str, directory inside root containing images
        :param grids: dict, full image basename as keys and list of patches as values
        """
        dest = common.maybe_create(dest, dirname)
        for img, patches in grids.items():
            im = self.load_image(os.path.join(source, dirname, img))
            for pm in patches:
                cimg.save_img(self.extract_patch(im, pm), os.path.join(dest, pm['patch_path']))

    @staticmethod
    def extract_patch(im, pm):
        """Extracts patch from image using patch map info
        :param im: array, image
        :param pm: dict, patch map
        :return: array, image patch
        """
        ps, id_h, id_w = pm['ps'], pm['h'], pm['w']
        return im[id_h:id_h + ps, id_w:id_w + ps]

    @staticmethod
    def are_neighbors(p1, p2, d=1):
        """Determines if two patches are neighbors with certain patch-wise distance
        :param p1: str, patch filename
        :param p2: str, patch filename
        :param d: int, allowed distance
        :return: bool
        """
        pm1, pm2 = PatchExtractor.patch_name_to_pm(p1), PatchExtractor.patch_name_to_pm(p2)
        if pm1['full_img'] != pm2['full_img']:
            return False
        ps, oh, ow = pm1['ps'], pm1['oh'], pm1['ow']
        disth, distw = (ps - oh) * d, (ps - ow) * d
        (h1, w1), (h2, w2) = PatchExtractor.get_position(p1), PatchExtractor.get_position(p2)
        hmin1, hmax1 = max(0, h1 - disth), h1 + 2 * disth
        wmin1, wmax1 = max(0, w1 - distw), w1 + 2 * distw
        h_ok, hps_ok = hmin1 < h2 < hmax1, hmin1 < h2 + disth < hmax1
        w_ok, wps_ok = wmin1 < w2 < wmax1, wmin1 < w2 + distw < wmax1
        return (h_ok and w_ok) or (h_ok and wps_ok) or (hps_ok and w_ok) or (hps_ok and wps_ok)

    @staticmethod
    def get_neighbors_dict(patches, d=1):
        """Creates dict of neighbors
        :param patches: list of patches
        :param d: int, distance
        :return: tuple, dict of patches (keys) with neighbors list (values), arr of same but with patch indexes,
        arr of arr of index of patches with 0th item arr of patches with 0 neighbors, 1st item 1 neighbor, etc;
        arr of patch group, index 0 is patch name 0 group id, group id is nb of neighbors
        """
        pidx = {p: i for i, p in enumerate(patches)}    # patch name to patch index in lst patches
        full_im__p = {}     # full img name to patch name
        neigh = {}      # patch name to neighboring patches names
        for p in patches:
            neigh[p] = []
            full_im = PatchExtractor.get_full_img_from_patch(p)
            if full_im in full_im__p:
                full_im__p[full_im].append(p)
            else:
                full_im__p[full_im] = [p]

        neigh_pidx = []     # item 0 corresponds to patch at index 0, item is a lst of neighboring patch indexes
        groups_pidx = [[]]  # item 0 corresponds to patches with 0 neighbors, item is lst of patch indexes
        pidx_groups = np.zeros(patches.size, dtype=np.int)  # patch index to corresponding patch group
        for p1 in patches:
            idxs = []
            full_im = PatchExtractor.get_full_img_from_patch(p1)
            for p2 in full_im__p[full_im]:
                if p1 != p2 and PatchExtractor.are_neighbors(p1, p2, d):
                    neigh[p1].append(p2)
                    idxs.append(pidx[p2])
            neigh_pidx.append(np.array(sorted(idxs)))
            if len(idxs) > len(groups_pidx)-1:
                groups_pidx.extend([[] for _ in range(len(idxs) - (len(groups_pidx) - 1))])
            groups_pidx[len(idxs)].append(pidx[p1])
            pidx_groups[pidx[p1]] = len(idxs)
        return neigh, np.array(neigh_pidx), np.array([np.array(g) for g in groups_pidx]), pidx_groups

    @staticmethod
    def compute_side_overlap(n, ps):
        """Computes minimum overlap between patches maximizing the number of patches. Height/width overlap can differ
        :param n: int, total size (img height or width)
        :param ps: int, patch size
        :return: int, patch overlap
        """
        remainder = n % ps
        quotient = max(1, n // ps)
        overlap = math.ceil((ps - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_overlap(patch_name):
        """Extracts patch size from its filename
        :param patch_name: dict, patch map
        :return: tuple, height overlap and width overlap
        """
        pm = PatchExtractor.patch_name_to_pm(patch_name)
        return pm['oh'], pm['ow']

    @staticmethod
    def get_patch_size(patch_name):
        """Extracts patch size from its filename
        :param patch_name: dict, patch map
        :return: int, patch size
        """
        pm = PatchExtractor.patch_name_to_pm(patch_name)
        return pm['ps']

    @staticmethod
    def get_position(patch_name):
        """Extracts patch coordinates from its filename
        :param patch_name: dict, patch map
        :return: tuple, (y, x)
        """
        pm = PatchExtractor.patch_name_to_pm(patch_name)
        return pm['h'], pm['w']

    @staticmethod
    def get_full_img_from_patch(patch_name):
        """Extract full image name from patch filename
        :param patch_name: str, patch filename
        :return: str, full image name
        """
        return PatchExtractor.patch_name_to_pm(patch_name)['full_img']

    @staticmethod
    def create_pm(img_path, ps, oh, ow, h, w, randomly_sampled=False, patch_name=None):
        """Creates patch map from args
        :param img_path: str, full image path
        :param ps: int, patch size
        :param oh: int, height overlap
        :param ow: int, width overlap
        :param h: int, top left corner y coordinate
        :param w: int, top left corner y coordinate
        :param randomly_sampled: bool, if patch was randomly sampled
        :param patch_name: str, patch filename
        :return: dict, patch map
        """
        file, ext = os.path.splitext(os.path.basename(img_path))
        if patch_name is None:
            sep = PatchExtractor.SEP + PatchExtractor.RAND if randomly_sampled else PatchExtractor.SEP
            patch_name = f'{file}{sep}_ps{ps}_oh{oh}_ow{ow}_h{h}_w{w}{ext}'
        return {'patch_path': patch_name, 'full_img': img_path, 'ps': ps, 'oh': oh, 'ow': ow, 'h': h, 'w': w}

    @staticmethod
    def patch_name_re():
        """REGEX used to extract patch information from patch filename
        :return: compiled regex
        """
        sep = fr'(?P<sep>{PatchExtractor.SEP}(?:{PatchExtractor.RAND})?)'
        reg = re.compile(fr'^(?P<img>.+){sep}_ps(?P<ps>\d+)_oh(?P<oh>\d+)_ow(?P<ow>\d+)_h(?P<h>\d+)_w(?P<w>\d+)$')
        return reg

    @staticmethod
    def can_extract_pm_from_patch_name(patch_name):
        """Checks if patch filename contains all necessary patch information
        :param patch_name: str, patch filename
        :return: bool
        """
        file, ext = os.path.splitext(os.path.basename(patch_name))
        return PatchExtractor.patch_name_re().search(file) is not None

    @staticmethod
    def patch_name_to_pm(patch_name):
        """Converts a patch filename to a patch map
        :param patch_name: str, patch filename
        :return: dict, patch map
        """
        file, ext = os.path.splitext(os.path.basename(patch_name))
        results = PatchExtractor.patch_name_re().search(file)
        assert results, f"Cannot extract necessary information from patch name: {patch_name}"
        img_path = results.group('img') + ext
        ps, oh, ow, h, w = tuple((int(results.group(k)) for k in ['ps', 'oh', 'ow', 'h', 'w']))
        randomly_sampled = PatchExtractor.RAND in results.group('sep')
        return PatchExtractor.create_pm(img_path, ps, oh, ow, h, w, randomly_sampled, patch_name)

    @staticmethod
    def rebuild_im_from_patches(pms, patches, full_shape=None, interpol=None):
        """Rebuild image from patches
        :param pms: list of dict, all patch maps
        :param patches: list of array, all patches
        :param full_shape: tuple of int, (h, w) full image size
        :param interpol: cv2 interpolation flag, used for resize (masks cannot be resized like images)
        :return: array, rebuilt image
        """
        if full_shape is None: full_shape = cv2.imread(pms[0]['full_img'], cv2.IMREAD_UNCHANGED).shape
        im = np.zeros(full_shape)
        for pm, patch in zip(pms, patches):
            h, w, ps = pm['h'], pm['w'], pm['ps']
            pdim = ps, ps
            if patch.shape[:2] != pdim:
                if interpol is None: interpol = cv2.INTER_NEAREST if len(patch.shape) < 3 else cv2.INTER_LINEAR
                patch = cv2.resize(patch, pdim, interpolation=interpol)
            im[h:h + ps, w:w + ps] = patch
        return im

    @staticmethod
    def impath_to_patches(img_path, ps):
        """Tiles full image to patches
        :param img_path: str, full image path
        :param ps: int, patch size
        :return: tuple with list of patches and list of patch maps
        """
        return PatchExtractor.im_to_patches(PatchExtractor([ps]).load_image(img_path), ps, img_path)

    @staticmethod
    def im_to_patches(im, ps, imname="fi_img.jpg"):
        """Converts full image array to patches, resizing the image if it was too small with respect to patch size
        :param im: array, image
        :param ps: int, patch size
        :param imname: str, full image name for patch map
        :return: tuple with provided image array resized if necessary, list of patches and list of patch maps
        """
        patcher = PatchExtractor([ps])
        im_resized = patcher.maybe_resize(im)
        pms = patcher.patch_grid(imname, im_resized)
        return im_resized, [PatchExtractor.extract_patch(im_resized, pm) for pm in pms], pms


def multiprocess_patching(proc_id, pmq, patcher, data, dirs, dest):
    """Function used to apply patching in multiprocess mode
    :param proc_id: int, process id
    :param pmq: mpqueue, queue where all patches will be stored
    :param patcher: PatchExtractor, used to extract patches from full images
    :param data: str, root source dir
    :param dirs: str, list of image directories in source dir
    :param dest: str, destination directory path where to save the patches
    """
    pms = []
    for c in dirs:
        print(f"Process {proc_id}: patching {c} patches")
        grids = patcher.dir_images_to_patches(os.path.join(data, c))
        print(f"Process {proc_id}: saving {c} patches")
        patcher.save_patches(data, dest, c, grids)
        pms.append((c, grids))
    pmq.put(pms)


def get_dirs(root, level):
    """Used to get directories containing images at an arbitrary level in the file hierarchy
    :param root: str, root dataset dir
    :param level: int, level to extract subdirectories from
    :return: list of subdirectories
    """
    all_dirs = common.list_dirs(root)
    while level > 0:
        all_dirs = [os.path.join(a, b) for a in all_dirs for b in common.list_dirs(os.path.join(root, a))]
        level -= 1
    return all_dirs

def main(args):
    """Performs the multiprocess patching of dataset according to command line args
    :param args: command line args
    """
    all_dirs = get_dirs(args.data, args.level)
    if not all_dirs:
        all_dirs = ['']
    workers, batch_size, batched_dirs = concurrency.batch_lst(all_dirs)
    patcher = PatchExtractor(args.patch_sizes, dynamic_res=args.dynamic_ps, root_dir=args.data)
    pmq = mp.Queue()
    jobs = []
    for i, dirs in zip(range(workers), batched_dirs):
        jobs.append(mp.Process(target=multiprocess_patching, args=(i, pmq, patcher, args.data, dirs, args.dest)))
        jobs[i].start()
    pms = concurrency.unload_mpqueue(pmq, jobs)
    for j in jobs:
        j.join()
    pickle.dump(pms, open(os.path.join(args.dest, f'patches_{"_".join(map(str, args.patch_sizes))}.p'), "wb"))
    print("done")


def get_patcher_arg_parser():
    """Helper function used to create argparser
    :return: argparser
    """
    parser = argparse.ArgumentParser(description="Creates patch dataset from image dataset")
    parser.add_argument('--data', type=str, required=True, help="source data root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the patches should be saved")
    parser.add_argument('-p', '--patch-sizes', nargs='+', default=[512], type=int, help="patch sizes")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--level', default=0, type=int, help="nested level of class folders compared to args.data")
    parser.add_argument('--dynamic-ps', action='store_true', help="Adjusts ps for imgs res different from median res")
    return parser


if __name__ == '__main__':
    parser = get_patcher_arg_parser()
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    args.data = args.data.rstrip('/')

    args.patch_sizes = sorted(args.patch_sizes)

    if args.dest is None:
        suffix = f'_{"dynamic_" if args.dynamic_ps else ""}patched_{"_".join(map(str, args.patch_sizes))}'
        args.dest = common.maybe_create(f'{args.data}{suffix}')

    common.time_method(main, args)
