import os
import re
import sys
import math
import pickle
import argparse
import multiprocessing as mp

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, concurrency


class DrawHelper:
    """Helper class to draw patches delimitations on images"""
    def __init__(self, thickness=1, style='dotted', gap=10):
        self.thickness = thickness
        self.style = style
        self.gap = gap

    def drawline(self, im, pt1, pt2, color):
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
        e = pts[0]
        pts.append(pts.pop(0))
        for p in pts:
            s = e
            e = p
            self.drawline(im, s, e, color)

    def drawrect(self, im, pt1, pt2, color):
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        self.drawpoly(im, pts, color)

    def draw_patches(self, img_arr, pm, patch_size):
        for p in pm:
            s = (p['w'], p['h'])
            e = (p['w'] + patch_size, p['h'] + patch_size)
            self.drawrect(img_arr, s, e, (255, 255, 255))
        return img_arr

    def test(self):
        im = np.zeros((800, 800, 3), dtype='uint8')
        patcher = PatchExtractor(256)
        patch_maps = patcher.patch_grid(im, 'test.jpg')
        self.draw_patches(im, patch_maps, 256)
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(im)
        plt.axis('off')
        plt.show()
        plt.close()


class PatchExtractor:
    """Extracts patches from images either by random sampling or splitting the image as a grid"""
    SEP = '__SEP__'
    RAND = '__RAND__'

    def __init__(self, patch_sizes, seed=42):
        """Seed used when sampling patches from image"""
        self.patch_sizes = sorted(patch_sizes) if type(patch_sizes) == list else [patch_sizes]
        np.random.seed(seed)

    def dir_images_to_patches(self, dirname):
        files = common.list_files(dirname, full_path=True)
        exclude = [p for p in files if not(p.endswith('.png') or p.endswith('.jpg'))]
        if exclude:
            print("Warning: In", dirname, "these files are excluded:", exclude)
        return {os.path.basename(f): self.patch_grid(f) for f in files if f not in exclude}

    def sample_patches(self, img_path, nb, im_arr=None):
        """Samples nb patches from img"""
        assert nb > 1, f"Cannot sample nb={nb} patches from {img_path}."
        im_arr = self.load_image(img_path) if im_arr is None else self.maybe_resize(im_arr)
        im_h, im_w = im_arr.shape[:2]
        patches = []
        for ps in self.patch_sizes:
            idx_h, idx_w = (np.random.randint(low=0, high=1 + im_h - ps, size=nb),
                            np.random.randint(low=0, high=1 + im_w - ps, size=nb))
            patches.extend([PatchExtractor.create_pm(img_path, ps, 0, 0, h, w, True) for h, w in zip(idx_h, idx_w)])
        return patches

    def patch_grid(self, img_path, im_arr=None):
        """Converts img into a grid of patches"""
        im_arr = self.load_image(img_path) if im_arr is None else self.maybe_resize(im_arr)
        im_h, im_w = im_arr.shape[:2]
        patches = []
        for ps in self.patch_sizes:
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
        """Resize img only if dim smaller than the max patch size otherwise returns img unchanged"""
        ps = self.patch_sizes[-1]
        h, w = im_arr.shape[:2]
        smallest = min(h, w)
        if smallest < ps:
            ratio = ps / smallest
            # resize new dim takes first w then h!
            return cv2.resize(im_arr, (max(ps, int(w * ratio)), max(ps, int(h * ratio))))
        else:
            return im_arr

    def load_image(self, img_path):
        common.check_file_valid(img_path)
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return self.maybe_resize(im)

    def save_patches(self, source, dest, dirname, grids):
        dest = common.maybe_create(dest, dirname)
        for img, patches in grids.items():
            im = self.load_image(os.path.join(source, dirname, img))
            for pm in patches:
                cv2.imwrite(os.path.join(dest, pm['patch_path']), self.extract_patch(im, pm))

    @staticmethod
    def extract_patch(im, pm):
        """Extracts patch from im using patch map info"""
        ps, id_h, id_w = pm['ps'], pm['h'], pm['w']
        return im[id_h:id_h + ps, id_w:id_w + ps]

    @staticmethod
    def are_neighbors(p1, p2, d=1):
        """p1, p2 the patch filenames, o the overlap (oh, ow) between patches, will use p1 patch size if available
        Uses parameter ps=512 only if ps not available in patch_name"""
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
        """Returns dict of patches with neighbors; arr of same but with patch indexes;
        arr of arr of index of patches with 0th item arr of patches with 0 neighbors, 1st item 1 neighbor, etc;
        arr of patch group, index 0 is patch name 0 group id, group id is nb of neighbors"""
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
        """Computes minimum overlap between patches, n is the total size (img height or width), ps is the patch size"""
        remainder = n % ps
        quotient = max(1, n // ps)
        overlap = math.ceil((ps - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_overlap(patch_name):
        """Extracts patch size from its filename"""
        pm = PatchExtractor.patch_name_to_pm(patch_name)
        return pm['oh'], pm['ow']

    @staticmethod
    def get_patch_size(patch_name):
        """Extracts patch size from its filename"""
        pm = PatchExtractor.patch_name_to_pm(patch_name)
        return pm['ps']

    @staticmethod
    def get_position(patch_name):
        """Extracts patch coordinates from its filename"""
        pm = PatchExtractor.patch_name_to_pm(patch_name)
        return pm['h'], pm['w']

    @staticmethod
    def get_full_img_from_patch(patch_name):
        """patch_name is the patch filename"""
        return PatchExtractor.patch_name_to_pm(patch_name)['full_img']

    @staticmethod
    def create_pm(img_path, ps, oh, ow, h, w, randomly_sampled=False, patch_name=None):
        """Creates patch map"""
        file, ext = os.path.splitext(os.path.basename(img_path))
        if patch_name is None:
            sep = PatchExtractor.SEP + PatchExtractor.RAND if randomly_sampled else PatchExtractor.SEP
            patch_name = f'{file}{sep}_ps{ps}_oh{oh}_ow{ow}_h{h}_w{w}{ext}'
        return {'patch_path': patch_name, 'full_img': img_path, 'ps': ps, 'oh': oh, 'ow': ow, 'h': h, 'w': w}

    @staticmethod
    def patch_name_re():
        sep = fr'(?P<sep>{PatchExtractor.SEP}(?:{PatchExtractor.RAND})?)'
        reg = re.compile(fr'^(?P<img>.+){sep}_ps(?P<ps>\d+)_oh(?P<oh>\d+)_ow(?P<ow>\d+)_h(?P<h>\d+)_w(?P<w>\d+)$')
        return reg

    @staticmethod
    def can_extract_pm_from_patch_name(patch_name):
        file, ext = os.path.splitext(os.path.basename(patch_name))
        return PatchExtractor.patch_name_re().search(file) is not None

    @staticmethod
    def patch_name_to_pm(patch_name):
        file, ext = os.path.splitext(os.path.basename(patch_name))
        results = PatchExtractor.patch_name_re().search(file)
        assert results, f"Cannot extract necessary information from patch name: {patch_name}"
        img_path = results.group('img') + ext
        ps, oh, ow, h, w = tuple((int(results.group(k)) for k in ['ps', 'oh', 'ow', 'h', 'w']))
        randomly_sampled = PatchExtractor.RAND in results.group('sep')
        return PatchExtractor.create_pm(img_path, ps, oh, ow, h, w, randomly_sampled, patch_name)

    @staticmethod
    def rebuild_im_from_patches(pms, patches, full_shape=None):
        if full_shape is None: full_shape = cv2.imread(pms[0]['full_img'], cv2.IMREAD_UNCHANGED).shape
        im = np.zeros(full_shape)
        for pm, patch in zip(pms, patches):
            h, w, ps = pm['h'], pm['w'], pm['ps']
            im[h:h + ps, w:w + ps] = patch
        return im


def multiprocess_patching(proc_id, pmq, patcher, data, dirs, dest):
    pms = []
    for c in dirs:
        print(f"Process {proc_id}: patching {c} patches")
        grids = patcher.dir_images_to_patches(os.path.join(data, c))
        print(f"Process {proc_id}: saving {c} patches")
        patcher.save_patches(data, dest, c, grids)
        pms.append((c, grids))
    pmq.put(pms)


def get_dirs(root, level):
    all_dirs = common.list_dirs(root)
    while level > 0:
        all_dirs = [os.path.join(a, b) for a in all_dirs for b in common.list_dirs(os.path.join(root, a))]
        level -= 1
    return all_dirs

def main(args):
    all_dirs = get_dirs(args.data, args.level)
    if not all_dirs:
        all_dirs = ['']
    workers, batch_size, batched_dirs = concurrency.batch_lst(all_dirs)
    patcher = PatchExtractor(args.patch_sizes)
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


def get_patcher_arg_parser(desc="Creates patch dataset from image dataset"):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, required=True, help="source data root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the patches should be saved")
    parser.add_argument('-p', '--patch-sizes', nargs='+', default=[512], type=int, help="patch sizes")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--level', default=0, type=int, help="nested level of class folders compared to args.data")
    return parser


if __name__ == '__main__':
    parser = get_patcher_arg_parser()
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    args.data = args.data.rstrip('/')

    args.patch_sizes = sorted(args.patch_sizes)

    if args.dest is None:
        args.dest = common.maybe_create(f'{args.data}_patched_{"_".join(map(str, args.patch_sizes))}')

    common.time_method(main, args)
