import os
import cv2
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import multiprocessing as mp
import concurrency

class DrawHelper(object):
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
            s = (p['idx_w'], p['idx_h'])
            e = (p['idx_w'] + patch_size, p['idx_h'] + patch_size)
            self.drawrect(img_arr, s, e, (255, 255, 255))
        return img_arr

    def test(self):
        im = np.zeros((800, 800, 3), dtype='uint8')
        patcher = PatchExtractor(256)
        patch_maps = patcher.img_as_grid_of_patches(im, 'test.jpg')
        patcher.draw_patches(im, patch_maps)
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(im)
        plt.axis('off')
        plt.show()
        plt.close()


class PatchExtractor(object):
    SEP = '__SEP__'

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def imgs_to_patches(self, dirname):
        grids = {}
        for img_path in [os.path.join(dirname, im) for im in sorted(os.listdir(dirname))]:
            im = os.path.basename(img_path)
            if im not in grids:
                grids[im] = []
            grids[im].extend(self.img_as_grid_of_patches(self.load_img_from_disk(img_path), img_path))
        return grids

    def img_as_grid_of_patches(self, im_arr, img_path):
        """Converts img into a grid of patches"""
        im_h, im_w = im_arr.shape[:2]
        if im_h < self.patch_size or im_w < self.patch_size:
            raise Exception(f'Error, patch size {self.patch_size} do not fit img shape {im_arr.shape}')
        step_h = self.patch_size - PatchExtractor.get_overlap(im_h, self.patch_size)
        # Don't forget + 1 in the stop argument otherwise the upper bound won't be included
        grid_h = np.arange(start=0, stop=1 + im_h - self.patch_size, step=step_h)
        step_w = self.patch_size - PatchExtractor.get_overlap(im_w, self.patch_size)
        grid_w = np.arange(start=0, stop=1 + im_w - self.patch_size, step=step_w)
        grid_idx = [self.get_patch_map(img_path, 0, a, b) for a in grid_h for b in grid_w]
        if not grid_idx:
            grid_idx = [self.get_patch_map(img_path, 0, 0, 0)]
        return grid_idx

    def maybe_resize(self, im_arr):
        """Resize img only if one of its dimensions is smaller than the patch size otherwise returns img unchanged"""
        h, w = im_arr.shape[:2]
        smallest = min(h, w)
        if smallest < self.patch_size:
            ratio = self.patch_size / smallest
            # resize new dim takes first w then h!
            return cv2.resize(im_arr, (max(self.patch_size, int(w * ratio)), max(self.patch_size, int(h * ratio))))
        else:
            return im_arr

    def load_img_from_disk(self, img_path):
        return self.maybe_resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))

    def get_patch_from_patch_map(self, im, pm):
        return self.get_patch_from_idx(im, pm['idx_h'], pm['idx_w'])

    def get_patch_from_idx(self, im, id_h, id_w):
        return im[id_h:id_h + self.patch_size, id_w:id_w + self.patch_size]

    def get_patch_map(self, img_path, rotation, idx_h, idx_w):
        patch_name = PatchExtractor.get_patch_fname(img_path, idx_h, idx_w)
        return {'patch_path': patch_name, 'rotation': rotation, 'idx_h': idx_h, 'idx_w': idx_w}

    def get_neighboring_patches(self, pm, img_shape, d=1, include_self=True):
        """ Will include itself in the list of neighbors """
        img_name = PatchExtractor.get_img_fname_from_patch(pm)
        max_h, max_w = img_shape[0] - self.patch_size, img_shape[1] - self.patch_size
        step_h = self.patch_size - PatchExtractor.get_overlap(img_shape[0], self.patch_size)
        step_w = self.patch_size - PatchExtractor.get_overlap(img_shape[1], self.patch_size)
        scope_h = [i * step_h for i in range(-d, d+1)]
        scope_w = [i * step_w for i in range(-d, d + 1)]
        neighbors = [(a, b) for a, b in [(m + pm['idx_h'], n + pm['idx_w']) for m in scope_h for n in scope_w]
                     if 0 <= a <= max_h and 0 <= b <= max_w
                     and (include_self or not (a == pm['idx_h'] and b == pm['idx_w']))]
        res = [PatchExtractor.get_patch_fname(img_name, h, w) for h, w in neighbors]
        # print(pm['patch_path'], 'neighbors:', [r['patch_path'] for r in res])
        return res

    def save_patches(self, source, dest, dirname, grids, mask_prefix=None):
        dest = os.path.join(dest, dirname)
        if not os.path.exists(dest):
            os.makedirs(dest)
        for img, patches in grids.items():
            im = self.load_img_from_disk(os.path.join(source, dirname, img))
            if mask_prefix is not None and dirname.startswith(mask_prefix):
                im = im[:, :, 0]    # discard all but first channel
            for pm in patches:
                cv2.imwrite(os.path.join(dest, pm['patch_path']), self.get_patch_from_patch_map(im, pm))

    @staticmethod
    def get_pos_from_patch_name(patch_name):
        ext = os.path.splitext(patch_name)[1]
        return tuple(int(t) for t in patch_name.replace(ext, '').split(f'{PatchExtractor.SEP}_h')[1].split('_w'))

    @staticmethod
    def get_overlap(n, div):
        remainder = n % div
        quotient = max(1, int(n / div))
        overlap = math.ceil((div - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_patch_suffix(idx_h, idx_w):
        return '_h' + str(idx_h) + '_w' + str(idx_w)

    @staticmethod
    def get_img_fname_from_patch(pm):
        return pm['patch_path'].split(PatchExtractor.SEP)[0] + os.path.splitext(pm['patch_path'])[1]

    @staticmethod
    def get_patch_fname(img_path, idx_h, idx_w):
        """Create patch file basename"""
        file, ext = os.path.splitext(os.path.basename(img_path))
        return file + PatchExtractor.SEP + PatchExtractor.get_patch_suffix(idx_h, idx_w) + ext


def multiprocess_patching(proc_id, pmq, patcher, data, dirs, dest, m_prefix):
    pms = []
    for c in dirs:
        grids = patcher.imgs_to_patches(os.path.join(data, c))
        print(f"Process {proc_id}: saving {c} patches")
        patcher.save_patches(data, dest, c, grids, m_prefix)
        pms.append((c, grids))
    pmq.put(pms)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data is None or not os.path.exists(args.data):
        raise Exception("Error, --data invalid")
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    workers, batch_size, batched_dirs = concurrency.batch_dirs(sorted(os.listdir(args.data)))
    patcher = PatchExtractor(args.patch_size)
    pmq = mp.Queue()
    jobs = []
    for i, dirs in zip(range(workers), batched_dirs):
        jobs.append(mp.Process(target=multiprocess_patching, args=(i, pmq, patcher, args.data, dirs, args.dest, args.mdir_prefix)))
        jobs[i].start()
    pms = concurrency.unload_mpqueue(pmq, jobs)
    for j in jobs:
        j.join()
    pickle.dump(pms, open(os.path.join(args.dest, "patches.p"), "wb"))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts dataset to a patch dataset without data augmentation")
    parser.add_argument('--data', type=str, required=True, help="source data root directory absolute path")
    parser.add_argument('--dest', type=str, required=True, help="directory where the patches should be saved")
    parser.add_argument('-p', '--patch-size', default=512, type=int, help="patch size")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--mdir-prefix', type=str, default='masks_', help="prefix of mask dirs (for these we keep only 1 channel)")
    args = parser.parse_args()

    main()
