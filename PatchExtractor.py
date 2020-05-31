import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import multiprocessing as mp
import concurrency
import common

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
            s = (p['idx_w'], p['idx_h'])
            e = (p['idx_w'] + patch_size, p['idx_h'] + patch_size)
            self.drawrect(img_arr, s, e, (255, 255, 255))
        return img_arr

    def test(self):
        im = np.zeros((800, 800, 3), dtype='uint8')
        patcher = PatchExtractor(256)
        patch_maps = patcher.patch_grid(im, 'test.jpg')
        patcher.draw_patches(im, patch_maps)
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

    def __init__(self, patch_size, seed=42):
        """Seed used when sampling patches from image"""
        self.patch_size = patch_size
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
        idx_h, idx_w = (np.random.randint(low=0, high=1 + im_h - self.patch_size, size=nb),
                        np.random.randint(low=0, high=1 + im_w - self.patch_size, size=nb))
        return [PatchExtractor.create_pm(img_path, h, w) for h, w in zip(idx_h, idx_w)]

    def patch_grid(self, img_path, im_arr=None):
        """Converts img into a grid of patches"""
        im_arr = self.load_image(img_path) if im_arr is None else self.maybe_resize(im_arr)
        im_h, im_w = im_arr.shape[:2]
        step_h = self.patch_size - PatchExtractor.get_overlap(im_h, self.patch_size)
        # Don't forget + 1 in the stop argument otherwise the upper bound won't be included
        grid_h = np.arange(start=0, stop=1 + im_h - self.patch_size, step=step_h)
        step_w = self.patch_size - PatchExtractor.get_overlap(im_w, self.patch_size)
        grid_w = np.arange(start=0, stop=1 + im_w - self.patch_size, step=step_w)
        grid_idx = [PatchExtractor.create_pm(img_path, a, b) for a in grid_h for b in grid_w]
        return grid_idx if grid_idx else [PatchExtractor.create_pm(img_path, 0, 0)]

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

    def load_image(self, img_path):
        common.check_file_valid(img_path)
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return self.maybe_resize(im)

    def pm_to_patch(self, im, pm):
        """Creates patch from patch map"""
        return self.extract_patch(im, pm['idx_h'], pm['idx_w'])

    def extract_patch(self, im, id_h, id_w):
        """Extracts patch at provided position"""
        return im[id_h:id_h + self.patch_size, id_w:id_w + self.patch_size]

    def find_full_img_res_from_patches(self, patches):
        positions = [PatchExtractor.get_position(p) for p in patches]
        max_h = max([p[0] for p in positions])
        max_w = max([p[1] for p in positions])
        return max_h + self.patch_size, max_w + self.patch_size

    def neighboring_patches(self, pm, img_shape, d=1, include_self=True):
        img_name = PatchExtractor.get_full_img(pm)
        max_h, max_w = img_shape[0] - self.patch_size, img_shape[1] - self.patch_size
        step_h = self.patch_size - PatchExtractor.get_overlap(img_shape[0], self.patch_size)
        step_w = self.patch_size - PatchExtractor.get_overlap(img_shape[1], self.patch_size)
        scope_h = [i * step_h for i in range(-d, d+1)]
        scope_w = [i * step_w for i in range(-d, d + 1)]
        neighbors = [(a, b) for a, b in [(m + pm['idx_h'], n + pm['idx_w']) for m in scope_h for n in scope_w]
                     if 0 <= a <= max_h and 0 <= b <= max_w
                     and (include_self or not (a == pm['idx_h'] and b == pm['idx_w']))]
        res = [PatchExtractor.get_patch_name(img_name, h, w) for h, w in neighbors]
        # print(pm['patch_path'], 'neighbors:', [r['patch_path'] for r in res])
        return res

    def are_neighbors(self, p1, p2, d=1, first=True):
        dist = self.patch_size * d
        h1, w1 = PatchExtractor.get_position(p1)
        hmin1, hmax1 = max(0, h1 - dist), h1 + self.patch_size + dist
        wmin1, wmax1 = max(0, w1 - dist), w1 + self.patch_size + dist
        h2, w2 = PatchExtractor.get_position(p2)
        # recursion for hypothetic case where one patch is englobed in the second
        return (hmin1 <= h2 + self.patch_size <= hmax1 and wmin1 <= w2 + self.patch_size <= wmax1) or\
               (hmin1 <= h2 <= hmax1 and wmin1 <= w2 <= wmax1) or (first and self.are_neighbors(p2, p1, d, False))

    def save_patches(self, source, dest, dirname, grids, mask_prefix=None):
        dest = common.maybe_create(dest, dirname)
        for img, patches in grids.items():
            im = self.load_image(os.path.join(source, dirname, img))
            if mask_prefix is not None and dirname.startswith(mask_prefix):     # mask img are 2D
                im = im[:, :, 0]    # discard all but first channel
            for pm in patches:
                cv2.imwrite(os.path.join(dest, pm['patch_path']), self.pm_to_patch(im, pm))

    @staticmethod
    def create_pm(img_path, idx_h, idx_w, randomly_sampled=False):
        """Creates patch map"""
        patch_name = PatchExtractor.get_patch_name(img_path, idx_h, idx_w, randomly_sampled)
        return {'patch_path': patch_name, 'full_img': img_path, 'idx_h': idx_h, 'idx_w': idx_w}

    @staticmethod
    def pm_from_patch(patch_name):
        pm = PatchExtractor.create_pm(PatchExtractor.get_full_img_from_patch(patch_name),
                                      *PatchExtractor.get_position(patch_name))
        assert pm['patch_path'] == os.path.basename(patch_name)
        return pm

    @staticmethod
    def get_position(patch_name):
        """Extracts patch coordinates from its filename"""
        ext = os.path.splitext(patch_name)[1]
        return tuple(int(t) for t in patch_name.replace(ext, '').split(f'{PatchExtractor.SEP}_h')[1].split('_w'))

    @staticmethod
    def get_overlap(n, div):
        """Computes minimum overlap between patches"""
        remainder = n % div
        quotient = max(1, int(n / div))
        overlap = math.ceil((div - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_full_img(pm):
        """Returns filename of full image"""
        return os.path.basename(pm['full_img'])

    @staticmethod
    def get_full_img_from_patch(patch_name):
        return os.path.basename(patch_name).split(PatchExtractor.SEP)[0] + os.path.splitext(patch_name)[1]

    @staticmethod
    def get_patch_suffix(idx_h, idx_w):
        return '_h' + str(idx_h) + '_w' + str(idx_w)

    @staticmethod
    def get_patch_name(img_path, idx_h, idx_w, randomly_sampled=False):
        """Create patch file basename"""
        file, ext = os.path.splitext(os.path.basename(img_path))
        sep = PatchExtractor.SEP + PatchExtractor.RAND if randomly_sampled else PatchExtractor.SEP
        return file + sep + PatchExtractor.get_patch_suffix(idx_h, idx_w) + ext


def multiprocess_patching(proc_id, pmq, patcher, data, dirs, dest, m_prefix):
    pms = []
    for c in dirs:
        print(f"Process {proc_id}: patching {c} patches")
        grids = patcher.dir_images_to_patches(os.path.join(data, c))
        print(f"Process {proc_id}: saving {c} patches")
        patcher.save_patches(data, dest, c, grids, m_prefix)
        pms.append((c, grids))
    pmq.put(pms)

def main(args):
    all_dirs = [d for d in sorted(os.listdir(args.data)) if os.path.isdir(os.path.join(args.data, d))]
    workers, batch_size, batched_dirs = concurrency.batch_dirs(all_dirs)
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


def get_patcher_arg_parser(desc="Creates patch dataset from image dataset"):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, required=True, help="source data root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the patches should be saved")
    parser.add_argument('-p', '--patch-size', default=512, type=int, help="patch size")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--mdir-prefix', type=str, default='masks_',
                        help="prefix of mask dirs (for these we keep only 1 channel)")
    return parser

if __name__ == '__main__':
    parser = get_patcher_arg_parser()
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)
    if args.dest is None:
        args.dest = common.maybe_create(os.path.dirname(args.data),
                                        f'{os.path.basename(args.data)}_patched{args.patch_size}')

    main(args)
