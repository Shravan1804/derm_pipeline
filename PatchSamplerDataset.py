import os
import cv2
import math
import random
import numpy as np
from tqdm import tqdm
from scipy import ndimage


class PatchSamplerDataset(object):
    def __init__(self, root, patch_size, sample_n_patch=-1, rotation_padding='constant', seed=42, transforms=None):
        random.seed(seed)
        self.transforms = transforms
        self.patch_size = patch_size
        self.sample_n_patch = sample_n_patch
        self.rotation_padding = rotation_padding
        self.classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c))]
        self.patches = []
        for c in self.classes:
            print("Preparing patches for class", c)
            for img_file in tqdm(list(sorted(os.listdir(os.path.join(root, c))))):
                self.sample_patch_from_img(os.path.join(root, c, img_file), c)
        random.shuffle(self.patches)

    def get_sample_n_patch(self, im):
        im_w, im_h, im_c = im.shape
        return self.sample_n_patch if self.sample_n_patch != -1 \
            else int(im_h * im_w / self.patch_size / self.patch_size)

    def maybe_resize(self, im):
        w, h, c = im.shape
        smallest = min(h, w)
        if smallest < self.patch_size:
            ratio = self.patch_size / smallest
            return cv2.resize(im, (max(self.patch_size, int(w * ratio)),
                                   max(self.patch_size, int(h * ratio))))
        else:
            return im

    def img_as_grid_of_patches(self, cl, img_path, im_arr):
        im_h, im_w, im_c = np.shape(im_arr)
        step_h = self.patch_size - PatchSamplerDataset.get_overlap(im_h, self.patch_size)
        grid_h = np.arange(start=0, stop=im_h - self.patch_size, step=step_h)
        step_w = self.patch_size - PatchSamplerDataset.get_overlap(im_w, self.patch_size)
        grid_w = np.arange(start=0, stop=im_w - self.patch_size, step=step_w)
        grid_idx = [PatchSamplerDataset.get_patch_map(cl, img_path, 0, a, b) for a in grid_h for b in grid_w]
        if not grid_idx:
            grid_idx = [PatchSamplerDataset.get_patch_map(cl, img_path, 0, 0, 0)]
        self.patches.extend(grid_idx)

    def sample_patch_from_img(self, img_path, cl):
        im = self.maybe_resize(cv2.imread(img_path))
        self.img_as_grid_of_patches(cl, img_path, im)
        n_samples = self.get_sample_n_patch(im)
        rotations = np.random.randint(low=1, high=359, size=n_samples).tolist()
        for i, rotation in enumerate(rotations):
            im_arr_rotated = ndimage.rotate(im.astype(np.int16), rotation, reshape=True,
                                            mode=self.rotation_padding, cval=-1)
            h, w, c = np.shape(im_arr_rotated)
            patch = [-1]
            loop_count = 0
            while -1 in patch and loop_count < 1000:
                idx_h, idx_w = (np.random.randint(low=0, high=h - self.patch_size, size=1)[0],
                                np.random.randint(low=0, high=w - self.patch_size, size=1)[0])
                patch = PatchSamplerDataset.get_img_patch(im_arr_rotated, idx_h, idx_w, self.patch_size)
                loop_count += 1
            if loop_count >= 1000:
                continue
            self.patches.append(PatchSamplerDataset.get_patch_map(cl, img_path, rotation, idx_h, idx_w))

    def __getitem__(self, idx):
        patch = self.patches[idx]
        im = self.maybe_resize(cv2.imread(patch['path']))
        im_arr_rotated = ndimage.rotate(im, patch['rotation'], reshape=True,
                                        mode=self.rotation_padding, cval=-1)
        im = im_arr_rotated[patch['idx_h']:patch['idx_h'] + self.patch_size,
             patch['idx_w']:patch['idx_w'] + self.patch_size]
        target = patch['class']

        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target

    def __len__(self):
        return len(self.patches)

    @staticmethod
    def get_overlap(n, div):
        remainder = n % div
        quotient = max(1, int(n / div))
        overlap = math.ceil((div - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_img_patch(im, id_h, id_w, patch_size):
        return im[id_h:id_h + patch_size, id_w:id_w + patch_size]

    @staticmethod
    def get_patch_map(cl, img_path, rotation, idx_h, idx_w):
        return {'class': cl, 'path': img_path, 'rotation': rotation, 'idx_h': idx_h, 'idx_w': idx_w}
