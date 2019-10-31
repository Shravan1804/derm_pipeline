
import os
import cv2
import math
import random
import numpy as np
from scipy import ndimage
from tqdm import tqdm

class PatchSamplerDataset(object):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate
    # rotation_padding=‘reflect’ or  ‘constant’ or ‘nearest’ or ‘mirror’ or ‘wrap’
    def __init__(self, root, patch_size, patch_per_img=-1, n_rotation=6, rotation_padding='constant',
                 seed=42, transforms=None, save_dir='/tmp'):
        random.seed(seed)
        self.root = root
        self.transforms = transforms
        self.patch_size = patch_size
        self.patch_per_img = patch_per_img
        self.rotations = np.linspace(0, 360, n_rotation, endpoint=False, dtype=np.int).tolist()
        self.rotation_padding = rotation_padding
        rotated_dir = type(self).__name__ + '-rotated-pad' + rotation_padding
        self.save_img_rotated = os.path.join(save_dir, rotated_dir)
        if not os.path.exists(self.save_img_rotated):
            os.makedirs(self.save_img_rotated)
        patches_file = type(self).__name__ + '-patches-p' + str(patch_size) + '-n' + str(patch_per_img) + \
                       '-r' + str(n_rotation) + '-pad' + rotation_padding + '-seed' + str(seed) + '.npy'
        self.patches_path = os.path.join(save_dir, patches_file)
        self.patches = np.load(self.patches_path).tolist() if os.path.exists(self.patches_path) else []
        if len(self.patches) > 0:
            print("Loaded patches from previous run, seed = ", seed)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.patches)

    def maybe_resize(self, im_arr):
        w, h, c = im_arr.shape
        smallest = min(h, w)
        if smallest < self.patch_size:
            ratio = self.patch_size / smallest
            return cv2.resize(im_arr, (max(self.patch_size, int(w * ratio)), max(self.patch_size, int(h * ratio))))
        else:
            return im_arr

    def prepare_patches_from_img_files(self, dir_path, img_files):
        sampled_patches = []
        for img_file in tqdm(img_files):
            img_path = os.path.join(dir_path, img_file)
            sampled_patches.extend(self.img_as_grid_of_patches(img_path))
            sampled_patches.extend(self.sample_random_patch_from_img(img_path))
        return sampled_patches

    def img_as_grid_of_patches(self, img_path):
        im_arr = self.maybe_resize(cv2.imread(img_path))
        im_h, im_w, im_c = np.shape(im_arr)
        step_h = self.patch_size - PatchSamplerDataset.get_overlap(im_h, self.patch_size)
        grid_h = np.arange(start=0, stop=im_h - self.patch_size, step=step_h)
        step_w = self.patch_size - PatchSamplerDataset.get_overlap(im_w, self.patch_size)
        grid_w = np.arange(start=0, stop=im_w - self.patch_size, step=step_w)
        grid_idx = [PatchSamplerDataset.get_patch_map(img_path, 0, a, b) for a in grid_h for b in grid_w]
        if not grid_idx:
            grid_idx = [PatchSamplerDataset.get_patch_map(img_path, 0, 0, 0)]
        grid_idx = [x for x in grid_idx if self.is_valid_patch(x)]
        return grid_idx


    def sample_random_patch_from_img(self, img_path):
        im_arr = self.maybe_resize(cv2.imread(img_path))
        n_samples = max(1, int(self.get_nb_patch_per_img(im_arr) / len(self.rotations)))
        sampled_patches = []
        for rotation in self.rotations:
            im_arr_rotated, path_to_save = self.get_rotated_img(img_path, im_arr, rotation)
            h, w, c = np.shape(im_arr_rotated)
            for _ in range(n_samples):
                patch = [-1]
                loop_count = 0
                while (-1 in patch or not self.is_valid_patch(patch_map)) and loop_count < 1000:
                    idx_h, idx_w = (np.random.randint(low=0, high=h - self.patch_size, size=1)[0],
                                    np.random.randint(low=0, high=w - self.patch_size, size=1)[0])
                    patch = self.get_patch_from_idx(im_arr_rotated, idx_h, idx_w)
                    patch_map = PatchSamplerDataset.get_patch_map(path_to_save, rotation, idx_h, idx_w)
                    loop_count += 1
                if loop_count >= 1000:
                    continue
                sampled_patches.append(patch_map)
        return sampled_patches

    def is_valid_patch(self, patch_map):
        raise NotImplementedError

    def get_nb_patch_per_img(self, im_arr):
        im_w, im_h, im_c = im_arr.shape
        return self.patch_per_img if self.patch_per_img != -1 \
            else int(im_h * im_w / self.patch_size / self.patch_size)

    #rotate if was not already done once and saved
    def get_rotated_img(self, img_path, im_arr, rotation):
        if rotation == 0:
            return im_arr, img_path
        file, ext = os.path.splitext(os.path.basename(img_path))
        new_filename = file + '-r' + str(rotation)
        path_to_save = os.path.join(self.save_img_rotated, new_filename + ext)
        if not os.path.exists(path_to_save):
            im_arr_rotated = ndimage.rotate(im_arr.astype(np.int16), rotation, reshape=True,
                                            mode=self.rotation_padding, cval=-1)
            im_arr_rotated = self.maybe_resize(im_arr_rotated)
            cv2.imwrite(path_to_save, im_arr_rotated)
        else:
            im_arr_rotated = self.maybe_resize(cv2.imread(path_to_save))
        return im_arr_rotated, path_to_save

    def get_patch_from_idx(self, im, id_h, id_w):
        return im[id_h:id_h + patch_size, id_w:id_w + self.patch_size]

    def get_patch_from_patch_map(self, patch_map):
        im = self.maybe_resize(cv2.imread(patch_map['path']))
        return self.get_patch_from_idx(im, id_h, id_w)

    def save_patches(self):
        np.save(self.patches_path, self.patches)

    @staticmethod
    def get_overlap(n, div):
        remainder = n % div
        quotient = max(1, int(n / div))
        overlap = math.ceil((div - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_patch_map(img_path, rotation, idx_h, idx_w):
        return {'path': img_path, 'rotation': rotation, 'idx_h': idx_h, 'idx_w': idx_w}

    @staticmethod
    def get_filename_no_ext(path):
        return os.path.splitext(os.path.basename(path))[0]
