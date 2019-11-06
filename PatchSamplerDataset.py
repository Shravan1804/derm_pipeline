import os
import cv2
import math
import random
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from timeit import default_timer as timer
import datetime


class PatchSamplerDataset(object):
    img_dir = 'images'
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate
    # rotation_padding=‘reflect’ or  ‘constant’ or ‘nearest’ or ‘mirror’ or ‘wrap’
    def __init__(self, root, patch_size, is_train, patch_per_img=-1, n_rotation=6, rotation_padding='constant',
                 seed=42, test_prop=0.15, transforms=None):
        random.seed(seed)
        self.root = root
        self.is_train = is_train
        self.test_prop = test_prop
        self.transforms = transforms
        self.patch_size = patch_size
        self.patch_per_img = patch_per_img
        self.rotations = np.linspace(0, 360, n_rotation, endpoint=False, dtype=np.int).tolist()
        self.rotation_padding = rotation_padding
        rotated_dir = 'cache_' + os.path.basename(self.root) + '_' + type(self).__name__ + '-rpad' + rotation_padding
        self.save_img_rotated = os.path.join(os.path.dirname(self.root), rotated_dir)
        if not os.path.exists(self.save_img_rotated):
            os.makedirs(os.path.join(self.save_img_rotated, PatchSamplerDataset.img_dir))
        patches_file = rotated_dir + '-patches-p' + str(patch_size) + '-n' + str(patch_per_img) \
                       + '-r' + str(n_rotation) + '-seed' + str(seed) + '.npy'
        self.patches_dir = os.path.join(os.path.dirname(self.root), patches_file.replace('.npy', ''))
        self.patches_path = os.path.join(self.patches_dir, patches_file)
        if not os.path.exists(self.patches_dir):
            os.makedirs(os.path.join(self.patches_dir, PatchSamplerDataset.img_dir))
        self.patches = []
        self.patches_train = []
        self.patches_test = []
        if os.path.exists(self.patches_path):
            self.patches = np.load(self.patches_path, allow_pickle=True).tolist()
            self.populate_train_test_lists()

        if len(self.patches) > 0:
            print("Loaded patches from previous run, seed = ", seed)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.patches_train) if self.is_train else len(self.patches_test)

    def populate_train_test_lists(self):
        for patch_map in self.patches:
            if patch_map['train_test'] == 'train':
                self.patches_train.append(patch_map)
            else:
                self.patches_test.append(patch_map)

    def maybe_resize(self, im_arr):
        h, w = im_arr.shape[:2]
        smallest = min(h, w)
        if smallest < self.patch_size:
            ratio = self.patch_size / smallest
            # resize new dim takes first w then h!
            return cv2.resize(im_arr, (max(self.patch_size, int(w * ratio)), max(self.patch_size, int(h * ratio))))
        else:
            return im_arr

    def prepare_patches_from_imgs(self, dir_path):
        start = timer()
        img_files = list(sorted(os.listdir(dir_path)))
        sampled_patches = []
        print("Sampling patches ...")
        for img_file in tqdm(img_files):
            train_or_test = np.random.choice(['train', 'test'], size=1, p=[1 - self.test_prop, self.test_prop])[0]
            img_path = os.path.join(dir_path, img_file)
            img_patches = []
            img_patches.extend(self.img_as_grid_of_patches(img_path))
            img_patches.extend(self.sample_random_patch_from_img(img_path))
            sampled_patches.extend([{'train_test': train_or_test, **m} for m in img_patches])
        # store all patches in cache
        end = timer()
        print("Done,", dir_path, "images were processed in", datetime.timedelta(seconds=end - start))
        return sampled_patches

    def img_as_grid_of_patches(self, img_path):
        im_arr = self.maybe_resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
        im_h, im_w = im_arr.shape[:2]
        step_h = self.patch_size - PatchSamplerDataset.get_overlap(im_h, self.patch_size)
        grid_h = np.arange(start=0, stop=1 + im_h - self.patch_size, step=step_h)
        step_w = self.patch_size - PatchSamplerDataset.get_overlap(im_w, self.patch_size)
        grid_w = np.arange(start=0, stop=1 + im_w - self.patch_size, step=step_w)
        grid_idx = [self.get_patch_map(img_path, 0, a, b) for a in grid_h for b in grid_w]
        if not grid_idx:
            grid_idx = [self.get_patch_map(img_path, 0, 0, 0)]
        grid_idx = [x for x in grid_idx if self.is_valid_patch(x)]
        return grid_idx

    def sample_random_patch_from_img(self, img_path):
        im_arr = self.maybe_resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
        n_samples = max(1, int(self.get_nb_patch_per_img(im_arr) / len(self.rotations)))
        sampled_patches = []
        for rotation in self.rotations:
            im_arr_rotated, path_to_save = self.get_rotated_img(img_path, im_arr, rotation)
            h, w = im_arr_rotated.shape[:2]
            for _ in range(n_samples):
                patch = [-1]
                loop_count = 0
                while (-1 in patch or not self.is_valid_patch(patch_map)) and loop_count < 1000:
                    idx_h, idx_w = (np.random.randint(low=0, high=1 + h - self.patch_size, size=1)[0],
                                    np.random.randint(low=0, high=1 + w - self.patch_size, size=1)[0])
                    patch = self.get_patch_from_idx(im_arr_rotated, idx_h, idx_w)
                    patch_map = self.get_patch_map(path_to_save, rotation, idx_h, idx_w)
                    loop_count += 1
                if loop_count >= 1000:
                    continue
                sampled_patches.append(patch_map)
        return sampled_patches

    def is_valid_patch(self, patch_map):
        raise NotImplementedError

    def save_patches_map(self):
        np.save(self.patches_path, self.patches)

    def store_patches(self):
        for patch_map in tqdm(self.patches):
            _ = self.get_patch_from_patch_map(patch_map)

    def get_nb_patch_per_img(self, im_arr):
        im_h, im_w = im_arr.shape[:2]
        return self.patch_per_img if self.patch_per_img != -1 \
            else int(im_h * im_w / self.patch_size / self.patch_size)

    # rotate if was not already done once and saved
    def get_rotated_img(self, img_path, im_arr, rotation):
        if rotation == 0:
            return im_arr, img_path
        file, ext = os.path.splitext(os.path.basename(img_path))
        new_filename = file + '-r' + str(rotation)
        path_to_save = os.path.join(self.save_img_rotated, PatchSamplerDataset.img_dir, new_filename + ext)
        if not os.path.exists(path_to_save):
            im_arr_rotated = ndimage.rotate(im_arr.astype(np.int16), rotation, reshape=True,
                                            mode=self.rotation_padding, cval=-1)
            im_arr_rotated = self.maybe_resize(im_arr_rotated)
            cv2.imwrite(path_to_save, im_arr_rotated)
        else:
            im_arr_rotated = self.maybe_resize(cv2.imread(path_to_save, cv2.IMREAD_UNCHANGED))
        return im_arr_rotated, path_to_save

    def get_patch_from_idx(self, im, id_h, id_w):
        return im[id_h:id_h + self.patch_size, id_w:id_w + self.patch_size]

    def get_patch_from_patch_map(self, patch_map):
        if os.path.exists(patch_map['patch_path']):
            return cv2.imread(patch_map['patch_path'], cv2.IMREAD_UNCHANGED)
        else:
            img_path = os.path.join(os.path.dirname(self.root), patch_map['img_path'])
            im = self.maybe_resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
            patch = self.get_patch_from_idx(im, patch_map['idx_h'], patch_map['idx_w'])
            cv2.imwrite(patch_map['patch_path'], patch)
            return patch

    def get_patch_map(self, img_path, rotation, idx_h, idx_w):
        img_path = img_path.replace(os.path.dirname(self.root) + '/', '')
        file, ext = os.path.splitext(os.path.basename(img_path))
        if rotation == 0:  # otherwise already present in img filename
            file += '_r' + str(rotation)
        patch_path = os.path.join(self.patches_dir, PatchSamplerDataset.img_dir, file + '_h' + str(idx_h)
                                  + '_w' + str(idx_w) + ext)
        return {'img_path': img_path, 'patch_path': patch_path, 'rotation': rotation, 'idx_h': idx_h, 'idx_w': idx_w}

    @staticmethod
    def get_overlap(n, div):
        remainder = n % div
        quotient = max(1, int(n / div))
        overlap = math.ceil((div - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_fname_no_ext(path):
        return os.path.splitext(os.path.basename(path))[0]
