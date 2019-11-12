import os
import cv2
import math
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from timeit import default_timer as timer
import datetime
import pickle
import itertools

from sklearn.model_selection import train_test_split


class PatchSamplerDataset(object):
    root_img_dir = 'images'

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate
    # rotation_padding=‘reflect’ or  ‘constant’ or ‘nearest’ or ‘mirror’ or ‘wrap’
    #    mode       |   Ext   |         Input          |   Ext
    # -----------+---------+------------------------+---------
    # 'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
    # 'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6
    # 'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
    # 'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
    # 'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3

    # TODO: Find why ndimage.rotate produces different sizes for img than for mask...
    def __init__(self, root, patch_size, is_train, root_img_dir='images', patch_per_img=-1, n_rotation=6, rotation_padding='wrap',
                 seed=42, test_size=0.15, transforms=None):
        self.seed = seed
        np.random.seed(self.seed)
        self.root = root
        self.is_train = is_train
        self.root_img_dir = root_img_dir
        self.test_size = test_size
        self.transforms = transforms
        self.patch_size = patch_size
        self.patch_per_img = patch_per_img
        self.rotations = np.linspace(0, 360, n_rotation, endpoint=False, dtype=np.int).tolist()
        self.rotation_padding = rotation_padding
        self.create_cache_dirs()
        self.train_patches, self.test_patches = ([] for _ in range(2))
        if os.path.exists(self.patches_path):
            self.train_patches, self.test_patches = pickle.load(open(self.patches_path, "rb"))

        if len(self.train_patches) > 0:
            print("Loaded patches from previous run, seed = ", seed)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.train_patches) if self.is_train else len(self.test_patches)

    def create_cache_dirs(self):
        """Creates the dirs where the rotated images and the patches will be saved on disk"""
        rotated_dir = 'cache_' + os.path.basename(self.root) + '_' + type(self).__name__ + '-rpad' \
                      + self.rotation_padding
        self.rotated_img_dir = os.path.join(os.path.dirname(self.root), rotated_dir)
        if not os.path.exists(self.rotated_img_dir):
            os.makedirs(os.path.join(self.rotated_img_dir, self.root_img_dir))
        patches_file = rotated_dir + '-patches-p' + str(self.patch_size) + '-n' + str(self.patch_per_img) \
                       + '-r' + str(len(self.rotations)) + '-seed' + str(self.seed) + '.p'
        self.patches_dir = os.path.join(os.path.dirname(self.root), patches_file.replace('.p', ''))
        if not os.path.exists(self.patches_dir):
            os.makedirs(os.path.join(self.patches_dir, self.root_img_dir))
        self.patches_path = os.path.join(self.patches_dir, patches_file)

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

    def prepare_patches_from_imgs(self, dir_path):
        """Sample patches from imgs present in the specified directory"""
        start = timer()
        train, test = train_test_split(sorted(os.listdir(dir_path)), test_size=self.test_size, random_state=self.seed)
        print("Test images:", test)
        print("Sampling patches ...")
        train_p = list(itertools.chain(*map(lambda x: self.extract_img_patches(os.path.join(dir_path, x)), tqdm(train))))
        test_p = list(itertools.chain(*map(lambda x: self.extract_img_patches(os.path.join(dir_path, x)), tqdm(test))))
        np.random.shuffle(train_p)
        np.random.shuffle(test_p)
        print("Done,", dir_path, "images were processed in", datetime.timedelta(seconds=timer() - start))
        return train_p, test_p

    def extract_img_patches(self, img_path):
        im_arr = self.load_img_from_disk(img_path)
        return self.img_as_grid_of_patches(img_path, im_arr) + self.sample_random_patch_from_img(img_path, im_arr)

    def img_as_grid_of_patches(self, img_path, im_arr=None):
        """Converts img into a grid of patches and returns the valid patches in the grid"""
        im_arr = self.load_img_from_disk(img_path) if im_arr is None else im_arr
        im_h, im_w = im_arr.shape[:2]
        step_h = self.patch_size - PatchSamplerDataset.get_overlap(im_h, self.patch_size)
        # Don't forget + 1 in the stop argument otherwise the upper bound won't be included
        grid_h = np.arange(start=0, stop=1 + im_h - self.patch_size, step=step_h)
        step_w = self.patch_size - PatchSamplerDataset.get_overlap(im_w, self.patch_size)
        grid_w = np.arange(start=0, stop=1 + im_w - self.patch_size, step=step_w)
        grid_idx = [self.get_patch_map(img_path, 0, a, b) for a in grid_h for b in grid_w]
        if not grid_idx:
            grid_idx = [self.get_patch_map(img_path, 0, 0, 0)]
        grid_idx = list(filter(self.is_valid_patch, grid_idx))
        return grid_idx

    def sample_random_patch_from_img(self, img_path, im_arr=None):
        """Samples random patches from img at all rotations"""
        im_arr = self.load_img_from_disk(img_path) if im_arr is None else im_arr
        n_samples = max(1, int(self.get_nb_patch_per_img(im_arr) / len(self.rotations)))
        sampled_patches = []
        for rotation in self.rotations:
            im_arr_rotated, path_to_save = self.get_rotated_img(img_path, im_arr, rotation)
            h, w = im_arr_rotated.shape[:2]
            for _ in range(n_samples):
                loop_count = 0
                while loop_count == 0 or (not self.is_valid_patch(patch_map) and loop_count < 1000):
                    # Don't forget + 1 in the stop argument otherwise the upper bound won't be included
                    idx_h, idx_w = (np.random.randint(low=0, high=1 + h - self.patch_size, size=1)[0],
                                    np.random.randint(low=0, high=1 + w - self.patch_size, size=1)[0])
                    patch_map = self.get_patch_map(path_to_save, rotation, idx_h, idx_w)
                    loop_count += 1
                if loop_count >= 1000:
                    continue
                sampled_patches.append(patch_map)
        return sampled_patches

    def get_rotated_img(self, img_path, im_arr, rotation):
        """Rotates image if needed, loads from disk if was already rotated before.
        Returns the rotated image and the path where it was saved """
        if rotation == 0:
            return im_arr, img_path
        file, ext = os.path.splitext(os.path.basename(img_path))
        new_filename = file + '-r' + str(rotation)
        rotated_img_path = os.path.join(self.rotated_img_dir, self.root_img_dir, new_filename + ext)
        if not os.path.exists(rotated_img_path):
            im_arr_rotated = ndimage.rotate(im_arr, rotation, reshape=True,
                                            mode=self.rotation_padding)
            im_arr_rotated = self.maybe_resize(im_arr_rotated)
            cv2.imwrite(rotated_img_path, im_arr_rotated)
        else:
            im_arr_rotated = self.load_img_from_disk(rotated_img_path)
        return im_arr_rotated, rotated_img_path

    def is_valid_patch(self, patch_map):
        raise NotImplementedError

    def load_img_from_disk(self, img_path):
        return self.maybe_resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))

    def save_patch_maps_to_disk(self):
        pickle.dump((self.train_patches, self.test_patches), open(self.patches_path, "wb"))

    def save_patches_to_disk(self):
        print("Saving all patches on disk ...")
        self.save_patch_maps_to_disk()
        for patch_map in tqdm(self.train_patches + self.test_patches):
            _ = self.load_patch_from_patch_map(patch_map, cache=True)

    def get_nb_patch_per_img(self, im_arr):
        im_h, im_w = im_arr.shape[:2]
        return self.patch_per_img if self.patch_per_img != -1 \
            else int(im_h * im_w / self.patch_size / self.patch_size)

    def get_patch_from_idx(self, im, id_h, id_w):
        return im[id_h:id_h + self.patch_size, id_w:id_w + self.patch_size]

    def load_patch_from_patch_map(self, patch_map, cache=False):
        patch_path = self.get_absolute_path(patch_map['patch_path'])
        if os.path.exists(patch_path):
            return self.load_img_from_disk(patch_path)
        else:
            im = self.load_img_from_disk(self.get_absolute_path(patch_map['img_path']))
            patch = self.get_patch_from_idx(im, patch_map['idx_h'], patch_map['idx_w'])
            if cache:
                cv2.imwrite(patch_path, patch)
            return patch

    def get_patch_map(self, img_path, rotation, idx_h, idx_w):
        img_path = self.get_relative_path(img_path)
        file, ext = os.path.splitext(os.path.basename(img_path))
        if rotation == 0:  # otherwise already present in img filename
            file += '_r' + str(rotation)
        patch_path = os.path.join(self.patches_dir, self.root_img_dir, file + '_h' + str(idx_h)
                                  + '_w' + str(idx_w) + ext)
        patch_path = self.get_relative_path(patch_path)
        return {'img_path': img_path, 'patch_path': patch_path, 'rotation': rotation, 'idx_h': idx_h, 'idx_w': idx_w}

    def get_relative_path(self, path):
        return path.replace(os.path.dirname(self.root) + '/', '')

    def get_absolute_path(self, path):
        return os.path.join(os.path.dirname(self.root), path)

    def get_patch_list(self):
        return self.train_patches if self.is_train else self.test_patches

    @staticmethod
    def get_overlap(n, div):
        remainder = n % div
        quotient = max(1, int(n / div))
        overlap = math.ceil((div - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_fname_no_ext(path):
        return os.path.splitext(os.path.basename(path))[0]
