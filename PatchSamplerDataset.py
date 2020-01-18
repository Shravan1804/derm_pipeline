import os
import cv2
import math
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from timeit import default_timer as timer
import datetime
import pickle

from sklearn.model_selection import train_test_split, KFold


class PatchSamplerDataset(object):
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
    def __init__(self, root, patch_size, is_val=False, is_test=False, patch_per_img=-1, n_rotation=6, rotation_padding='reflect',
                 seed=42, test_size=0.15, cross_val_bypass=False, fold=0, nfolds=5, cross_val=False, split_data=True, root_img_dir=None, dest=None, transforms=None):
        self.seed = seed
        np.random.seed(self.seed)
        self.root = self.validate_path(root)
        self.root_img_dir = root_img_dir

        self.cross_val = cross_val
        if not self.cross_val:
            self.is_val = False
            self.cross_val_bypass = False
            self.patches_keys = ['train']
        else:
            self.cross_val_bypass = cross_val_bypass
            self.is_val = is_val
            self.nfolds = nfolds
            if fold > self.nfolds:
                raise Exception(f"Error, fold={fold} is greater than nfolds={self.nfolds}")
            else:
                self.fold = fold
            self.patches_keys = [lab + str(i) for lab in ['train', 'val'] for i in range(nfolds)]

        self.split_data = split_data
        if not self.split_data:
            self.is_test = False
        else:
            self.is_test = is_test
            self.test_size = test_size
            self.patches_keys += ['test']


        self.dest = self.get_default_cache_root_dir() if dest is None else dest
        self.transforms = transforms
        self.patch_size = patch_size
        self.patch_per_img = patch_per_img
        self.rotations = np.linspace(0, 360, n_rotation, endpoint=False, dtype=np.int).tolist()
        self.rotation_padding = rotation_padding
        self.create_cache_dirs()

        if os.path.exists(self.patches_path):
            self.patches = pickle.load(open(self.patches_path, "rb"))
            self.patches_keys = list(self.patches.keys())
            if (not self.cross_val) and ('train' not in self.patches_keys):
                self.cross_val = True
                self.cross_val_bypass = True
                print("Loaded dataset was built with cross-val => setting cross-val and bypass to true")
        else:
            self.patches = {key: [] for key in self.patches_keys}

        if len(self.get_patch_list()) > 0:
            print("Loaded patches from previous run, seed = ", seed)

    def validate_path(self, path):
        if path.endswith('/'):
            return path[:-1]
        else:
            return path

    def get_height_and_width(self, idx):
        im, pm = self[idx]
        return im.shape[:-1]

    def __getitem__(self, idx):
        patch_map = self.get_patch_list()[idx]
        img = cv2.cvtColor(self.load_patch_from_patch_map(patch_map), cv2.COLOR_BGR2RGB)
        return img, patch_map

    def __len__(self):
        return len(self.get_patch_list())

    def create_cache_dirs(self):
        """Creates the dirs where the rotated images and the patches will be saved on disk"""
        rotated_dir = 'cache_' + self.get_dataset_name() + '-rpad' + self.rotation_padding
        self.rotated_img_dir = os.path.join(self.dest, rotated_dir)
        if not os.path.exists(self.rotated_img_dir):
            os.makedirs(self.get_img_dir(self.rotated_img_dir))
        patches_file = rotated_dir + '-patches-p' + str(self.patch_size) + '-n' + str(self.patch_per_img) \
                       + '-r' + str(len(self.rotations)) + '-seed' + str(self.seed) + '.p'
        self.patches_dir = os.path.join(self.dest, patches_file.replace('.p', ''))
        if not os.path.exists(self.patches_dir):
            os.makedirs(self.get_img_dir(self.patches_dir))
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
        print("Sampling patches ...")
        img_sets = {'train': [os.path.join(dir_path, x) for x in sorted(os.listdir(dir_path))]}
        if self.split_data:
            train, test = train_test_split(img_sets.pop('train'), test_size=self.test_size, random_state=self.seed)
            print("Test images:", list(map(os.path.basename, test)))
            img_sets = {'train': train, 'test': test}
        ps = {}
        for key in img_sets.keys():
            imgs = img_sets[key]
            # merge list of lists: [[img patches] * number of images]
            p = PatchSamplerDataset.flatten(map(self.extract_img_patches, tqdm(imgs)))
            np.random.shuffle(p)
            ps[key] = p
        if self.cross_val:
            data = ps.pop('train')
            imgs = np.array(img_sets['train'])
            kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=self.seed)
            for i, idx in enumerate(kf.split(imgs)):
                val = [os.path.splitext(os.path.basename(x))[0] for x in imgs[idx[1]]]
                ps['train' + str(i)], ps['val' + str(i)] = [], []
                for d in data:
                    base_img = os.path.basename(self.get_img_path_from_patch_map(d)).split('-r')[0]
                    if base_img in val:
                        ps['val' + str(i)].append(d)
                    else:
                        ps['train' + str(i)].append(d)
        print("Done,", dir_path, "images were processed in", datetime.timedelta(seconds=timer() - start))
        return ps

    def extract_img_patches(self, img_path):
        """Extract patches from img at all rotations"""
        im_arr = self.load_img_from_disk(img_path)
        n_samples = max(1, int(self.get_nb_patch_per_img(im_arr) / len(self.rotations)))
        sampled_patches = []
        for rotation in self.rotations:
            im_arr_rotated, path_to_save = self.get_rotated_img(img_path, im_arr, rotation)
            if rotation == 0:
                sampled_patches.extend(self.img_as_grid_of_patches(im_arr_rotated, path_to_save))
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

    def img_as_grid_of_patches(self, im_arr, img_path):
        """Converts img into a grid of patches and returns the valid patches in the grid"""
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

    def get_rotated_img(self, img_path, im_arr, rotation):
        """Rotates image if needed, loads from disk if was already rotated before.
        Returns the rotated image and the path where it was saved """
        file, ext = os.path.splitext(os.path.basename(img_path))
        new_filename = file + '-r' + str(rotation)
        rotated_img_path = os.path.join(self.get_img_dir(self.rotated_img_dir), new_filename + ext)
        if not os.path.exists(rotated_img_path):
            im_arr_rotated = im_arr if rotation == 0 else \
                ndimage.rotate(im_arr, rotation, reshape=True, mode=self.rotation_padding)
            im_arr_rotated = self.maybe_resize(im_arr_rotated)
            cv2.imwrite(rotated_img_path, im_arr_rotated)
        else:
            im_arr_rotated = self.load_img_from_disk(rotated_img_path)
        return im_arr_rotated, rotated_img_path

    def is_valid_patch(self, patch_map):
        raise NotImplementedError

    def print_patches_overview(self):
        for k, v in self.patches.items():
            print(k, 'contains', len(v), 'patches')

    def load_img_from_disk(self, img_path):
        """Load image and resize is smaller than patch size"""
        return self.maybe_resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))

    def save_patch_maps_to_disk(self):
        pickle.dump((self.patches), open(self.patches_path, "wb"))

    def get_all_patches(self):
        return [elem for lst in [self.patches[k] for k in self.patches_keys] for elem in lst]

    def save_patches_to_disk(self):
        print("Saving all patches on disk ...")
        self.save_patch_maps_to_disk()
        for patch_map in tqdm(self.get_all_patches()):
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
            im = self.load_img_from_disk(self.get_img_path_from_patch_map(patch_map))
            patch = self.get_patch_from_idx(im, patch_map['idx_h'], patch_map['idx_w'])
            if cache:
                cv2.imwrite(patch_path, patch)
            return patch

    def get_patch_map(self, img_path, rotation, idx_h, idx_w):
        patch_path = self.create_patch_path(img_path, idx_h, idx_w)
        patch_path = self.get_relative_path(patch_path)
        return {'patch_path': patch_path, 'rotation': rotation, 'idx_h': idx_h, 'idx_w': idx_w}

    def create_patch_path(self, img_path, h, w):
        return os.path.join(self.get_img_dir(self.patches_dir), PatchSamplerDataset.get_patch_fname(img_path, h, w))

    def get_img_path_from_patch_map(self, patch_map):
        patch_suffix = PatchSamplerDataset.get_patch_suffix(patch_map['idx_h'], patch_map['idx_w'])
        img_name = os.path.basename(patch_map['patch_path']).replace(patch_suffix, '')
        return os.path.join(self.get_img_dir(self.rotated_img_dir), img_name)

    def get_relative_path(self, path):
        return path.replace(self.dest + '/', '')

    def get_absolute_path(self, path):
        return os.path.join(self.dest, path)

    def get_img_dir(self, path):
        """Method needed to be able to handle datasets where imgs are in specific subfolder"""
        return path if self.root_img_dir is None else os.path.join(path, self.root_img_dir)

    def get_dataset_name(self):
        dataset_name = os.path.basename(self.root)
        if self.split_data:
            return dataset_name
        else:
            return dataset_name + '_' + os.path.basename(os.path.dirname(self.root))

    def get_default_cache_root_dir(self):
        """Needs self.split_data and self.root set. If not self.split_data assumes data already separated in e.g. train, valid dir"""
        dir_name = os.path.dirname(self.root)
        if self.split_data:
            return dir_name
        else:
            return os.path.dirname(dir_name)

    def get_patch_list(self):
        if self.is_test:
            return self.patches['test']
        elif self.cross_val:
            if self.cross_val_bypass:
                return self.patches['train' + str(self.fold)] + self.patches['val' + str(self.fold)]
            elif self.is_val:
                return self.patches['val' + str(self.fold)]
            else:
                return self.patches['train' + str(self.fold)]
        else:
            return self.patches['train']

    @staticmethod
    def flatten(lst):
        """Flattens lst of lsts"""
        return [elem for sublst in lst for elem in sublst]

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
    def get_patch_fname(img_path, idx_h, idx_w):
        file, ext = os.path.splitext(os.path.basename(img_path))
        return file + PatchSamplerDataset.get_patch_suffix(idx_h, idx_w) + ext








