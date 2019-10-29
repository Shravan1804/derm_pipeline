import os
import cv2
import math
import random
import numpy as np
from tqdm import tqdm
from scipy import ndimage


class PatchSamplerDataset(object):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate
    # rotation_padding=‘reflect’ or  ‘constant’ or ‘nearest’ or ‘mirror’ or ‘wrap’
    def __init__(self, root, patch_size, sample_n_patch=-1, n_rotation=6, rotation_padding='constant', seed=42,
                 transforms=None, save_dir=None):
        random.seed(seed)
        self.transforms = transforms
        self.patch_size = patch_size
        self.save_dir_rotation = '/tmp/img-rotation' if save_dir is None \
            else os.path.join(save_dir, 'img-rotation')
        if not os.path.exists(self.save_dir_rotation):
            os.makedirs(self.save_dir_rotation)
        self.sample_n_patch = sample_n_patch
        self.rotations = np.linspace(0, 360, n_rotation, endpoint=False, dtype=np.int).tolist()
        self.rotation_padding = rotation_padding
        self.classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c))]
        patches_file = 'patches-p' + str(patch_size) + '-n' + str(sample_n_patch) + '-r' + str(n_rotation) + \
                       '-pad' + rotation_padding + '-seed' + str(seed) + '.npy'
        patches_path = os.path.join('/tmp', patches_file) if save_dir is None else os.path.join(save_dir, patches_file)
        if os.path.exists(patches_path):
            print("loading patches from previous run, seed = ", seed)
            self.patches = np.load(patches_path).tolist()
        else:
            self.patches = []
            for c in self.classes:
                print("Preparing patches for class", c)
                for img_file in tqdm(list(sorted(os.listdir(os.path.join(root, c))))):
                    self.sample_patch_from_img(os.path.join(root, c, img_file), c)
                print()
            random.shuffle(self.patches)
            np.save(patches_path, self.patches)

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
        grid_idx = [PatchSamplerDataset.get_patch_map(cl, img_path, a, b) for a in grid_h for b in grid_w]
        if not grid_idx:
            grid_idx = [PatchSamplerDataset.get_patch_map(cl, img_path, 0, 0)]
        self.patches.extend(grid_idx)

    def sample_patch_from_img(self, img_path, cl):
        im = self.maybe_resize(cv2.imread(img_path))
        self.img_as_grid_of_patches(cl, img_path, im)
        n = int(self.get_sample_n_patch(im) / len(self.rotations))
        for rotation in self.rotations:
            if rotation == 0:
                im_arr_rotated = im
                path_to_save = img_path
            else:
                file, ext = os.path.splitext(os.path.basename(img_path))
                path_to_save = os.path.join(self.save_dir_rotation, file + '-r' + str(rotation) + ext)
                if not os.path.exists(path_to_save):
                    im_arr_rotated = ndimage.rotate(im.astype(np.int16), rotation, reshape=True,
                                                    mode=self.rotation_padding, cval=-1)
                    im_arr_rotated = self.maybe_resize(im_arr_rotated)
                    cv2.imwrite(path_to_save, im_arr_rotated)
                else:
                    im_arr_rotated = self.maybe_resize(cv2.imread(path_to_save))
            h, w, c = np.shape(im_arr_rotated)
            for i in range(n):
                patch = [-1]
                loop_count = 0
                while -1 in patch and loop_count < 1000:
                    idx_h, idx_w = (np.random.randint(low=0, high=h - self.patch_size, size=1)[0],
                                    np.random.randint(low=0, high=w - self.patch_size, size=1)[0])
                    patch = PatchSamplerDataset.get_img_patch(im_arr_rotated, idx_h, idx_w, self.patch_size)
                    loop_count += 1
                if loop_count >= 1000:
                    continue
                self.patches.append(PatchSamplerDataset.get_patch_map(cl, path_to_save, idx_h, idx_w))

    def __getitem__(self, idx):
        patch = self.patches[idx]
        im = self.maybe_resize(cv2.imread(patch['path']))
        im = im[patch['idx_h']:patch['idx_h'] + self.patch_size, patch['idx_w']:patch['idx_w'] + self.patch_size]
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
    def get_patch_map(cl, img_path, idx_h, idx_w):
        return {'class': cl, 'path': img_path, 'idx_h': idx_h, 'idx_w': idx_w}


def main(args):
    dataset = PatchSamplerDataset('/home/shravan/deep-learning/data/skin_body_location_crops/train', 256, 20,
                                  rotation_padding='mirror')


if __name__ == "__main__":
    main(None)
