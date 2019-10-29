import os
import cv2
import math
import random
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from PatchSamplerDataset import PatchSamplerDataset


class ClassificationPatchSamplerDataset(PatchSamplerDataset):
    def __init__(self, root, patch_size, patch_per_img=-1, n_rotation=6, rotation_padding='constant', seed=42,
                 transforms=None, save_dir=None):
        super().__init__(patch_size, patch_per_img, n_rotation, rotation_padding, seed, transforms, save_dir)
        self.classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c))]

        self.patches = np.load(self.patches_path).tolist() if os.path.exists(self.patches_path) else []
        if len(self.patches) > 0:
            print("loaded patches from previous run, seed = ", seed)
        else:
            for c in self.classes:
                print("Preparing patches for class", c)
                count = 0
                class_patches = []
                for img_file in tqdm(list(sorted(os.listdir(os.path.join(root, c))))):
                    img_path = os.path.join(root, c, img_file)
                    class_patches.extend(self.img_as_grid_of_patches(img_path))
                    class_patches.extend(self.sample_random_patch_from_img(img_path))
                    count += 1
                    if count > 2: break
                print()
                self.patches.extend([{'class': c, **m} for m in class_patches])
            random.shuffle(self.patches)
            self.save_patches()

    def __getitem__(self, idx):
        patch = self.patches[idx]
        im = self.maybe_resize(cv2.imread(patch['path']))
        im = im[patch['idx_h']:patch['idx_h'] + self.patch_size, patch['idx_w']:patch['idx_w'] + self.patch_size]
        target = patch['class']

        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target

    def is_valid_patch(self, patch_map):
        return True


def main(args):
    dataset = ClassificationPatchSamplerDataset('/home/shravan/deep-learning/data/skin_body_location_crops/train', 256)
    c = tmp = 0
    for d in dataset.patches:
        print(d)
        if "tmp" in d['path']: tmp +=1
        else: c+=1
    print(type(dataset).__name__)
    print('tmp:',tmp,'vs not tmp:',c)

if __name__ == "__main__":
    main(None)
