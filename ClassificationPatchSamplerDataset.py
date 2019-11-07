import os
import random
import numpy as np
from PatchSamplerDataset import PatchSamplerDataset


class ClassificationPatchSamplerDataset(PatchSamplerDataset):
    def __init__(self, root, patch_size, is_train, patch_per_img=-1, n_rotation=6, rotation_padding='wrap',
                 seed=42, test=0.15, transforms=None):
        super().__init__(root, patch_size, is_train, patch_per_img, n_rotation, rotation_padding, seed, test, transforms)
        self.classes = [c for c in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, c))]

        if len(self.patches) == 0:
            for c in self.classes:
                print("Preparing patches for class", c)
                class_patches = self.prepare_patches_from_imgs(os.path.join(self.root, c))
                print()
                self.patches.extend([{'class': c, **m} for m in class_patches])
            random.shuffle(self.patches)
            self.populate_train_test_lists()
            self.save_patches_map()
            print("Storing all patch in cache ...")
            self.store_patches()

    def __getitem__(self, idx):
        patch_map = self.get_patch_list()[idx]
        im = self.get_patch_from_patch_map(patch_map)
        target = patch_map['class']

        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target

    def is_valid_patch(self, patch_map):
        return super().is_valid_patch(patch_map)
