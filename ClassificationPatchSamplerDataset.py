import os
import random
import numpy as np
from PatchSamplerDataset import PatchSamplerDataset


class ClassificationPatchSamplerDataset(PatchSamplerDataset):
    def __init__(self, root, patch_size, is_train, **kwargs):
        super().__init__(root, patch_size, is_train, **kwargs)
        self.classes = [c for c in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, c))]

        if len(self.patches) == 0:
            for c in self.classes:
                print("Preparing patches for class", c)
                class_patches = self.prepare_patches_from_imgs(os.path.join(self.root, c))
                print()
                self.patches.extend([{'class': c, **m} for m in class_patches])
            random.shuffle(self.patches)
            self.populate_train_test_lists()
            self.save_patch_maps_to_disk()
            self.save_patches_to_disk()

    def __getitem__(self, idx):
        patch_map = self.get_patch_list()[idx]
        im = self.load_patch_from_patch_map(patch_map)
        target = patch_map['class']

        if self.transforms is not None:
            im, target = self.transforms(im, target)

        return im, target

    def is_valid_patch(self, patch_map):
        return True
