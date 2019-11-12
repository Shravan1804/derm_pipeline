import os
import numpy as np
from PatchSamplerDataset import PatchSamplerDataset


class ClassificationPatchSamplerDataset(PatchSamplerDataset):
    def __init__(self, root, patch_size, is_train, **kwargs):
        super().__init__(root, patch_size, is_train, **kwargs)
        self.classes = [c for c in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, c))]

        if len(self.train_patches) == 0:
            for c in self.classes:
                print("Preparing patches for class", c)
                class_patches = self.prepare_patches_from_imgs(os.path.join(self.root, c))
                print()
                self.train_patches.extend([{'class': c, **m} for m in class_patches[0]])
                self.test_patches.extend([{'class': c, **m} for m in class_patches[1]])
            np.random.shuffle(self.train_patches)
            np.random.shuffle(self.test_patches)
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
