import os
import sys
import numpy as np
from PatchSamplerDataset import PatchSamplerDataset


class ClassificationPatchSamplerDataset(PatchSamplerDataset):
    def __init__(self, root, patch_size, is_test, split_data=False, **kwargs):
        self.root = root
        self.classes = [c for c in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, c))]
        super().__init__(self.root, patch_size, is_test=is_test, split_data=split_data, **kwargs)

        if len(self.train_patches) == 0:
            for c in self.classes:
                print("Class", c)
                self.root_img_dir = c
                class_patches = self.prepare_patches_from_imgs(os.path.join(self.root, c))
                class_patches = [[{'class': c, **m} for m in n] for n in class_patches]
                if self.split_data:
                    self.train_patches.extend(class_patches[0])
                    self.test_patches.extend(class_patches[1])
                else:
                    self.get_patch_list().extend(class_patches[0])
            self.root_img_dir = None
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

    def create_cache_dirs(self):
        """Creates the dirs where the rotated images and the patches will be saved on disk"""
        super().create_cache_dirs()
        create_dirs = []
        if not os.path.exists(os.path.join(self.rotated_img_dir, self.classes[0])):
            create_dirs.append(self.rotated_img_dir)
        if not os.path.exists(os.path.join(self.patches_dir, self.classes[0])):
            create_dirs.append(self.patches_dir)
        if create_dirs:
            list(map(os.makedirs, [os.path.join(m, n) for m in create_dirs for n in self.classes]))

    def is_valid_patch(self, patch_map):
        return True

    def get_img_path_from_patch_map(self, patch_map):
        path = super().get_img_path_from_patch_map(patch_map)
        c = os.path.basename(os.path.dirname(patch_map['patch_path']))
        return os.path.join(os.path.dirname(path), c, os.path.basename(path))

