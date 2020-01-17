import os
import numpy as np
from PatchSamplerDataset import PatchSamplerDataset


class ClassificationPatchSamplerDataset(PatchSamplerDataset):
    def __init__(self, root, patch_size, **kwargs):
        self.root = self.validate_path(root)
        self.classes = [c for c in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, c))]
        super().__init__(self.root, patch_size, **kwargs)

        if len(self.get_patch_list()) == 0:
            for c in self.classes:
                print("Class", c)
                self.root_img_dir = c
                class_patches = self.prepare_patches_from_imgs(os.path.join(self.root, c))
                for key in self.patches_keys:
                    self.patches[key].extend([{'class': c, **m} for m in class_patches[key]])
            self.root_img_dir = None
            for key in self.patches_keys:
                np.random.shuffle(self.patches[key])
            self.save_patches_to_disk()

    def __getitem__(self, idx):
        im, patch_map = super().__getitem__(idx)
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

