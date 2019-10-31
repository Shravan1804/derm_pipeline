import os
import random
import numpy as np
from PatchSamplerDataset import PatchSamplerDataset


class ClassificationPatchSamplerDataset(PatchSamplerDataset):
    def __init__(self, root, patch_size, patch_per_img=-1, n_rotation=6, rotation_padding='constant', seed=42,
                 transforms=None, save_dir=None):
        super().__init__(root, patch_size, patch_per_img, n_rotation, rotation_padding, seed, transforms, save_dir)
        self.classes = [c for c in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, c))]

        if len(self.patches) == 0:
            for c in self.classes:
                print("Preparing patches for class", c)
                class_patches = self.prepare_patches_from_img_files(self.root, list(sorted(os.listdir(os.path.join(root, c)))))
                print()
                self.patches.extend([{'class': c, **m} for m in class_patches])
            random.shuffle(self.patches)
            self.save_patches()

    def __getitem__(self, idx):
        patch_map = self.patches[idx]
        im = self.get_patch_from_patch_map(patch_map)
        target = patch_map['class']

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
