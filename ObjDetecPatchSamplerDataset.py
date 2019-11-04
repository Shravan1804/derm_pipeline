import os
import cv2
import random
import numpy as np
from scipy import ndimage
from PIL import Image
import sys

import torch

from PatchSamplerDataset import PatchSamplerDataset


class ObjDetecPatchSamplerDataset(PatchSamplerDataset):
    mask_file_ext = '.png'

    def __init__(self, root, patch_size, patch_per_img=-1, n_rotation=6, rotation_padding='constant', seed=42,
                 transforms=None):
        super().__init__(root, patch_size, patch_per_img, n_rotation, rotation_padding, seed, transforms)
        self.masks_dirs = [m for m in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, m))
                        and m.startswith('masks_')]
        if len(self.patches) == 0:
            images_path = os.path.join(self.root, 'images')
            self.patches.extend(self.prepare_patches_from_img_files(images_path, list(sorted(os.listdir(images_path)))))
            random.shuffle(self.patches)
            self.save_patches()

    def __getitem__(self, idx):
        patch_map = self.patches[idx]
        img = Image.fromarray(self.get_patch_from_patch_map(patch_map)).convert("RGB")
        raw_masks = self.get_masks_from_patch_map(patch_map)

        classes = masks = None
        boxes = []
        for i, mask in enumerate(raw_masks):
            c, b, m = ObjDetecPatchSamplerDataset.process_mask(mask)
            if b is None: continue  # empty mask
            classes = c if classes is None else np.append(classes, (i + 1) * c)
            boxes += b
            masks = m if masks is None else np.append(masks, m, axis=0)

        num_objs = len(classes)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except IndexError as error:
            print(
            "Image", self.imgs[idx], "with mask", self.masks[idx], "and boxes", boxes, "raised an IndexError exception")
            raise error
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def is_valid_patch(self, patch_map):
        is_valid = False
        for mask in self.get_masks_from_patch_map(patch_map):
            is_valid = is_valid or np.unique(mask).size > 1
        return is_valid

    def get_rotated_img(self, img_path, im_arr, rotation):
        im_arr_rotated, path_to_save = super().get_rotated_img(img_path, im_arr, rotation)
        if rotation != 0:
            mask_file = ObjDetecPatchSamplerDataset.get_mask_filename(img_path)
            rotated_mask_file = ObjDetecPatchSamplerDataset.get_mask_filename(path_to_save)
            for masks_dir in self.masks_dirs:
                rotated_mask_dir_path = os.path.join(self.save_img_rotated, masks_dir)
                if not os.path.exists(rotated_mask_dir_path):
                    os.makedirs(rotated_mask_dir_path)
                rotated_mask_file_path = os.path.join(rotated_mask_dir_path, rotated_mask_file)
                if not os.path.exists(rotated_mask_file_path):
                    mask_arr = cv2.imread(os.path.join(self.root, masks_dir, mask_file), cv2.IMREAD_UNCHANGED)
                    rotated_mask_arr = ndimage.rotate(mask_arr, rotation, reshape=True,
                                                    mode=self.rotation_padding)
                    rotated_mask_arr = self.maybe_resize(rotated_mask_arr)
                    cv2.imwrite(rotated_mask_file_path, rotated_mask_arr)

        return im_arr_rotated, path_to_save

    def get_masks_from_patch_map(self, patch_map):
        mask_file = ObjDetecPatchSamplerDataset.get_mask_filename(patch_map['path'])
        masks = []
        for masks_dir in self.masks_dirs:
            mask_dir_path = os.path.join(self.save_img_rotated, masks_dir) if patch_map['rotation'] != 0 \
                else os.path.join(self.root, masks_dir)
            masks.append(cv2.imread(os.path.join(mask_dir_path, mask_file), cv2.IMREAD_UNCHANGED))
        return masks

    @staticmethod
    def process_mask(mask):
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        if (len(obj_ids) < 2): return None, None, None  # no object, only background
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        return np.ones(num_objs, dtype=np.int), boxes, masks

    @staticmethod
    def get_mask_filename(path):
        return PatchSamplerDataset.get_filename_no_ext(path) + ObjDetecPatchSamplerDataset.mask_file_ext



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
