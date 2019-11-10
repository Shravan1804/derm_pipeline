import os
import sys
import cv2
import random
import numpy as np
from scipy import ndimage
from PIL import Image
from tqdm import tqdm
import torch

from PatchSamplerDataset import PatchSamplerDataset


class ObjDetecPatchSamplerDataset(PatchSamplerDataset):

    def __init__(self, root, patch_size, is_train, mask_file_ext='.png', min_object_px_size=30, **kwargs):
        self.root = root
        self.patch_size = patch_size
        self.mask_file_ext = mask_file_ext
        self.min_object_px_size = min_object_px_size
        self.masks_dirs = [m for m in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, m))
                           and m.startswith('masks_')]
        self.crop_dataset()
        super().__init__(self.root, self.patch_size, is_train, **kwargs)
        if len(self.patches) == 0:
            self.patches.extend(self.prepare_patches_from_imgs(os.path.join(self.root, self.root_img_dir)))
            random.shuffle(self.patches)
            self.populate_train_test_lists()
            self.save_patch_maps_to_disk()
            self.save_patches_to_disk()

    def __getitem__(self, idx):
        patch_map = self.get_patch_list()[idx]
        img = Image.fromarray(cv2.cvtColor(self.load_patch_from_patch_map(patch_map), cv2.COLOR_BGR2RGB)).convert("RGB")
        raw_masks = self.load_masks_from_patch_map(patch_map)

        classes = masks = None
        boxes = []
        for i, mask in enumerate(raw_masks):
            c, b, m = ObjDetecPatchSamplerDataset.process_mask(mask)
            if b is None: continue  # empty mask
            # all objects from one mask belong to the same class
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
                "Image", self.imgs[idx], "with mask", self.masks[idx], "and boxes", boxes,
                "raised an IndexError exception")
            raise error
        # TODO: suppose all instances are not crowd
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

    def create_cache_dirs(self):
        super().create_cache_dirs()
        create_dirs = []
        if not os.path.exists(os.path.join(self.rotated_img_dir, self.masks_dirs[0])):
            create_dirs.append(self.rotated_img_dir)
        if not os.path.exists(os.path.join(self.patches_dir, self.masks_dirs[0])):
            create_dirs.append(self.patches_dir)
        if create_dirs:
            list(map(os.makedirs, [os.path.join(m, n) for m in create_dirs for n in self.masks_dirs]))

    def crop_dataset(self):
        """Prepares cropped dataset according to patch_size. Crops imgs to the bounding box encompassing objects from
        all masks + patch_size/2"""
        cropped_dir = os.path.join(os.path.dirname(self.root), 'cache_' + os.path.basename(self.root) + '-cropped-p'
                                   + str(self.patch_size))
        if not os.path.exists(cropped_dir):
            list(map(os.makedirs, [os.path.join(cropped_dir, n) for n in self.masks_dirs + [self.root_img_dir]]))

            print("Cropping dataset ...")
            offset = int(self.patch_size / 2) + 1  # + 1 to comprise upper bound
            stats = [0, 0]
            data_dirs = [os.path.join(self.root, self.root_img_dir)] + [os.path.join(self.root, masks_dir)
                                                                        for masks_dir in self.masks_dirs]
            data_files = [sorted(os.listdir(p)) for p in data_dirs]
            imgs_masks = list(map(lambda x: [os.path.join(x[0], n) for n in x[1]], zip(data_dirs, data_files)))
            # iterate over (img, mask1, mask2, ...) tuples
            for img_masks in tqdm(zip(*imgs_masks)):
                im = self.load_img_from_disk(img_masks[0])
                im_h, im_w = im.shape[:2]
                stats[0] += im_h * im_w
                masks = list(map(self.load_img_from_disk, img_masks[1:]))
                masks_not_bg = list(map(lambda x: x > 0, masks))
                boxes = map(ObjDetecPatchSamplerDataset.get_bounding_box_of_true_values, masks_not_bg)
                boxes = list(zip(*list(filter(None.__ne__, boxes))))
                box = [max(0, min(boxes[0]) - offset), max(0, min(boxes[1]) - offset),
                       min(im_w, max(boxes[2]) + offset), min(im_h, max(boxes[3]) + offset)]
                cropped = [m[box[1]:box[3], box[0]:box[2]] for m in [im] + masks]
                im_h, im_w = cropped[0].shape[:2]
                stats[1] += im_h * im_w
                dest = [m.replace(self.root, cropped_dir) for m in img_masks]
                list(map(lambda x: cv2.imwrite(x[0], x[1]), zip(dest, cropped)))
            print("Cropping completed, dataset pixels reduced by", 100*stats[1]/stats[0], "%.")
        self.root = cropped_dir

    def is_valid_patch(self, patch_map):
        is_valid = False
        for mask in self.load_masks_from_patch_map(patch_map):
            # even a single mask containing an object makes a patch valid
            is_valid = is_valid or np.unique(self.clean_mask(mask)).size > 1
        return is_valid

    def get_rotated_img(self, img_path, im_arr, rotation):
        im_arr_rotated, path_to_save = super().get_rotated_img(img_path, im_arr, rotation)
        if rotation != 0:
            mask_file = self.get_img_mask_fname(img_path)
            rotated_mask_file = self.get_img_mask_fname(path_to_save)
            for masks_dir in self.masks_dirs:
                rotated_mask_file_path = os.path.join(self.rotated_img_dir, masks_dir, rotated_mask_file)
                if not os.path.exists(rotated_mask_file_path):
                    mask_arr = self.load_img_from_disk(os.path.join(self.root, masks_dir, mask_file))
                    rotated_mask_arr = ndimage.rotate(mask_arr, rotation, reshape=True, mode=self.rotation_padding)
                    rotated_mask_arr = self.clean_mask(self.maybe_resize(rotated_mask_arr))
                    cv2.imwrite(rotated_mask_file_path, rotated_mask_arr)

        return im_arr_rotated, path_to_save

    def load_masks_from_patch_map(self, patch_map, cache=False):
        mask_file = self.get_mask_fname(patch_map)
        masks = []
        for mask_dir in self.masks_dirs:
            mask_path = os.path.join(self.patches_dir, mask_dir, mask_file)
            if os.path.exists(mask_path):
                masks.append(self.load_img_from_disk(mask_path))
            else:
                mask_dir_path = os.path.join(self.rotated_img_dir, mask_dir) if patch_map['rotation'] != 0 \
                    else os.path.join(self.root, mask_dir)
                mask = self.load_img_from_disk(
                    os.path.join(mask_dir_path, self.get_img_mask_fname(patch_map['img_path'])))
                patch_mask = self.get_patch_from_idx(mask, patch_map['idx_h'], patch_map['idx_w'])
                if cache:
                    cv2.imwrite(mask_path, patch_mask)
                masks.append(patch_mask)
        return masks

    def save_patches_to_disk(self):
        print("Storing all patches on disk ...")
        for patch_map in tqdm(self.patches):
            _ = self.load_patch_from_patch_map(patch_map, cache=True)
            _ = self.load_masks_from_patch_map(patch_map, cache=True)

    def clean_mask(self, mask):
        """Removes all objects smaller than minimum size"""
        nb_obj, obj_labels = cv2.connectedComponents(mask)
        if nb_obj < 2:
            return mask  # only background
        obj_ids, inverse, sizes = np.unique(obj_labels, return_inverse=True, return_counts=True)
        for i, size in enumerate(sizes):
            if size < self.min_object_px_size:
                obj_ids[i] = 0  # set this component to background
        else:
            return np.reshape(obj_ids[inverse], np.shape(mask))

    def get_mask_fname(self, patch_map):
        return self.get_img_mask_fname(patch_map['patch_path'])

    def get_img_mask_fname(self, img_path):
        return PatchSamplerDataset.get_fname_no_ext(img_path) + self.mask_file_ext

    @staticmethod
    def process_mask(mask):
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        if (len(obj_ids) < 2): return None, None, None  # no object, only background
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = [ObjDetecPatchSamplerDataset.get_bounding_box_of_true_values(masks[i]) for i in range(num_objs)]
        boxes = list(filter(None.__ne__, boxes))
        return np.ones(num_objs, dtype=np.int), boxes, masks

    @staticmethod
    def get_bounding_box_of_true_values(mask_with_condition):
        pos = np.where(mask_with_condition)
        if pos[0].size == 0:    # no objects
            return None
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]
