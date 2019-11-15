import os
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
from tqdm import tqdm
import torch
import pickle

from PatchSamplerDataset import PatchSamplerDataset


class ObjDetecPatchSamplerDataset(PatchSamplerDataset):

    def __init__(self, root, patch_size, mask_file_ext='.png', min_object_px_size=30, quantiles=[0.25, 0.5, 0.75],
                 metrics_names=['bbox_areas', 'segm_areas', 'obj_count'], split_data=True, dest=None, **kwargs):
        self.root = root
        self.root_img_dir = 'images'
        self.patch_size = patch_size
        self.mask_file_ext = mask_file_ext
        self.min_object_px_size = min_object_px_size
        self.quantiles = quantiles
        self.metrics_names = metrics_names
        self.split_data = split_data
        self.dest = self.get_default_cache_root_dir() if dest is None else dest

        self.masks_dirs = [m for m in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, m))
                           and m.startswith('masks_')]
        self.crop_dataset()
        super().__init__(self.root, self.patch_size, root_img_dir=self.root_img_dir, dest=self.dest, **kwargs)
        if len(self.get_patch_list()) == 0:
            patches = self.prepare_patches_from_imgs(self.get_img_dir(self.root))
            if self.split_data:
                self.train_patches, self.test_patches = patches
            else:
                self.get_patch_list().extend(patches[0])
            self.save_patches_to_disk()
        self.compute_coco_evaluation_params()

    def __getitem__(self, idx):
        patch_map = self.get_patch_list()[idx]
        img = Image.fromarray(cv2.cvtColor(self.load_patch_from_patch_map(patch_map), cv2.COLOR_BGR2RGB)).convert("RGB")
        raw_masks = self.load_masks_from_patch_map(patch_map)

        classes = masks = segm_areas = None
        boxes = []
        for i, objects in enumerate(filter(None.__ne__, map(ObjDetecPatchSamplerDataset.process_mask, raw_masks))):
            c, b, m, s = objects
            if b is None: continue  # empty mask
            # all objects from one mask belong to the same class
            classes = c if classes is None else np.append(classes, (i + 1) * c)
            boxes += b
            segm_areas = s if segm_areas is None else np.append(segm_areas, s)
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
        # An instance iscrowd if it has a larger segmentation area than 75% of all objects
        crowd_objs = segm_areas > np.array([x['segm_areas'][2] for x in map(self.metrics.__getitem__,
                                                                            np.array(self.masks_dirs)[classes-1])])
        iscrowd = torch.as_tensor(crowd_objs.astype(np.int), dtype=torch.int64)

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
        cropped_dir = os.path.join(self.dest, 'cache_' + self.get_dataset_name() + '-cropped-p' + str(self.patch_size))
        if not os.path.exists(cropped_dir):
            list(map(os.makedirs, [os.path.join(cropped_dir, n) for n in self.masks_dirs + [self.root_img_dir]]))

            print("Cropping dataset ...")
            offset = int(self.patch_size / 2) + 1  # + 1 to comprise upper bound
            stats = [0, 0]
            data_dirs = [self.get_img_dir(self.root)] + [os.path.join(self.root, masks_dir)
                                                                        for masks_dir in self.masks_dirs]
            data_files = [sorted(os.listdir(p)) for p in data_dirs]
            imgs_masks = [[os.path.join(x[0], n) for n in x[1]] for x in zip(data_dirs, data_files)]
            # iterate over (img, mask1, mask2, ...) tuples
            for img_masks in tqdm(list(zip(*imgs_masks))):
                im = self.load_img_from_disk(img_masks[0])
                im_h, im_w = im.shape[:2]
                stats[0] += im_h * im_w
                masks = list(map(self.load_img_from_disk, img_masks[1:]))
                boxes = map(ObjDetecPatchSamplerDataset.get_bbox_of_true_values, [x > 0 for x in masks])
                boxes = list(zip(*list(filter(None.__ne__, boxes))))
                box = [max(0, min(boxes[0]) - offset), max(0, min(boxes[1]) - offset),
                       min(im_w, max(boxes[2]) + offset), min(im_h, max(boxes[3]) + offset)]
                cropped = [m[box[1]:box[3], box[0]:box[2]] for m in [im] + masks]
                im_h, im_w = cropped[0].shape[:2]
                stats[1] += im_h * im_w
                dest = [m.replace(self.root, cropped_dir) for m in img_masks]
                [cv2.imwrite(x[0], x[1]) for x in zip(dest, cropped)]
            print("Cropping completed, dataset pixels reduced by", 100 * (1 - stats[1] / stats[0]), "%.")
        self.root = cropped_dir

    def is_valid_patch(self, patch_map):
        is_valid = False
        for mask in self.load_masks_from_patch_map(patch_map):
            # even a single mask containing an object makes a patch valid
            is_valid = is_valid or np.unique(mask).size > 1
        return is_valid

    def get_rotated_img(self, img_path, im_arr, rotation):
        im_arr_rotated, path_to_save = super().get_rotated_img(img_path, im_arr, rotation)
        for masks_dir in self.masks_dirs:
            rotated_mask_file_path = os.path.join(self.rotated_img_dir, masks_dir, self.get_mask_fname(path_to_save))
            if not os.path.exists(rotated_mask_file_path):
                mask_arr = self.load_img_from_disk(os.path.join(self.root, masks_dir, self.get_mask_fname(img_path)))
                rotated_mask_arr = mask_arr if rotation == 0 else \
                    ndimage.rotate(mask_arr, rotation, reshape=True, mode=self.rotation_padding)
                rotated_mask_arr = self.clean_mask(self.maybe_resize(rotated_mask_arr))
                cv2.imwrite(rotated_mask_file_path, rotated_mask_arr)
        return im_arr_rotated, path_to_save

    def load_masks_from_patch_map(self, patch_map, cache=False):
        mask_file = self.get_mask_fname(patch_map['patch_path'])
        masks = []
        for mask_dir in self.masks_dirs:
            mask_path = os.path.join(self.patches_dir, mask_dir, mask_file)
            if os.path.exists(mask_path):
                masks.append(self.load_img_from_disk(mask_path))
            else:
                full_mask_file = mask_file.replace(PatchSamplerDataset.get_patch_suffix(patch_map['idx_h'],
                                                                                        patch_map['idx_w']), '')
                full_mask = self.load_img_from_disk(os.path.join(self.rotated_img_dir, mask_dir, full_mask_file))
                patch_mask = self.get_patch_from_idx(full_mask, patch_map['idx_h'], patch_map['idx_w'])
                if cache:
                    cv2.imwrite(mask_path, patch_mask)
                masks.append(patch_mask)
        return masks

    def save_patches_to_disk(self):
        print("Saving all patches on disk ...")
        self.save_patch_maps_to_disk()
        for patch_map in tqdm(self.train_patches + self.test_patches):
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

    def get_mask_fname(self, path):
        return os.path.splitext(os.path.basename(path))[0] + self.mask_file_ext

    def compute_coco_evaluation_params(self):
        metrics_path = os.path.join(self.patches_dir, 'coco-metrics.p')
        print(metrics_path)
        if not os.path.exists(metrics_path):
            print("Computing coco evaluation parameters")
            # area: all small medium large
            # nb of possible object detected: maxDets
            raw_metrics = []
            for patch_map in tqdm(self.train_patches + self.test_patches):
                raw_metrics.append(list(map(ObjDetecPatchSamplerDataset.get_mask_metrics,
                                        self.load_masks_from_patch_map(patch_map))))
            pickle.dump(raw_metrics, open(metrics_path, "wb"))
        else:
            raw_metrics = pickle.load(open(metrics_path, "rb"))
        self.metrics = ObjDetecPatchSamplerDataset.analyze_mask_metrics(raw_metrics, self.masks_dirs,
                                                                        self.metrics_names, self.quantiles)

    @staticmethod
    def analyze_mask_metrics(raw_metrics, mask_cats, metrics_names, quantiles):
        """Returns e.g. {'masks_pustules': {'bbox_areas': (array([143. , 323.5, 907. ]),
        array([129.  , 268.  , 646.75]), array([ 1.75, 16.  , 27.25]))},
                        'masks_spots': {'bbox_areas': (array([ 99. , 176. , 418.5]),
                        array([ 97.  , 139.  , 314.25]), array([ 2. , 16. , 25.5]))}}"""
        # metrics contains [nb of patches * [(bbox_areas, segm_areas, [obj_count]) * nb of mask categories]]
        # combine metrics per type for each mask category e.g. pustules and spots
        grp_per_mask = [zip(*m_metrics) for m_metrics in zip(*raw_metrics)]
        # flatten each metrics in an array
        masks_metrics = [[[n for sublst in m_lsts for n in sublst] for m_lsts in grouped] for grouped in grp_per_mask]
        # compute quantiles
        mask_quantiles = [[tuple(np.quantile(x, quantiles)) for x in m_metrics] for m_metrics in masks_metrics]
        return dict(zip(mask_cats, [dict(zip(metrics_names, mq)) for mq in mask_quantiles]))

    @staticmethod
    def get_mask_metrics(mask):
        """ Returns a tuple with three lists: (bbox_areas, segm_areas, [obj_count]), obj_count is a single number"""
        obj_ids, sizes = np.unique(mask, return_counts=True)
        if len(obj_ids) < 2:
            return [], [], 0
        # ignore background
        obj_ids = obj_ids[1:]
        obj_count = len(obj_ids)
        segm_areas = sizes[1:]
        masks = mask == obj_ids[:, None, None]
        boxes = [ObjDetecPatchSamplerDataset.get_bbox_of_true_values(masks[i]) for i in range(obj_count)]
        bbox_areas = ObjDetecPatchSamplerDataset.get_bbox_areas(list(filter(None.__ne__, boxes)))
        return bbox_areas, segm_areas, [obj_count]

    @staticmethod
    def process_mask(mask):
        # instances are encoded as different colors
        obj_ids, sizes = np.unique(mask, return_counts=True)
        if len(obj_ids) < 2: return None  # no object, only background
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        segm_areas = sizes[1:]
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = [ObjDetecPatchSamplerDataset.get_bbox_of_true_values(masks[i]) for i in range(num_objs)]
        return np.ones(num_objs, dtype=np.int), list(filter(None.__ne__, boxes)), masks, segm_areas

    @staticmethod
    def get_bbox_areas(bbox_coord):
        """Computes bbox area, input is list of bbox coordinates i.e. a list of list"""
        bbox_coord = np.reshape(bbox_coord, (len(bbox_coord), 4))
        return list((bbox_coord[:, 3] - bbox_coord[:, 1]) * (bbox_coord[:, 2] - bbox_coord[:, 0]))

    @staticmethod
    def get_bbox_of_true_values(mask_with_condition):
        """Returns bbox coordinates: [xmin, ymin, xmax, ymax]. None if no objects"""
        pos = np.where(mask_with_condition)
        if pos[0].size == 0:  # no objects
            return None
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

