import os
import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import torch
import pickle

from PatchSamplerDataset import PatchSamplerDataset


class ObjDetecPatchSamplerDataset(PatchSamplerDataset):

    def __init__(self, root, patch_size, mask_file_ext='.png', min_object_px_size=30, dilate_small=False, quantiles=None,
                 metrics_names=None, split_data=True, dest=None, filter_obj_size="all", **kwargs):
        if filter_obj_size not in ["small", "medium", "large", "all"]:
            raise Exception("filter_by_size argument should be either small, medium, large or all")
        else:
            self.filter_obj_size = filter_obj_size
        self.root = self.validate_path(root)
        self.root_img_dir = 'images'
        self.all_obj_cat_key = '__ALL__'
        self.patch_size = patch_size
        self.mask_file_ext = mask_file_ext
        self.min_object_px_size = min_object_px_size
        self.dilate_small = dilate_small
        self.quantiles = [0.25, 0.5, 0.75, 0.99] if quantiles is None else quantiles
        self.metrics_names = ['bbox_areas', 'segm_areas', 'obj_count'] if metrics_names is None else metrics_names
        self.split_data = split_data
        self.dest = self.get_default_cache_root_dir() if dest is None else dest

        self.masks_dirs = [m for m in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, m))
                           and m.startswith('masks_')]
        self.crop_dataset()
        super().__init__(self.root, self.patch_size, split_data=split_data, root_img_dir=self.root_img_dir,
                         dest=self.dest, **kwargs)
        if len(self.get_patch_list()) == 0:
            self.patches = self.prepare_patches_from_imgs(self.get_img_dir(self.root))
            self.save_patches_to_disk()
        self.compute_coco_evaluation_params()
        print(self.coco_metrics)

    def __getitem__(self, idx):
        img, patch_map = super().__getitem__(idx)
        gt = self.process_raw_masks(self.load_masks_from_patch_map(patch_map))
        if gt is None:
            raise Exception(f"Error with image {idx} all masks are empty. Patch map: {patch_map}")
        if self.filter_obj_size != "all":
            gt = self.filter_obj_by_size(gt)
        classes, obj_ids, segm_areas, boxes, bbox_areas, obj_masks, crowd_objs = (gt[k] for k in ['classes', 'obj_ids',
                                                        'segm_areas', 'boxes', 'bbox_areas', 'obj_masks', 'iscrowd'])
        if self.dilate_small:
            self.dilate_small_objs(classes, segm_areas, boxes, bbox_areas, obj_masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(bbox_areas, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        #TODO: masks are in uint16, torch has torch.int16 but model produces error: Buffer dtype mismatch, expected 'uint8_t' but got 'short'
        masks = torch.as_tensor(obj_masks, dtype=torch.uint8)
        # instances with iscrowd=True are ignored during evaluation which decreases both precision and recall
        # thus consider no objects as crowd
        crowd_objs = np.zeros(crowd_objs.shape)
        iscrowd = torch.as_tensor(crowd_objs.astype(np.int), dtype=torch.int64)
        image_id = torch.tensor([idx])
        gt = (boxes, labels, masks, image_id, areas, iscrowd)

        target = dict(zip(('boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'), gt))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def filter_obj_by_size(self, gt):
        if self.filter_obj_size == "all":
            return gt
        classes, obj_ids, segm_areas, boxes, bbox_areas, obj_masks = (gt[k] for k in['classes', 'obj_ids', 'segm_areas',
                                                                                     'boxes', 'bbox_areas', 'obj_masks'])
        sizes = ['all', 'small', 'medium', 'large']
        coco = {m: list(zip(sizes, self.get_coco_params(key=m)[2])) for m in self.masks_dirs}
        target_area = {s: {m: tuple(*[r[1] for r in coco[m] if r[0] == s]) for m in self.masks_dirs} for s in sizes}
        rm_index = []
        largest = (0,0)
        # iterate over all obj areas
        for i, area in enumerate(segm_areas):
            if largest[1] < area:
                largest = (i, area)
            low, high = target_area[self.filter_obj_size][self.masks_dirs[classes[i] - 1]]
            if not(low <= area <= high):
                rm_index.append(i)
        #TODO: improve, dilates largest obj to prevent patch becoming empty without object and causing errors
        if len(rm_index) == obj_ids.size:
            i = largest[0]
            min_size = target_area[self.filter_obj_size][self.masks_dirs[classes[i] - 1]][0]
            obj_masks[i], segm_areas[i], boxes[i], bbox_areas[i] = ObjDetecPatchSamplerDataset.dilate_obj(obj_masks[i],
                                                                                                          min_size)
            rm_index.remove(largest[0])
            for k in gt.keys():
                gt[k] = np.delete(gt[k], rm_index, axis=0)
        return gt


    def dilate_small_objs(self, classes, segm_areas, boxes, bbox_areas, obj_masks):
        target_area = {m: self.coco_metrics[m]['segm_areas'][1] for m in self.masks_dirs}
        for i, obj_mask in enumerate(obj_masks):
            dilated = ObjDetecPatchSamplerDataset.dilate_obj(obj_mask, target_area[self.masks_dirs[classes[i]-1]])
            obj_masks[i], segm_areas[i], boxes[i], bbox_areas[i] = dilated

    def get_patch_image_root(self):
        return os.path.join(self.patches_dir, self.root_img_dir)

    def save_as_COCO_json(self, path):
        from pycocotools import mask as coco_mask
        import json
        dataset = {'info': {'year': 2020, 'description': 'PPP data'},
                   'licenses': [{'id': 1, 'name': 'confidential, cannot be shared'}],
                   'images': [], 'categories': [], 'annotations': []}
        categories = set()
        ann_id = 0
        for img_idx in range(len(self)):
            img, targets = self[img_idx]
            image_id = targets["image_id"].item()
            dataset['images'].append({
                'id': image_id,
                'file_name': os.path.basename(self.get_patch_list()[img_idx]['patch_path']),
                'license': dataset['licenses'][0]['id'],
                'height': img.shape[-2],
                'width': img.shape[-3]
            })
            bboxes = targets["boxes"]
            bboxes[:, 2:] -= bboxes[:, :2]
            bboxes = bboxes.tolist()
            labels = targets['labels'].tolist()
            areas = targets['area'].tolist()
            iscrowd = targets['iscrowd'].tolist()
            if 'masks' in targets:
                masks = targets['masks']
                # make masks Fortran contiguous for coco_mask
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            num_objs = len(bboxes)
            for i in range(num_objs):
                rle = coco_mask.encode(masks[i].numpy())
                rle['counts'] = str(rle['counts'], 'utf-8')
                categories.add(labels[i])
                dataset['annotations'].append({
                    'image_id': image_id,
                    'bbox': bboxes[i],
                    'category_id': labels[i],
                    'area': areas[i],
                    'iscrowd': iscrowd[i],
                    'id': ann_id,
                    'segmentation': rle
                })
                ann_id += 1
        classes = [v.replace('masks_', '').upper() for v in self.masks_dirs]
        dataset['categories'] = [{'id': i, 'name': classes[i-1], 'supercategory': classes[i-1]}
                                 for i in sorted(categories)]
        with open(path, 'w') as json_file:
            json.dump(dataset, json_file)

    def process_raw_masks(self, raw_masks):
        objs_in_masks = [ObjDetecPatchSamplerDataset.extract_mask_objs(raw_mask) for raw_mask in raw_masks]
        objs = ObjDetecPatchSamplerDataset.merge_all_masks_objs(objs_in_masks)
        if not objs:
            return None
        class_metrics = [self.coco_metrics[x] for x in np.array(self.masks_dirs)[objs['classes'] - 1]]
        objs['iscrowd'] = objs['segm_areas'] > np.array([x['segm_areas'][2] for x in class_metrics])
        return objs
        #return classes, obj_ids, segm_areas, boxes, bbox_areas, obj_masks, crowd_objs

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
                rotated_mask_arr = ObjDetecPatchSamplerDataset.clean_mask(mask_arr, self.min_object_px_size) \
                    if rotation == 0 else ndimage.rotate(mask_arr, rotation, reshape=True, mode=self.rotation_padding)
                rotated_mask_arr = ObjDetecPatchSamplerDataset.clean_mask(self.maybe_resize(rotated_mask_arr),
                                                                          self.min_object_px_size)
                cv2.imwrite(rotated_mask_file_path, rotated_mask_arr)
        return im_arr_rotated, path_to_save

    def get_obj_cats_including_all_class_key(self):
        return [self.all_obj_cat_key] + [v.replace('masks_', '').upper() for v in self.masks_dirs]

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
                patch_mask = self.clean_mask(self.get_patch_from_idx(full_mask, patch_map['idx_h'], patch_map['idx_w']),
                                             self.min_object_px_size)
                if cache:
                    cv2.imwrite(mask_path, patch_mask)
                masks.append(patch_mask)
        return masks

    def save_patches_to_disk(self):
        print("Saving all patches on disk ...")
        self.save_patch_maps_to_disk()
        for patch_map in tqdm(self.get_all_patches()):
            _ = self.load_patch_from_patch_map(patch_map, cache=True)
            _ = self.load_masks_from_patch_map(patch_map, cache=True)

    def get_mask_fname(self, path):
        return os.path.splitext(os.path.basename(path))[0] + self.mask_file_ext

    def get_coco_params(self, key=None, f=1.2):
        if key is None:
            key = self.all_obj_cat_key
        classes = [i + 1 for i in range(len(self.masks_dirs))] if key == self.all_obj_cat_key \
            else [self.masks_dirs.index(key) + 1]
        m = self.coco_metrics[key]
        n ='bbox_areas'
        bbox_areas = [[0, int(f * m[n][3])], [0, int(m[n][0])], [int(m[n][0]), int(m[n][2])],
                      [int(m[n][2]), int(f * m[n][3])]]   # ['all', 'small', 'medium', 'large']
        n = 'segm_areas'
        segm_areas = [[0, int(f * m[n][3])], [0, int(m[n][0])], [int(m[n][0]), int(m[n][2])],
                      [int(m[n][2]), int(f * m[n][3])]]  # ['all', 'small', 'medium', 'large']
        n = 'obj_count'
        max_detections = [int(m[n][1]), int(m[n][2]), int(f * m[n][3])]

        return classes, bbox_areas, segm_areas, max_detections

    def compute_coco_evaluation_params(self):
        metrics_path = os.path.join(self.patches_dir, 'coco-metrics.p')
        if not os.path.exists(metrics_path):
            print("Computing coco evaluation parameters")
            # area: all small medium large
            # nb of possible object detected: maxDets
            raw_metrics = []
            for patch_map in tqdm(self.get_all_patches()):
                raw_metrics.append(list(map(ObjDetecPatchSamplerDataset.get_mask_metrics,
                                        self.load_masks_from_patch_map(patch_map))))
            pickle.dump(raw_metrics, open(metrics_path, "wb"))
        else:
            raw_metrics = pickle.load(open(metrics_path, "rb"))
        self.coco_metrics = ObjDetecPatchSamplerDataset.analyze_mask_metrics(raw_metrics, self.all_obj_cat_key,
                                                                             self.masks_dirs, self.metrics_names,
                                                                             self.quantiles)

    @staticmethod
    def analyze_mask_metrics(raw_metrics, all_cat_key, mask_cats, metrics_names, quantiles):
        """Returns e.g. {'masks_pustules': {'bbox_areas': (88.0, 182.0, 598.5),
                                            'segm_areas': (79.5, 150.5, 379.0),
                                            'obj_count': (5.0, 11.0, 27.0)},
                        'masks_spots': {'bbox_areas': (48.0, 100.0, 287.0),
                                        'segm_areas': (45.0, 94.0, 191.0),
                                        'obj_count': (2.0, 5.5, 15.25)}}"""
        # metrics contains [nb of patches * [(bbox_areas, segm_areas, [obj_count]) * nb of mask categories]]
        # combine metrics per type for each mask category e.g. pustules and spots
        grp_per_mask = [list(zip(*filter(None.__ne__, m_metrics))) for m_metrics in zip(*raw_metrics)]
        # flatten each metrics in an array
        masks_metrics = [[[n for sublst in m_lsts for n in sublst] for m_lsts in grouped] for grouped in grp_per_mask]
        # compute quantiles
        mask_quantiles = [[tuple(np.quantile(x, quantiles)) for x in m_metrics] for m_metrics in masks_metrics]
        metrics = dict(zip(mask_cats, [dict(zip(metrics_names, mq)) for mq in mask_quantiles]))

        all_metrics = [[y for x in m for y in x] for m in zip(*masks_metrics)]
        metrics[all_cat_key] = dict(zip(metrics_names, [tuple(np.quantile(x, quantiles)) for x in all_metrics]))
        return metrics

    @staticmethod
    def merge_all_masks_objs(objs_in_masks):
        objs = {}
        for c, objs_in_masks in enumerate(objs_in_masks):
            if objs_in_masks is None:
                continue
            objs_in_masks['classes'] = (c + 1) * np.ones(objs_in_masks['obj_count'], dtype=np.int)
            del objs_in_masks['obj_count']
            if not objs:
                objs = {k: v for k, v in objs_in_masks.items()}
            else:
                for k in objs.keys():
                    objs[k] = np.concatenate([objs[k], objs_in_masks[k]])
        return objs

    @staticmethod
    def extract_mask_objs(mask):
        """Returns obj_count, obj_ids, obj_segm_areas, obj_bbox, obj_bbox_areas, obj_masks (without backgroud)"""
        obj_ids, segm_areas = np.unique(mask, return_counts=True)
        # remove background
        obj_ids = obj_ids[1:]
        segm_areas = segm_areas[1:]
        obj_count = len(obj_ids)
        if obj_count < 1:
            return None
        obj_masks = mask == obj_ids[:, None, None]
        boxes = [ObjDetecPatchSamplerDataset.get_bbox_of_true_values(obj_masks[i]) for i in range(obj_count)]
        obj_del = [i for i, v in enumerate(boxes) if v is None]
        obj_count -= len(obj_del)
        if obj_count < 1:
            return None
        obj_ids, segm_areas, obj_masks = (np.delete(i, obj_del, axis=0) for i in [obj_ids, segm_areas, obj_masks])
        boxes = np.reshape(list(filter(None.__ne__, boxes)), (obj_count, 4))
        bbox_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = np.zeros(obj_ids.shape)
        return {'obj_count': obj_count, 'obj_ids': obj_ids, 'segm_areas': segm_areas, 'boxes': boxes,
                'bbox_areas': bbox_areas, 'obj_masks': obj_masks, 'iscrowd': iscrowd}

    @staticmethod
    def get_mask_metrics(mask):
        """ Returns a tuple with three lists: (bbox_areas, segm_areas, [obj_count]), obj_count is a single number"""
        objs = ObjDetecPatchSamplerDataset.extract_mask_objs(mask)
        if objs is None:
            return None
        obj_count, segm_areas, bbox_areas = (objs[k] for k in ['obj_count', 'segm_areas', 'bbox_areas'])
        return bbox_areas.tolist(), segm_areas.tolist(), [obj_count]

    @staticmethod
    def rm_small_objs_and_sep_instance(mask, min_object_px_size, check_bbox=False):
        """Removes all objects smaller than minimum size and separate obj instance by giving them different id"""
        mask[mask > 0] = 1
        nb_obj, obj_labels = cv2.connectedComponents(mask.astype(np.uint8))
        if nb_obj < 2:
            return mask  # only background
        obj_ids, inverse, sizes = np.unique(obj_labels, return_inverse=True, return_counts=True)
        for i, size in enumerate(sizes):
            if size < min_object_px_size:
                obj_ids[i] = 0  # set this component to background

        mask_cleaned = np.reshape(obj_ids[inverse], np.shape(mask)).astype(np.uint16)
        if not check_bbox:
            return mask_cleaned
        # Now remove object with None bbox
        for i, obj_id in enumerate(obj_ids):
            if i == 0:  # skip background
                continue
            if ObjDetecPatchSamplerDataset.get_bbox_of_true_values(mask_cleaned == obj_id) is None:
                obj_ids[i] = 0  # set this component to background
        return np.reshape(obj_ids[inverse], np.shape(mask)).astype(np.uint16)

    @staticmethod
    def clean_mask(mask, min_object_px_size, kernel_size=(3, 3)):
        mask_cleaned = ObjDetecPatchSamplerDataset.rm_small_objs_and_sep_instance(mask, min_object_px_size)
        # if you don't rm small elements before, you risk to enhance noise in the mask_cleaned
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, np.ones(kernel_size, np.uint16))
        mask_cleaned = ObjDetecPatchSamplerDataset.rm_small_objs_and_sep_instance(mask_cleaned, min_object_px_size,
                                                                                  check_bbox=True)
        return mask_cleaned

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
        if xmin == xmax or ymin == ymax:    # Faulty object
            return None
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def dilate_obj(obj_mask, target_area):
        kernel = np.ones((3, 3), np.uint8)
        segm_areas = np.count_nonzero(obj_mask)
        while segm_areas < target_area:
            obj_mask = cv2.dilate(obj_mask.astype(np.uint8), kernel, iterations=1)
            segm_areas = np.count_nonzero(obj_mask)
        obj_masks = obj_mask > 0
        box = np.array(ObjDetecPatchSamplerDataset.get_bbox_of_true_values(obj_masks))
        box_area = (box[3] - box[1]) * (box[2] - box[0])
        return obj_mask, segm_areas, box, box_area
