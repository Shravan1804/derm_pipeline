import os
import cv2
import json
import argparse
import numpy as np
import multiprocessing as mp


import common
import concurrency
import coco_format
from ObjDetecPatchSamplerDataset import ObjDetecPatchSamplerDataset


def raw_mask_to_objs(mask_file, rm_size=25):
    """Extract all objs from mask_file. A mask file only contains obj of the same category.
    Removes objs smaller than rm_size pixels.
    Returns dict with obj_count, obj_ids, obj_segm_areas, obj_bbox, obj_bbox_areas, obj_masks (without backgroud)."""
    mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    if len(mask.shape) > 2:  # uncleaned HSLU mask file
        mask = mask[:, :, 0]
    mask = ObjDetecPatchSamplerDataset.rm_small_objs_and_sep_instance(mask, rm_size, check_bbox=True)
    return ObjDetecPatchSamplerDataset.extract_mask_objs(mask)


def img_objs_to_annos(img, objs_in_masks, to_polygon):
    targets = ObjDetecPatchSamplerDataset.merge_all_masks_objs(objs_in_masks)
    if not targets:
        return []
    targets = {'labels': targets['classes'], 'area': targets['bbox_areas'], 'boxes': targets['boxes'],
               'masks': targets['obj_masks'], 'iscrowd': targets['iscrowd']}
    masks = []
    for i, m in enumerate(targets['masks']):
        try:
            if to_polygon:
                masks.append(coco_format.convert_obj_mask_to_poly(m))
            else:
                m = np.array(m[:, :, None], order="F")
                masks.append(coco_format.convert_obj_mask_to_rle(m))
        except Exception as err:
            print(f"Image {img} is causing a problem with obj {i} of category {targets['labels'][i]}: {err}")
    targets['masks'] = masks
    _, annos, _ = coco_format.get_img_annotations(-1, 0, targets)
    return annos


def extract_annos(proc_id, q_annos, img_with_masks, to_poly):
    for i, img_masks in enumerate(img_with_masks):
        img, masks = img_masks
        img_dict = coco_format.get_img_record(-1, img)
        annotations = img_objs_to_annos(img, [raw_mask_to_objs(mfile) for mfile in masks], to_poly)
        q_annos.put([(os.path.basename(img), (img_dict, annotations))])
        if i % 5 == 0:
            print(f"Process {proc_id}: processed {i}/{len(img_with_masks)} images")


def to_coco_format(root, img_dir, img_annos, classes):
    dataset = coco_format.get_default_dataset(os.path.basename(root))
    cats = set()
    ann_id = 0
    for idx, img in enumerate(common.list_files(os.path.join(root, img_dir))):
        img_dict, annotations = img_annos[img]
        img_dict['id'] = idx
        dataset['images'].append(img_dict)
        for img_ann_id in range(len(annotations)):
            annotations[img_ann_id]['image_id'] = idx
            annotations[img_ann_id]['id'] = ann_id
            ann_id += 1
            cats.add(annotations[img_ann_id]['category_id'])
    dataset['categories'] = [{'id': i, 'name': classes[i - 1], 'supercategory': classes[i - 1]} for i in sorted(cats)]
    return dataset


def get_img_with_masks(root, img_dir, mask_dirs, mask_ext):
    lst = []
    for img in common.list_files(os.path.join(root, img_dir)):
        name, ext = os.path.splitext(img)
        masks = [os.path.join(root, mdir, name + mask_ext) for mdir in mask_dirs]
        lst.append((os.path.join(root, img_dir, img), masks))
    return lst


def main(args):
    mask_dirs = [d for d in common.list_dirs(args.data) if d.startswith(args.mdir_prefix)]
    img_with_masks = get_img_with_masks(args.data, args.img_dir, mask_dirs, args.mext)
    workers, batch_size, batches = concurrency.batch_lst(img_with_masks)
    q_annos = mp.Queue()
    jobs = []
    for i, files in zip(range(workers), batches):
        jobs.append(mp.Process(target=extract_annos, args=(i, q_annos, img_with_masks, args.to_polygon)))
        jobs[i].start()
    img_annos = concurrency.unload_mpqueue(q_annos, jobs)
    for j in jobs:
        j.join()
    img_annos = {item[0]: item[1] for item in img_annos}

    print("Converting and saving as coco json")
    json_path = os.path.join(args.dest, f"instances_{os.path.basename(args.data)}_coco_format.json")
    classes = [m.replace(args.mdir_prefix, '') for m in mask_dirs]
    json.dump(to_coco_format(args.data, args.img_dir, img_annos, classes), open(json_path, 'w'),
              sort_keys=True, indent=4, separators=(',', ': '))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts dataset to a patch dataset without data augmentation")
    parser.add_argument('--data', type=str, required=True, help="dataset root directory absolute path")
    parser.add_argument('--dest', type=str, help="dir where annotation json file will be saved")
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--to_polygon', action='store_true', help="converts bitmask to polygon")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")
    parser.add_argument('--mdir-prefix', type=str, default='masks_', help="prefix of mask dirs")
    common.add_multi_proc_args(parser)
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    if args.dest is None:
        args.dest = args.data
    else:
        common.check_dir_valid(args.dest)
    common.set_seeds(args.seed)

    common.time_method(main, args)

