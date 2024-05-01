import os
import json
import argparse
import multiprocessing as mp
from functools import partial

from ..general import crypto, concurrency, common_img as cimg, common
from ..segmentation import mask_utils
from ..segmentation import segmentation_utils as segm_utils
from ..object_detection import coco_format


def get_img_shape(impath, crypto_key):
    """Get image shape from (maybe encrypted) image
    :param impath: str, image path
    :param crypto_key: Fernet key used to decrypt image
    :return: tuple (h,w)
    """
    if crypto_key is None: return cimg.quick_img_size(impath)
    else: return crypto.decrypt_img(impath, crypto_key).shape[:2]


def convert_segm_mask_to_obj_det_cats(mask, args):
    """Converts segmentation mask category codes to object detection category codes
    :param mask: array, mask
    :param args: command line arguments
    :return: array, modified mask
    """
    for idx, segm_cat in enumerate(args.segm_cats):
        if idx == args.bg: continue
        elif segm_cat not in args.od_cats: mask[mask == idx] = args.bg
        else: mask[mask == idx] = args.od_cats.index(segm_cat) + 1  # od cats idx start at 1, assumes same cat order
    return mask


def extract_annos(proc_id, q_annos, im_mask, args):
    """Function called by processes to extract annotations from the segmentation mask
    Places the annotations in the multiprocess queue
    :param proc_id: int, process id
    :param q_annos: multiprocess queue, where the annotations will be stored
    :param im_mask: tuple of image and mask paths
    :param args: command line arguments
    """
    for i, (im_id, impath, mpath) in enumerate(im_mask):
        img_dict = coco_format.get_img_record(im_id, impath, im_shape=get_img_shape(impath, args.ckey))
        mask = cimg.load_img(mpath) if args.ckey is None else crypto.decrypt_img(mpath, args.ckey)
        obj_cats_with_masks = mask_utils.separate_objs_in_mask(convert_segm_mask_to_obj_det_cats(mask, args))
        if obj_cats_with_masks is None: continue
        else: obj_cats, obj_cats_masks = obj_cats_with_masks
        _, img_annos = coco_format.get_annos_from_objs_mask(im_id, 0, obj_cats, obj_cats_masks, to_poly=args.polygon)
        q_annos.put([(img_dict, img_annos)])
        if i % max(15, (10*(len(im_mask)//30))) == 0:     # prints status at every third of whole task
            print(f"Process {proc_id}: processed {i}/{len(im_mask)} images")


def to_coco_format(img_dict_with_annos, args):
    """Converts extracted annotations to coco json format
    :param img_dict_with_annos: list of tuples with the image dict and image annotations
    :param args: command line arguments
    :return: dict, dataset in coco json format
    """
    dataset = coco_format.get_default_dataset(os.path.basename(args.data))
    cids = set()
    ann_id = 1  # MUST start at 1 since pycocotools.cocoeval uses detId to track matches and checks with > 0
    for img_dict, annotations in img_dict_with_annos:
        dataset['images'].append(img_dict)
        for img_ann_id in range(len(annotations)):
            annotations[img_ann_id]['image_id'] = img_dict['id']
            annotations[img_ann_id]['id'] = ann_id
            ann_id += 1
            cids.add(annotations[img_ann_id]['category_id'])
        dataset['annotations'].extend(annotations)
    dataset['categories'] = [{'id': i, 'name': c, 'supercategory': c} for i, c in zip(sorted(cids), args.od_cats)]
    return dataset


def main(args):
    """Performs the multiprocess segmentation to coco dataset conversion
    :param args: command line arguments
    """
    get_mask_path = partial(segm_utils.get_mask_path, img_dir=args.img_dir, mask_dir=args.mask_dir, mext=args.mext)
    impaths = common.list_files(os.path.join(args.data, args.img_dir), full_path=True)
    img_with_masks = [(im_id+1, impath, get_mask_path(impath)) for im_id, impath in enumerate(impaths)]
    workers, _, batches = concurrency.batch_lst(img_with_masks, args.bs, args.workers)
    q_annos = mp.Queue()
    jobs = []
    for i, batch in zip(range(workers), batches):
        jobs.append(mp.Process(target=extract_annos, args=(i, q_annos, batch, args)))
        jobs[i].start()
    img_dict_with_annos = concurrency.unload_mpqueue(q_annos, jobs)
    for j in jobs:
        j.join()

    if len(img_dict_with_annos) == 0: raise Exception("Error while processing images and masks")

    coco_ds = to_coco_format(sorted(img_dict_with_annos, key=lambda x: x[0]['id']), args)

    print("Converting and saving as coco json")
    json_path = os.path.join(args.data, f"{os.path.basename(args.data)}.json")
    json.dump(coco_ds, open(json_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts segm dataset to obj detec coco json")
    parser.add_argument('--data', type=str, required=True, help="dataset root directory absolute path")
    parser.add_argument('--segm-cats', type=str, required=True, nargs='+', help="Segm categories")
    parser.add_argument('--od-cats', type=str, nargs='+', help="Obj det categories")
    crypto.add_encrypted_args(parser)
    parser = segm_utils.common_segm_args(parser)
    parser.add_argument('--polygon', action='store_true', help="converts bitmask to polygon")
    parser = concurrency.add_multi_proc_args(parser)
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    if args.od_cats is None: args.od_cats = args.segm_cats[1:]  # skip bg
    else:
        for c in args.od_cats:
            assert c in args.segm_cats, f"found OD cat {c} which is not present in segm cats ({args.segm_cats})"
    if args.ckey is not None: args.ckey = crypto.request_key(None, args.ckey)

    common.time_method(main, args)

