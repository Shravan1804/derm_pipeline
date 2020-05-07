import os
import copy
import json
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPoint

import common
from PatchExtractor import PatchExtractor


def main(args):
    with open(args.labels, 'r') as f:
        labels = json.load(f)
    id_to_patch_name = {img['id']: img['file_name'] for img in labels['images']}
    patch_id_to_full_img = {k: PatchExtractor.get_img_fname_from_patch_fname(v) for k, v in id_to_patch_name.items()}
    full_img_id = {k: i for i, k in enumerate(sorted(set(patch_id_to_full_img.values())))}

    print("CREATING NEW LIST OF FULL IMAGES")
    new_images = []
    for full_img in full_img_id.keys():
        img_path = os.path.join(args.full_imgs, full_img)
        if not os.path.exists(img_path):
            print("Warning", img_path, "does not exist...")
            height, width = -1, -1
        else:
            im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            height, width = im.shape[:2]
        new_images.append({'width': width, 'license': 0, 'file_name': full_img, 'id': full_img_id[full_img],
                           'height': height})

    print("ADJUSTING ANNOTATIONS POLYGONS TO FULL IMAGES")
    full_img_annos = {k: [] for k in full_img_id.keys()}
    for lanno in labels['annotations']:
        for seg in lanno['segmentation']:
            anno = copy.deepcopy(lanno)
            anno['segmentation'] = [seg]
            full_img = patch_id_to_full_img[anno['image_id']]
            y, x = PatchExtractor.get_pos_from_patch_name(id_to_patch_name[anno['image_id']])
            anno['image_id'] = full_img_id[full_img]
            anno['bbox'] = (np.array(anno['bbox']) + np.ones(4) * np.array([x, y, x, y])).tolist()
            n = len(anno['segmentation'][0])
            anno['segmentation'] = (np.array(anno['segmentation']) + np.ones((1, n)) * np.array([[x, y] * int(n/2)])).tolist()
            full_img_annos[full_img].append(anno)

    print("MERGING OVERLAPPING POLYGONS")
    new_annotations = []
    anno_id = 0
    for full_img, f_annos in tqdm(full_img_annos.items()):
        annos_per_cats = {c['id']: [a for a in f_annos if a['category_id'] == c['id']] for c in labels['categories']}
        for annos in annos_per_cats.values():
            polys = [Polygon([c[n:n+2] for n in range(0, len(c), 2)]) for c in [a['segmentation'][0] for a in annos]]
            for i, poly in enumerate(polys):
                if not poly.is_valid:
                    polys[i] = poly.buffer(0)
                    annos[i] = poly_to_anno(-1, polys[i], annos[i], recompute=True)
            polys_annos = list(zip(polys, annos))
            merge = True
            while merge:
                for a in polys_annos:
                    merge = False
                    for b in polys_annos:
                        if a != b and a[0].intersects(b[0]):
                            merge = True
                            p1_a1, p2_a2 = a, b
                            break
                    if merge:
                        break
                if merge:
                    polys_annos.remove(p1_a1)
                    polys_annos.remove(p2_a2)
                    p3 = unary_union([p1_a1[0], p2_a2[0]])
                    polys_annos.append((p3, poly_to_anno(-1, p3, p1_a1[1])))
            for poly, anno in polys_annos:
                new_annotations.append(poly_to_anno(anno_id, poly, anno))
                anno_id += 1

    labels['images'] = new_images
    labels['annotations'] = new_annotations
    json_path = os.path.join(args.full_imgs, 'full_img_coco.json')
    print("SAVING", json_path)
    json.dump(labels, open(json_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


def poly_to_anno(new_id, poly, old_anno, recompute=False):
    new_anno = copy.deepcopy(old_anno)
    if new_anno['id'] == -1 or recompute:    # merged polygon, recompute segmentation, bbox, area
        # last two poly coords are starting point
        new_anno['segmentation'] = np.array(poly.exterior.coords).ravel()[:-2].reshape((1, -1)).tolist()
        bbox = list(MultiPoint(poly.exterior.coords).bounds)    # [xmin, ymin, xmax, ymax]
        new_anno['bbox'] = [*bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1]]    # [x, y, width, height]
        new_anno['area'] = poly.area
    new_anno['id'] = new_id
    return new_anno


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts patch coco labels to full img labels")
    parser.add_argument('--full-imgs', type=str, required=True, help="Path of dir containing full imgs")
    parser.add_argument('--labels', type=str, help="JSON file containing patch labels in coco format")
    parser.add_argument('-p', '--patch-size', default=512, type=int, help="args.labels patch size")
    args = parser.parse_args()

    common.check_dir_valid(args.full_imgs)
    common.check_file_valid(args.labels)

    common.time_method(main, args)

