import os
import json
import argparse
import itertools

import cv2
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPoint

import common
import PatchExtractor


def main(args):
    with open(args.labels, 'r') as f:
        labels = json.load(f)
    patch_id_to_full_img = {img['id']: PatchExtractor.get_img_fname_from_patch_fname(img['file_name'])
                            for img in labels['images']}
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
    for anno in labels['annotations']:
        if len(anno['segmentation']) > 1:
            print(img_path, "Unsupported polygon (", anno['segmentation'], ") skipping...")
            continue
        full_img = patch_id_to_full_img[anno['image_id']]
        patch_name = [im['file_name'] for im in labels['images'] if im['id'] == anno['image_id']][0]
        y, x = PatchExtractor.get_pos_from_patch_name(patch_name)
        anno['image_id'] = full_img_id[full_img]
        anno['bbox'] = (np.array(anno['bbox']) + np.ones(4) * np.array([x, y, x, y])).tolist()
        n = len(anno['segmentation'][0])
        anno['segmentation'] = (np.array(anno['segmentation']) + np.ones((1, n)) * np.array([[x, y] * n])).tolist()
        full_img_annos[full_img].append(anno)

    print("MERGING OVERLAPPING POLYGONS")
    new_annotations = []
    anno_id = 0
    for full_img, f_annos in full_img_annos.items():
        annos_per_cats = {c['id']: [a for a in f_annos if a['category_id'] == c['id']] for c in labels['categories']}
        for annos in annos_per_cats.values():
            polys = [Polygon([c[n:n+2] for c in range(0, len(c), 2)]) for c in [a['segmentation'][0] for a in annos]]
            merge = None
            while merge is None or merge:
                merge = [(a, b) for a, b in itertools.combinations(zip(polys, annos), 2) if a[0].intersects(b[0])]
                for p1_a1, p2_a2 in merge:
                    p3 = unary_union([p1_a1[0], p2_a2[0]])
                    if p1_a1[0] in polys:
                        polys.remove(p1_a1[0])
                        annos.remove(p1_a1[1])
                    if p2_a2[0] in polys:
                        polys.remove(p2_a2[0])
                        annos.remove(p2_a2[1])
                    polys.append(p3)
                    annos.append(poly_to_anno(-1, p3, p1_a1[1]))
            for poly, anno in zip(polys, annos):
                new_annotations.append(poly_to_anno(anno_id, poly, anno))
                anno_id += 1

    labels['images'] = new_images
    labels['annotations'] = new_annotations
    json_path = os.path.join(args.full_imgs, 'full_img_coco.json')
    print("SAVING", json_path)
    json.dump(labels, open(json_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


def poly_to_anno(new_id, poly, old_anno):
    if old_anno['id'] == -1:    # merged polygon, recompute segmentation, bbox, area
        old_anno['segmentation'] = np.array(poly.exterior.coords).ravel()[:-2].reshape((1, -1)).tolist()
        bbox = list(MultiPoint(poly.exterior.coords).bounds)    # [xmin, ymin, xmax, ymax]
        old_anno['bbox'] = [*bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1]]    # [x, y, width, height]
        old_anno['area'] = poly.area
    old_anno['id'] = new_id
    return old_anno


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts patch coco labels to full img labels")
    parser.add_argument('--full-imgs', type=str, required=True, help="Path of dir containing full imgs")
    parser.add_argument('--labels', type=str, help="JSON file containing patch labels in coco format")
    parser.add_argument('-p', '--patch-size', default=512, type=int, help="args.labels patch size")
    args = parser.parse_args()

    common.check_dir_valid(args.full_imgs)
    common.check_file_valid(args.labels)

    common.time_method(main, args)

