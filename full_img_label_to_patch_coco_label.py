import os
import copy
import json
import argparse

import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPoint

import common
import PatchExtractor


def main(args):
    ps = args.patch_size
    with open(args.labels, 'r') as f:
        labels = json.load(f)
    id_full_img = {lab['id']: lab['file_name'] for lab in labels['images']}
    f_annos = {img: [] for img in id_full_img.values()}
    for anno in labels['annotations']:
        f_annos[id_full_img[anno['image_id']]].append(anno)

    print("CONVERTING FULL IMG LABELS TO NEW PATCH SIZE")
    patcher = PatchExtractor(ps)
    img_pms = patcher.imgs_to_patches(args.full_imgs)
    new_image_patches, new_annotations = [], []
    anno_id = patch_id = 0
    for img, pms in tqdm(img_pms.items()):
        for pm in pms:
            new_image_patches.append({'width': ps, 'license': 0, 'file_name': pm['patch_path'], 'id': patch_id,
                                      'height': ps})
            x, y = pm['idx_w'], pm['idx_h']
            patch_poly = Polygon([(x, y), (x + ps, y), (x + ps, y + ps), (x, y + ps)])
            for anno in f_annos[img]:
                if len(anno['segmentation']) > 1:
                    print(img, "Unsupported polygon (", anno['segmentation'], ") skipping...")
                    continue
                a_poly = Polygon([anno['segmentation'][0][n:n+2] for n in range(0, len(anno['segmentation'][0]), 2)])
                if patch_poly.intersects(a_poly):
                    p_anno = copy.deepcopy(anno)
                    inter = patch_poly.intersection(a_poly)
                    # last two poly coords are starting point
                    pts = np.array(inter.exterior.coords).ravel()[:-2]
                    p_anno['segmentation'] = [(pts - np.ones(pts.shape) * np.array([x, y] * pts.size/2)).tolist()]
                    bbox = list(MultiPoint(inter.exterior.coords).bounds)
                    p_anno['bbox'] = [*bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # [x, y, width, height]
                    p_anno['area'] = inter.area
                    p_anno['image_id'] = patch_id
                    p_anno['id'] = anno_id
                    anno_id += 1
                    new_annotations.append(p_anno)
            patch_id += 1

    labels['images'] = new_image_patches
    labels['annotations'] = new_annotations
    json_path = os.path.join(args.full_imgs, f'patched{ps}_coco.json')
    print("SAVING", json_path)
    json.dump(labels, open(json_path, 'w'), sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts patch coco labels to full img labels")
    parser.add_argument('--full-imgs', type=str, required=True, help="Path of dir containing full imgs")
    parser.add_argument('--labels', type=str, help="JSON file containing patch labels in coco format")
    parser.add_argument('-p', '--patch-size', default=1024, type=int, help="patch size to convert labels to")
    args = parser.parse_args()

    common.check_dir_valid(args.full_imgs)
    common.check_file_valid(args.labels)

    common.time_method(main, args)

