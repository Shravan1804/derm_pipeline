import argparse
import copy
import json
import os
from collections import defaultdict

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from ..general import common
from ..general.PatchExtractor import PatchExtractor
from ..object_detection import coco_format


def patch_to_full_image(patch):
    """
    Convert patch filename to corresponding full image filename.

    :param patch: str, patch filename
    :return: str, full image filename
    """
    if PatchExtractor.can_extract_pm_from_patch_name(patch):
        return PatchExtractor.get_full_img_from_patch(patch)
    file, ext = os.path.splitext(os.path.basename(patch))
    return f"{file.split(PatchExtractor.SEP)[0]}{ext}"


def patch_position(patch):
    """
    Extract patch position (h,w) from its filename.

    :param patch: str, patch filename
    :return: tuple of int (h,w)
    """
    if PatchExtractor.can_extract_pm_from_patch_name(patch):
        return PatchExtractor.get_position(patch)
    h_w = os.path.splitext(os.path.basename(patch))[0].split(f"{PatchExtractor.SEP}_h")[
        1
    ]
    return tuple(map(int, h_w.split("_w")))


def main(args):
    """
    Perform patch to full image dataset conversion.

    :param args: command line arguments
    """
    with open(args.labels, "r") as f:
        labels = json.load(f)
    id_to_patch_name = {img["id"]: img["file_name"] for img in labels["images"]}
    patch_id_to_full_img = {
        k: patch_to_full_image(v) for k, v in id_to_patch_name.items()
    }
    full_img_id = {
        k: i for i, k in enumerate(sorted(set(patch_id_to_full_img.values())))
    }

    print("CREATING NEW LIST OF FULL IMAGES")
    new_images = []
    for fi, fid in full_img_id.items():
        img_path = os.path.join(args.full_imgs, fi)
        if not os.path.exists(img_path):
            print("Warning", img_path, "does not exist, skipping ...")
            continue
        new_images.append(coco_format.get_img_record(fid, img_path))

    print("ADJUSTING ANNOTATIONS POLYGONS TO FULL IMAGES")
    full_img_annos = defaultdict(list)
    for lanno in labels["annotations"]:
        for seg in lanno["segmentation"]:
            anno = copy.deepcopy(lanno)
            anno["segmentation"] = [seg]
            full_img = patch_id_to_full_img[anno["image_id"]]
            y, x = patch_position(id_to_patch_name[anno["image_id"]])
            anno["image_id"] = full_img_id[full_img]
            anno["bbox"] = (
                np.array(anno["bbox"]) + np.ones(4) * np.array([x, y, x, y])
            ).tolist()
            n = len(anno["segmentation"][0])
            anno["segmentation"] = (
                np.array(anno["segmentation"])
                + np.ones((1, n)) * np.array([[x, y] * int(n / 2)])
            ).tolist()
            full_img_annos[full_img].append(anno)

    print("MERGING OVERLAPPING POLYGONS")
    new_annotations = []
    anno_id = 1  # MUST start at 1 otherwise pycocotools.cocoeval will not match anno with id 0
    for full_img, f_annos in tqdm(full_img_annos.items()):
        cat_annos = {
            c["id"]: [a for a in f_annos if a["category_id"] == c["id"]]
            for c in labels["categories"]
        }
        for annos in cat_annos.values():
            polys = [
                Polygon([c[n : n + 2] for n in range(0, len(c), 2)])
                for c in [a["segmentation"][0] for a in annos]
            ]
            polys_annos = list(zip(polys, annos))
            merge = True
            while merge and len(polys_annos) > 0:
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
                    p3 = p3 if type(p3) is Polygon else unary_union(p3)
                    if type(p3) is Polygon:
                        polys_annos.append((p3, poly_to_anno(-1, p3, p1_a1[1])))
                    else:
                        print("Warning result of union not merged, skipping object")

            for _, anno in polys_annos:
                anno["id"] = anno_id
                new_annotations.append(anno)
                anno_id += 1

    labels["images"] = new_images
    labels["annotations"] = new_annotations
    json_path = os.path.join(
        os.path.dirname(args.full_imgs), f"{os.path.basename(args.full_imgs)}.json"
    )
    print("SAVING", json_path)
    json.dump(
        labels, open(json_path, "w"), sort_keys=True, indent=4, separators=(",", ": ")
    )


def poly_to_anno(new_id, poly, old_anno):
    """
    Convert polygon objects to annotation dict.

    :param new_id: int, annotation id
    :param poly: Polygon object
    :param old_anno: dict, base annotation dict
    :return: dict, annotation
    """
    new_anno = copy.deepcopy(old_anno)
    # last two poly coords are starting point
    new_anno["segmentation"] = (
        np.array(poly.exterior.coords).ravel()[:-2].reshape((1, -1)).tolist()
    )
    bbox = list(MultiPoint(poly.exterior.coords).bounds)  # [xmin, ymin, xmax, ymax]
    new_anno["bbox"] = [
        *bbox[:2],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]  # [x, y, width, height]
    new_anno["area"] = poly.area
    new_anno["id"] = new_id
    return new_anno


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts patch coco labels to full img labels"
    )
    parser.add_argument(
        "--full-imgs", type=str, required=True, help="Path of dir containing full imgs"
    )
    parser.add_argument(
        "--labels", type=str, help="JSON file containing patch labels in coco format"
    )
    parser.add_argument(
        "-p", "--patch-size", default=512, type=int, help="args.labels patch size"
    )
    args = parser.parse_args()

    common.check_dir_valid(args.full_imgs)
    common.check_file_valid(args.labels)

    common.time_method(main, args)
