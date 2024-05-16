import argparse
import os
import random
from functools import partial

from p_tqdm import p_umap
from tqdm import tqdm

from ..general import common
from ..general import common_img as cimg


def knapSack(W, wt, val):
    """
    Solve knapSack problem, source https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/.

    :param W: total capacity to fill
    :param wt: weights of items
    :param val: values of items
    :return total value, dynamic programming table
    """
    n = len(wt)
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    return K[n][W], K


def knapSack_with_items(W, wt, val, items):
    """
    Solve knapSack problem, source https://www.geeksforgeeks.org/printing-items-01-knapsack/.

    :param W: total capacity to fill
    :param wt: weights of items
    :param val: values of items
    :return total value and selected items
    """
    n, selected_items = len(wt), []
    res, K = knapSack(W, wt, val)
    w = W
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # either the result comes from the
        # top (K[i-1][w]) or from (val[i-1]
        # + K[i-1] [w-wt[i-1]]) as in Knapsack
        # table. If it comes from the latter
        # one/ it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:
            # This item is included.
            selected_items.append(items[i - 1])
            # Since this weight is included
            # its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]
    return res, selected_items


def test_knapSack():
    """Test knapSack implementation."""
    items = ["a", "b", "c"]
    val = [21, 22, 29]
    wt = [21, 22, 29]
    knapSack_with_items(51, wt, val, items)


def split_files(seed, dirpath, test_prop, get_patient_id):
    """
    Split images from directory into train and test images.

    :param seed: random use for split
    :param dirpath: directory path
    :param test_prop: test proportion
    :param get_patient_id: function to extract patient id from image path
    :return tuple of lists with train and test images
    """
    impaths = common.list_images(dirpath, full_path=True)
    patient_pics = {}
    for impath in impaths:
        pid = get_patient_id(impath)
        if pid in patient_pics:
            patient_pics[pid].append(impath)
        else:
            patient_pics[pid] = [impath]
    pids = list(patient_pics.keys())
    random.Random(seed).shuffle(pids)  # added for HE dataset (4th april 22)
    pid_npics = [len(patient_pics[pid]) for pid in pids]

    # better sample train size first in case lots of pictures with same pid
    train_size = len(impaths) - int(test_prop * len(impaths)) - 1
    _, train_pids = knapSack_with_items(train_size, pid_npics, pid_npics, pids)
    test_pids = [pid for pid in pids if pid not in train_pids]
    train_impaths = [impath for pid in train_pids for impath in patient_pics[pid]]
    test_impaths = [impath for pid in test_pids for impath in patient_pics[pid]]
    return train_impaths, test_impaths


def split_and_save(category, dirpath, test_prop, dest_root, pidx_fn, seed):
    """
    Split directory and save images in respective dir.

    :param category: directory name
    :param dirpath: directory path
    :param test_prop: test proportion
    :param dest_root: destination directory path
    :param pidx_fn: function to extract patient id from image path
    :param seed: random seed for split
    """
    train_impaths, test_impaths = split_files(seed, dirpath, test_prop, pidx_fn)
    for setname, impaths in zip(("train", "test"), (train_impaths, test_impaths)):
        dest_dir = common.maybe_create(dest_root, setname, category)
        for impath in impaths:
            # shutil copy can create issues...
            cimg.save_img(
                cimg.load_img(impath), os.path.join(dest_dir, os.path.basename(impath))
            )


def split_classif_dataset(root_dir, test_prop, pidx_fn, seed):
    """
    Split typical classification dataset.

    :param root_dir: image directory path
    :param test_prop: test proportion
    :param pidx_fn: function to extract patient id from image path
    :return:
    """
    classes = common.list_dirs(root_dir)
    dest_root = common.maybe_create(root_dir.rstrip("/") + "_splitted")
    fn = partial(
        split_and_save,
        test_prop=test_prop,
        dest_root=dest_root,
        pidx_fn=pidx_fn,
        seed=seed,
    )
    p_umap(fn, classes, [os.path.join(root_dir, c) for c in classes])


def split_segm_dataset(
    root_dir, test_prop, pidx_fn, seed, imdir="images", mdir="masks"
):
    """
    Split typical segmentation dataset.

    :param root_dir: directory path where images and masks are located
    :param test_prop: test proportion
    :param pidx_fn: function to extract patient id from image path
    :param imdir: image directory name
    :param mdir: mask directory name
    """
    dest_root = common.maybe_create(root_dir.rstrip("/") + "_splitted")
    split_and_save(
        imdir, os.path.join(root_dir, imdir), test_prop, dest_root, pidx_fn, seed
    )
    for d in ["train", "test"]:
        dest_mdir = common.maybe_create(dest_root, d, mdir)
        for imname in tqdm(common.list_images(os.path.join(dest_root, d, imdir))):
            mpath = os.path.join(root_dir, mdir, os.path.splitext(imname)[0] + ".png")
            cimg.save_img(
                cimg.load_img(mpath), os.path.join(dest_mdir, os.path.basename(mpath))
            )


def usz_get_patient_id(impath):
    """
    Extract patient id from usz image path.

    :param impath: image path
    :return: patient id as str
    """
    imname = os.path.basename(impath)
    # return imname.split('_')[0] if imname[0].isnumeric() else imname.split('-')[0]
    return imname.split("_")[0]


def he_get_patient_idx(impath):
    """
    Extract patient id from hand eczema image path.

    :param impath: image path
    :return: patient id as str
    """
    idx = os.path.basename(impath).split("_")[0]
    # if photobox img then idx is the patient idx
    # else last 5 char are D (dorsal) or P (palm) + ext
    return idx[:-5] if ".jpg" in idx else idx


def main(args):
    """
    Split dataset following command line arguments.

    :param args: command line arguments
    """
    if args.pidx_fn == "usz":
        pidx_fn = usz_get_patient_id
    elif args.pidx_fn == "he":
        pidx_fn = he_get_patient_idx
    if args.classif:
        split_classif_dataset(args.data, args.test_size, pidx_fn, args.seed)
    elif args.segm:
        split_segm_dataset(args.data, args.test_size, pidx_fn, args.seed)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset in train/test dataset")
    parser.add_argument(
        "--data", type=str, required=True, help="source dataset root directory"
    )
    parser.add_argument(
        "--test-size", default=0.2, type=float, help="Proportion of test set"
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--pidx-fn", type=str, default="usz", help="Function to extract patient idx"
    )
    parser.add_argument("--classif", action="store_true", help="Classification dataset")
    parser.add_argument("--segm", action="store_true", help="Segmentation dataset")
    args = parser.parse_args()

    common.check_dir_valid(args.data)
    args.data = args.data.rstrip("/")

    common.time_method(main, args)
