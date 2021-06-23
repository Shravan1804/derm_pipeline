#!/usr/bin/env python

"""check_no_leak_train_test.py: Script used to find files present in two directories (also searches subdirectories)"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)



import os
import sys
import argparse
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, concurrency
from segmentation.segmentation_utils import common_segm_args


def get_image_name(args, path):
    """Extracts image name from path. If dataset is patched or cropped, will look for full img name
    :param args: command line args
    :param path: str, img path
    :return basename of original image without extension
    """
    img = os.path.splitext(os.path.basename(path))[0]
    img = '_'.join(img.split('_')[:-1]) if args.crops else img
    return img.split(args.patch_sep)[0] if args.patch else img


def get_files_to_search(args):
    """List files from provided image dataset according to specified dataset type
    :param args: command line args
    :return: tuple with two provided dirs files lists
    """
    if args.classif:
        dir1_files = common.list_files(args.dir1, full_path=True, recursion=True)
        dir2_files = common.list_files(args.dir2, full_path=True, recursion=True)
    elif args.segm:
        dir1_files = common.list_files(os.path.join(args.dir1, args.img_dir))
        dir2_files = common.list_files(os.path.join(args.dir2, args.img_dir))
    else:
        dir1_files = common.list_files(args.dir1)
        dir2_files = common.list_files(args.dir2)

    if len(dir1_files) > len(dir2_files):
        terms = dir1_files
        search_in = dir2_files
    else:
        terms = dir2_files
        search_in = dir1_files

    terms = [(get_image_name(args, t), t) for t in terms]
    search_in = [(get_image_name(args, t), t) for t in search_in]
    return terms, search_in


def search_terms(proc_id, terms, search_in, args):
    """Searches terms' items in search_in lst, prints number of matches, if verbose also prints the list of matches
    :param proc_id: int, process id
    :param terms: list, filepaths
    :param search_in: list, filepaths
    :param args: command line args
    """
    if args.verbose:
        print(f'Proc {proc_id} searching for {len(terms)} terms in a list of {len(search_in)} items.')
    count = 0
    duplicates = {}
    for t, tt in terms:
        for s, ss in search_in:
            if t == s:
                duplicates[t] = duplicates.get(t, []) + [(tt, ss)]
        if args.verbose and count > 0 and count % 5000 == 0:
            print(f'Proc {proc_id} completed {count}/{len(terms)} lookups ({len(terms)-count} remaining).')
        count += 1
    print(f'Proc {proc_id} found {len(duplicates.keys())} duplicates: {duplicates.keys()}')
    if args.verbose:
        for k, v in duplicates.items():
            print(f"{k} has {len(v)} occurrences: {v}")


def main(args):
    """Runs the multiprocess search based on the provided command line arguments
    :param args: command line args
    """
    terms, search_in = get_files_to_search(args)
    workers, batch_size, batched_dirs = concurrency.batch_lst(terms)
    concurrency.multi_process_fn(workers, batched_dirs, partial(search_terms, search_in=search_in, args=args))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finds if dir2 files present in dir1")
    parser.add_argument('--dir1', type=str, required=True, help="first directory (e.g. train dir)")
    parser.add_argument('--dir2', type=str, required=True, help="second directory (e.g. test dir)")
    parser.add_argument('--classif', action='store_true', help="expect class subdirs in provided dirs")
    parser.add_argument('--segm', action='store_true', help="if dataset is segm dataset")
    common_segm_args(parser)
    parser.add_argument('--crops', action='store_true', help="dir contains imgs_crops => checks if same orig img")
    parser.add_argument('--patch', action='store_true', help="dir contains patches => checks if same img in both dirs")
    parser.add_argument('--patch-sep', type=str, default='__SEP__', help="patch name separator")
    parser.add_argument('--verbose', action='store_true', help="show occurrences")
    args = parser.parse_args()

    common.check_dir_valid(args.dir1)
    common.check_dir_valid(args.dir2)

    common.time_method(main, args)
