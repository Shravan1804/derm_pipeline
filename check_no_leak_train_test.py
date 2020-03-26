import os
import argparse
import multiprocessing as mp

import common
import concurrency


def get_image_name(args, path):
    img = os.path.splitext(os.path.basename(path))[0]
    return img.split(args.patch_sep)[0] if args.patch else img


def get_files_to_search(args):
    if args.classif:
        dir1_files = common.list_files_in_dirs(args.dir1, full_path=True)
        dir2_files = common.list_files_in_dirs(args.dir2, full_path=True)
    elif args.obj_detec:
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


def search_terms(proc_id, terms, search_in):
    print(f'Proc {proc_id} searching for {len(terms)} terms in a list of {len(search_in)} items.')
    count = 0
    for t, tt in terms:
        for s, ss in search_in:
            if t == s:
                print(f'Proc {proc_id} found image {t} in both {tt} and {ss}')
        if count > 0 and count % 5000 == 0:
            print(f'Proc {proc_id} completed {count}/{len(terms)} lookups ({len(terms)-count} remaining).')
        count += 1



def main(args):
    terms, search_in = get_files_to_search(args)
    workers, batch_size, batched_dirs = concurrency.batch_dirs(terms)
    jobs = []
    for i, batch in zip(range(workers), batched_dirs):
        jobs.append(mp.Process(target=search_terms, args=(i, batch, search_in)))
        jobs[i].start()
    for j in jobs:
        j.join()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finds if dir2 files present in dir1")
    parser.add_argument('--dir1', type=str, required=True, help="first directory (e.g. train dir)")
    parser.add_argument('--dir2', type=str, required=True, help="second directory (e.g. test dir)")
    parser.add_argument('--classif', action='store_true', help="expect class subdirs in provided dirs")
    parser.add_argument('--patch', action='store_true', help="dir contains patches => checks if same img in both dirs")
    parser.add_argument('--patch-sep', type=str, default='__SEP__', help="patch name separator")
    common.add_obj_detec_args(parser)
    args = parser.parse_args()

    common.check_dir_valid(args.dir1)
    common.check_dir_valid(args.dir2)

    common.time_method(main, args)
