import os
import sys
import argparse
import multiprocessing as mp

import PIL

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, concurrency, crypto


def get_all_files(args):
    datasets = common.list_dirs(args.data, full_path=True) if args.multi else [args.data]
    files = []
    for ds in datasets:
        if args.classif or args.segm:
            ds_files = common.list_files_in_dirs(ds, full_path=True)
        else:
            ds_files = common.list_files(ds, full_path=True)
        files.extend(ds_files)
    return files


def try_open_files(proc_id, files, args):
    if args.verbose: print(f'Proc {proc_id} started processing {len(files)} files.')
    for f in files:
        try:
            im = crypto.decrypt_img(f, args.ckey) if args.encrypted else PIL.Image.open(f)
        except Exception as err:
            print(f'Proc {proc_id} encountered an error with file {f}, skipped')
    if args.verbose: print(f'Proc {proc_id} completed processing {len(files)} files.')


def main(args):
    print("Checking data integrity of", args.data)
    files = get_all_files(args)
    workers, batch_size, batched_files = concurrency.batch_lst(files, bs=args.bs, workers=args.workers)
    jobs = []
    for i, batch in zip(range(workers), batched_files):
        jobs.append(mp.Process(target=try_open_files, args=(i, batch, args)))
        jobs[i].start()
    for j in jobs:
        j.join()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Goes through every files and verifies if it can be loaded")
    parser.add_argument('--data', type=str, required=True, help="data dir")
    parser.add_argument('--multi', action='store_true', help="if --data contains multiple datasets")
    parser.add_argument('--classif', action='store_true', help="expects class subdirs in provided dirs")
    parser.add_argument('--segm', action='store_true', help="expects images and masks subdirs")
    parser.add_argument('--verbose', action='store_true', help="informs on progress")
    concurrency.add_multi_proc_args(parser)
    crypto.add_encrypted_args(parser)
    args = parser.parse_args()

    common.check_dir_valid(args.data)

    if args.encrypted: args.ckey = crypto.request_key(args.data, args.ckey)

    common.time_method(main, args)
