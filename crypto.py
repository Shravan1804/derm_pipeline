import os
import argparse
import multiprocessing as mp

from cryptography.fernet import Fernet

import common
import concurrency


def load_key(key_file):
    with open(key_file, "rb") as kf:
        fkey = Fernet(kf.read())
    assert fkey is not None, f"Error while reading key {key_file}"
    return fkey


def decrypt(encrypted_file_path, fkey):
    with open(encrypted_file_path, "rb") as encrypted_data:
        return fkey.decrypt(encrypted_data.read())


def mp_encrypt(proc_id, files, fkey, dest):
    for i, file in enumerate(files):
        encrypted_dest = os.path.join(dest, os.path.basename(os.path.dirname(file)), os.path.basename(file))
        with open(file, "rb") as data, open(encrypted_dest, "wb") as writer:
            encrypted_data = fkey.encrypt(data.read())
            writer.write(encrypted_data)
        if i % 500 == 0:
            print(f"Proc {proc_id} encrypted {i}/{len(files)} files")

def main(args):
    fkey = load_key(args.key_file)
    workers, batch_size, batched_files = concurrency.batch_files_in_dirs(args.data)
    jobs = []
    for i, files in zip(range(workers), batched_files):
        jobs.append(mp.Process(target=mp_encrypt, args=(i, files, fkey, args.dest)))
        jobs[i].start()
    for j in jobs:
        j.join()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encrypt, decrypt data")
    parser.add_argument('--data', type=str, required=True, help="data to be encrypted")
    parser.add_argument('--dest', type=str, help="directory where the encrypted data is saved")
    parser.add_argument('--new-key', action='store_true', help="Creates new key file in --key-file")
    parser.add_argument('--key-file', type=str, help="file where to save key")
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)

    if args.new_key:
        if args.key_file is None:
            args.key_file = os.path.join(args.data, f'key_{os.path.basename(args.data)}.key')
        key = Fernet.generate_key()
        with open(args.key_file, "wb") as kf:
            kf.write(key)
    else:
        assert args.key_file is not None, "Error either provide --key-file or --new-key parameters"

    if args.dest is None:
        args.dest = common.maybe_create(os.path.dirname(args.data), f'{os.path.basename(args.data)}_encrypted')
    else:
        common.check_dir_valid(args.dest)
    for d in common.list_dirs(args.data):
        common.maybe_create(args.dest, d)


    common.time_method(main, args)
