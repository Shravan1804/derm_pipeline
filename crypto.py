import os
import pickle
import argparse
import multiprocessing as mp

from bidict import bidict
from cryptography.fernet import Fernet

import common
import concurrency


def request_key(key_dir, user_key=None):
    if user_key is None:
        print("Please provide dataset encryption key")
        user_key = input().encode()
    try:
        user_fkey = load_user_provided_key(user_key)
        crypted_key = [kf for kf in common.list_files(key_dir, full_path=True) if kf.endswith('.key')][0]
        with open(crypted_key, "rb") as kf:
            assert user_key == user_fkey.decrypt(kf.read()), f"Error with provided key, checked with {crypted_key}"
    except Exception:
        raise Exception("Provided key invalid.")
    return user_fkey


def load_user_provided_key(user_key=None):
    user_key = input().encode() if user_key is None else user_key
    return Fernet(user_key)


def load_key(key_file):
    with open(key_file, "rb") as kf:
        fkey = Fernet(kf.read())
    assert fkey is not None, f"Error while reading key {key_file}"
    return fkey


def decrypt(encrypted_file_path, fkey):
    with open(encrypted_file_path, "rb") as encrypted_data:
        return fkey.decrypt(encrypted_data.read())


def mp_encrypt(proc_id, files, fkey, root, dest, filenames_dict):
    for i, file in enumerate(files):
        name, ext = os.path.splitext(os.path.basename(file))
        encrypted_dest = file.replace(root, dest).replace(name + ext, filenames_dict[name] + ext)
        with open(file, "rb") as data, open(encrypted_dest, "wb") as writer:
            encrypted_data = fkey.encrypt(data.read())
            writer.write(encrypted_data)
        if i % 500 == 0:
            print(f"Proc {proc_id} encrypted {i}/{len(files)} files")


def create_filenames_dict(files, save_path):
    n0 = len(str(len(files))) + 1
    filenames_dict = bidict()
    for i, f in enumerate(files):
        name, ext = os.path.splitext(os.path.basename(f))
        if name not in filenames_dict:
            filenames_dict[name] = str(i).zfill(n0)
    pickle.dump(filenames_dict, open(save_path, "wb"))
    return filenames_dict


def main(args):
    fkey = load_key(args.key_file)
    files = common.list_files(args.data, full_path=True, recursion=True)
    filenames_dict = create_filenames_dict(files, os.path.join(args.data, f'{os.path.basename(args.data)}_filenames.p'))
    workers, batch_size, batched_files = concurrency.batch_lst(files)
    jobs = []
    for i, files in zip(range(workers), batched_files):
        jobs.append(mp.Process(target=mp_encrypt, args=(i, files, fkey, args.data, args.dest, filenames_dict)))
        jobs[i].start()
    for j in jobs:
        j.join()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encrypt, decrypt dataset")
    parser.add_argument('--data', type=str, required=True, help="dataset to be encrypted")
    parser.add_argument('--dest', type=str, help="directory where the encrypted data is saved")
    parser.add_argument('--new-key', action='store_true', help="Creates new key file in --key-file")
    parser.add_argument('--key-file', type=str, help="file where to save key")
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)

    if args.new_key:
        if args.key_file is None:
            args.key_file = os.path.join(args.data, f'{os.path.basename(args.data)}.key')
        key = Fernet.generate_key()
        with open(args.key_file, "wb") as kf:
            kf.write(key)
    else:
        assert args.key_file is not None, "Error either provide --key-file or --new-key parameters"

    if args.dest is None:
        args.dest = common.maybe_create(os.path.dirname(args.data), f'{os.path.basename(args.data)}_encrypted')
    else:
        common.check_dir_valid(args.dest)
    common.reproduce_dir_structure(args.data, args.dest)

    common.time_method(main, args)
