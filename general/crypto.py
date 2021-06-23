#!/usr/bin/env python

"""crypto.py: Encrypts directory structure, also contains methods used for decryption"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)


import os
import io
import sys
import shutil
import pickle
import argparse
import multiprocessing as mp

import PIL
import numpy as np
from bidict import bidict
from cryptography.fernet import Fernet

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, concurrency


def add_encrypted_args(parser):
    """Adds crypto command line arguments to provided argument parser
    :param parser: ArgumentParser object
    """
    parser.add_argument('--encrypted', action='store_true', help="Data is encrypted")
    parser.add_argument('--ckey', type=str, help="Data encryption key")


def decrypt_img(img_path, fkey):
    """Decrypts provided image file using Fernet key object and load it as np array
    :param img_path: str, path of image to decrypt
    :param fkey: Fernet object, key used to decrypt file
    :return np array with image data
    """
    return np.array(PIL.Image.open(io.BytesIO(decrypt(img_path, fkey))))
    # CV2 version, hangs in distributed mode
    #import cv2
    #im = cv2.imdecode(np.frombuffer(decrypt(img_path, fkey), np.uint8), cv2.IMREAD_UNCHANGED)
    #return common.img_bgr_to_rgb(im) if len(im.shape) > 2 else im


def request_key(test_key_dir=None, user_key=None):
    """Asks crypto key from user input and converts it to Fernet object key, testing it if requested
    :param test_key_dir: str, directory where crypted version of the key can be found to test user key.
    Can be left None if no key testing required.
    :param user_key: str, crypto key, if left None will ask for user input
    :return Fernet object key
    """
    if user_key is None:
        print("Please provide dataset encryption key")
        user_key = input().encode()
    try:
        user_fkey = Fernet(user_key)
        if test_key_dir is not None:
            crypted_key_files = [kf for kf in common.list_files(test_key_dir, full_path=True) if kf.endswith('.key')]
            assert len(crypted_key_files) > 0, f"No crypted key in provided dir {test_key_dir}"
            crypted_key = crypted_key_files[0]
            with open(crypted_key, "rb") as kf:
                assert user_key == user_fkey.decrypt(kf.read()), f"Error with provided key, checked with {crypted_key}"
    except Exception:
        raise Exception("Provided key invalid.")
    return user_fkey


def load_key(key_file):
    """Loads key from provided file path
    :param key_file: str, path of key file
    returns: Fernet object key
    """
    with open(key_file, "rb") as kf:
        fkey = Fernet(kf.read())
    assert fkey is not None, f"Error while reading key {key_file}"
    return fkey


def decrypt(encrypted_file_path, fkey):
    """Decrypts provided file using provided key
    :param encrypted_file_path: str, path of encrypted file
    :param fkey: Fernet object, key
    :return decrypted file content
    """
    with open(encrypted_file_path, "rb") as encrypted_data:
        return fkey.decrypt(encrypted_data.read())


def mp_encrypt(proc_id, files, fkey, root, dest, filenames_dict):
    """Method used to multiprocess file encryption of provided files
    :param proc_id: int, id of process
    :param files: list, filepaths to be encrypted by process
    :param fkey: Fernet object, key
    :param root:  str, directory containing the original files
    :param dest: str, directory where to store the encrypted files
    :param filenames_dict: bidict, mapping of encrypted filenames to original filenames
    """
    for i, file in enumerate(files):
        name, ext = os.path.splitext(os.path.basename(file))
        encrypted_dest = file.replace(root, dest).replace(name + ext, filenames_dict[name] + ext)
        with open(file, "rb") as data, open(encrypted_dest, "wb") as writer:
            encrypted_data = fkey.encrypt(data.read())
            writer.write(encrypted_data)
        if i % 500 == 0:
            print(f"Proc {proc_id} encrypted {i}/{len(files)} files")


def create_filenames_dict(files, save_path):
    """Creates bidict mapping between encrypted filenames and original filenames
    :param files: list, filepaths
    :param save_path: str, path where to save the created mapping
    :return created bidict
    """
    n0 = len(str(len(files))) + 1
    filenames_dict = bidict()
    for i, f in enumerate(files):
        name, ext = os.path.splitext(os.path.basename(f))
        if name not in filenames_dict:
            filenames_dict[name] = str(i).zfill(n0)
    pickle.dump(filenames_dict, open(save_path, "wb"))
    return filenames_dict


def main(args):
    """Runs the multiprocess encryption based on the provided command line arguments
    :param args: command line args
    """
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
    parser.add_argument('--dest', type=str, help="directory where the encrypted data is saved, default will be data"
                                                 "dirname with _encrypted appended")
    parser.add_argument('--new-key', action='store_true', help="Creates new key file in --key-file")
    parser.add_argument('--key-file', type=str, help="filepath where to save key, default will be data dirname with"
                                                     ".key ext stored in data dir")
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
        copied_key = os.path.join(args.data, f'{os.path.basename(args.data)}.key')
        shutil.copy(args.key_file, copied_key)
        args.key_file = copied_key

    if args.dest is None:
        args.dest = common.maybe_create(os.path.dirname(args.data), f'{os.path.basename(args.data)}_encrypted')
    else:
        common.check_dir_valid(args.dest)
    common.reproduce_dir_structure(args.data, args.dest)

    common.time_method(main, args)
