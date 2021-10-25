#!/usr/bin/env python

"""common.py: File regrouping useful methods"""

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
import math
import datetime
import contextlib
from timeit import default_timer
from string import ascii_uppercase
from collections import defaultdict
from pathlib import Path, PosixPath


import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))


def recursive_default_dict():
    """Creates recursive default dict.
    :return recursive default dict
    """
    return lambda: defaultdict(recursive_default_dict)


def merge(lst, cond_fn, merge_fn):
    """Applies recursively merge fn on pairs of lst items if cond_fn is satisfied
    :param lst: list, elements to be merged
    :param conf_fn: function, fn to be called on two elements, should return boolean
    :param merge_fn: function, fn performing the merge on two elements
    :return list resulting list after merge
    """
    for i, item_a in enumerate(lst):
        sub_lst = lst[:i] + lst[i+1:]
        for j, item_b in enumerate(sub_lst):
            if cond_fn(item_a, item_b):
                new_lst = sub_lst[:j] + sub_lst[j+1:] + [merge_fn(item_a, item_b)]
                return merge(new_lst, cond_fn, merge_fn)
    return lst


def decimal_to_base(n, base, ndigits=-1):
    """Converts n from base 10 to specified base
    :param n: int, number to be converted
    :param base: int, base to convert number in
    :param ndigits_ int, optional number of digits to be used
    :return list digits of n in provided base
    :raise AssertionError if ndigits too small compared to n
    """
    res, count = [], 0
    while(n > 0):
        rem = int(n % base)
        res = [rem] + res
        n = (n-rem)/base
        count += 1
    if ndigits != -1:
        assert count <= ndigits, f"{int(n)} cannot be converted in base {base} with only {ndigits} digits."
        while count+1 <= ndigits:
            res = [0] + res
            count += 1
    elif count == 0: res = [0]
    return res


def generate_codes(ncodes, alphabet=ascii_uppercase, size=None):
    """Generates unique codes based on provided alphabet
    :param ncodes: int, number of codes to generate
    :param alphabet: list, alphabet to be used
    :param size: int, length of codes
    :return list list of codes
    :raise AssertionError if size too small compared to ncodes
    """
    n_symbols = len(set(ascii_uppercase))
    min_symbols = int(math.log(ncodes, len(alphabet))) + 1
    if size is None: size = min_symbols
    if size != -1:
        error_msg = f"Error, requested {ncodes} different codes but an alphabet with only {n_symbols}" \
                    f"different symbols was provided (require at least {min_symbols} different symbols)."
        assert size >= min_symbols, error_msg
    return ["".join([alphabet[i] for i in decimal_to_base(c, len(alphabet), ndigits=size)]) for c in range(ncodes)]


def int_to_bins(n, bins, rand=False):
    """Integer division of number into bins. If rand true, random split else equal split
    :param n: int, number to be integer divided in bins
    :param bins: int, number of bins (=N)
    :param rand: bool, random split or equal split
    :return array, counts in each bins, size N
    """
    if n <= 0 or bins <= 0:
        return np.array([])
    if rand:
        temp = np.concatenate([np.zeros(n, dtype=np.bool), np.ones(bins-1, dtype=np.bool)])
        np.random.shuffle(temp)
        return np.array([(~t).sum() for t in np.split(temp, temp.nonzero()[0])])
    else:
        return np.arange(n+bins-1, n-1, -1) // bins


def equidistant_pts(mini, maxi, inter):
    """Prepares list of equidistant points between range separated by fixed distance
    :param mini: float, starting value
    :param maxi: float, ending value (comprised)
    :param inter: float, distance between pts
    :return: list of floats, equidistant pts
    """
    return np.linspace(mini, maxi, np.int(np.round(np.abs(maxi - mini) / inter)) + 1, endpoint=True).tolist()


def most_common(arr, top=3, return_index=False, return_counts=False):
    """Returns most common elements in array
    :param arr: array, elements to compare
    :param top: int, number of elements to return (=N)
    :param return_index: array, index of elements, size N
    :param return_counts: array, counts of elements, size N
    :return single array or tuple of array depending on parameters
    """
    u, c = np.unique(arr, return_counts=True)
    sorted_c = c.argsort()[::-1]
    res = u[sorted_c[:top]]
    if return_index:
        res = res, sorted_c[:top]
    if return_counts:
        res = *res, c[res[-1]]
    return res


def is_path(item):
    """Checks if input can be path
    :return bool, item is path
    """
    return type(item) in (str, PosixPath, Path)


def maybe_create(*d):
    """Receives arbitrary number of dirnames, joins them and create them if they don't exist. Returns joined path.
    :param *d: list, strings to be joined together as path
    :return str, joined path
    """
    path = os.path.join(*d)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def reproduce_dir_structure(source, dest):
    """Reproduce dir structure of source in dest. Raise exception if source invalid. Creates dest if needed.
    :param source: str, dir to be reproduced
    :param dest: str, dir where to copy source structure
    """
    check_dir_valid(source)
    maybe_create(dest)
    for d in list_dirs(source):
        reproduce_dir_structure(os.path.join(source, d), os.path.join(dest, d))


def list_images(root, **kwargs):
    """List images ('.jpg', '.png', '.tiff') in provided dir. Check list_files args for kwargs
    :param root: str, dir to retrieve file from
    :param kwargs: Check list_files args for kwargs
    :return: Check list_files args for kwargs
    """
    return [f for f in list_files(root, **kwargs) if os.path.splitext(f)[1] in ('.jpg', '.png', '.tiff')]


def list_files(root, full_path=False, posix_path=False, recursion=False, max_rec_level=-1):
    """Return the list of files, if recursion, stops at max_rec_level unless negative then goes all levels
    :param root: str, dir to retrieve file from
    :param full_path: bool, whether to return full path
    :param posix_path: bool, whether to return paths as posix objs
    :param recursion: bool, whether to look in subdirectories
    :param max_rec_level: int, max subdirectory level
    :return: list of files
    """
    lf = [os.path.join(root, f) if full_path else f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    if recursion and max_rec_level != 0:
        for d in list_dirs(root, full_path, posix_path, recursion, max_rec_level-1):
            full_d = d if full_path else os.path.join(root, d)
            lf.extend([os.path.join(d, f) for f in os.listdir(full_d) if os.path.isfile(os.path.join(full_d, f))])
    return [Path(i) for i in sorted(lf)] if posix_path else sorted(lf)


def list_dirs(root, full_path=False, posix_path=False, recursion=False, max_rec_level=-1, rec_level=0):
    """Return the list of dirs, if recursion stops at max_rec_level unless negative then goes all levels.
    :param root: str, dir to retrieve file from
    :param full_path: bool, whether to return full path
    :param posix_path: bool, whether to return paths as posix objs
    :param recursion: bool, whether to look in subdirectories
    :param max_rec_level: int, max subdirectory level
    :param rec_level: used for recursion, do not be set
    :return: list of dirs
    """
    if recursion and rec_level > max_rec_level >= 0:
        return []
    ld = [os.path.join(root, d) if full_path else d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if recursion:
        ld_rec = []
        for d in ld:
            new_root = d if full_path else os.path.join(root, d)
            children = list_dirs(new_root, full_path, posix_path, recursion, max_rec_level, rec_level+1)
            ld_rec.extend([c if full_path else os.path.join(d, c) for c in children])
        ld += ld_rec
    return [Path(d) for d in sorted(ld)] if posix_path else sorted(ld)


def list_files_in_dirs(root, full_path=False, posix_path=False):
    """Return the list of files present in level 1 subdirs. Typicall classification dataset format.
    :param root: str, dir to retrieve file from
    :param full_path: bool, whether to return full path
    :param posix_path: bool, whether to return paths as posix objs
    :return: list of files
    """
    lf = [os.path.join(d, f) for d in list_dirs(root, full_path)
          for f in list_files(d if full_path else os.path.join(root, d))]
    return [Path(i) for i in lf] if posix_path else lf


def check_file_valid(filepath):
    """Check if filepath exists and is a file
    :param filepath: str
    :raise: AssertionError if filepath not valid
    """
    assert os.path.exists(filepath) and os.path.isfile(filepath), f"Provided file {filepath} invalid."


def check_dir_valid(dirpath):
    """Check if dirpath exists and is a dir
    :param dirpath: str
    :raise: AssertionError if dirpath not valid
    """
    assert os.path.exists(dirpath) and os.path.isdir(dirpath), f"Provided dir {dirpath} invalid."


def batch_list(lst, bs):
    """Batch list in sublist
    :param lst: list, elements to be batched
    :param bs: int, batch size
    :return: list of sublist, sublists are the batches
    """
    return [lst[i:min(len(lst), i + bs)] for i in range(0, len(lst), bs)]


def now():
    """Get current datetime as str
    :return: str, formatted datetime
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def zero_pad(it, max_it):
    """Prepend zeros to iteration id so that alphabetic sort corresponds to iteration order
    :param it: int, iteration id
    :param max_it: int, total number of iteration
    :return: str iteration id with zeros prepended
    """
    return str(it).zfill(len(str(max_it)) + 1)


def set_seeds(seed, cuda_seeded=False):
    """Seed everything
    :param seed: int
    :param cuda_seeded: bool, should cuda be seeded as well
    """
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_seeded and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def mynullcontext(enter_result=None):
    """Python 3.6 alternative to contextlib.nullcontext() available from 3.8
    :param enter_result:
    :return:
    """
    yield enter_result


@contextlib.contextmanager
def elapsed_timer():
    """Context manager timing whatever executes within
    source: https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python?page=2&tab=active#tab-top
    :return: int, elapsed time
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def time_method(m, *args, text="Work", **kwargs):
    """Helper function to use elapsed_timer context manager on function
    :param m: function, to be executed and timed
    :param args: list, positional arguments for function m
    :param text: str, description of method to be printed alongside time information
    :param kwargs: dict, keywords argument for function m
    :return: output of m
    """
    with elapsed_timer() as elapsed:
        res = m(*args, **kwargs)
        print(f"{text} completed in {datetime.timedelta(seconds=elapsed())}.")
    return res


class PrintPrepender:
    """Prepend text to stdout
    source: https://stackoverflow.com/questions/58866481/how-could-i-override-pythons-print-function-to-prepend-some-arbitrary-text-to-e
    """
    stdout = sys.stdout

    def __init__(self, text_to_prepend):
        self.text_to_prepend = text_to_prepend
        self.buffer = [self.text_to_prepend]

    def write(self, text):
        lines = text.splitlines(keepends=True)
        for line in lines:
            self.buffer.append(line)
            self.flush()
            if line.endswith(os.linesep):
                self.buffer.append(self.text_to_prepend)

    def flush(self, *args):
        self.stdout.write(''.join(self.buffer))
        self.stdout.flush()
        self.buffer.clear()


def stdout_prepend(f, pre_msg, *args):
    """Helper function to execute function f with stdout prepend
    :param f: function, to execture
    :param pre_msg: str, message to be prepended
    :param args: list, positional arguments for function m
    """
    with contextlib.redirect_stdout(PrintPrepender(pre_msg)):
        f(*args)


def update_df1_with_df2(df1, df2, on):
    """Update pandas dataframe with values from other dataframe
    :param df1: dataframe 1 to be updated
    :param df2: dataframe 2 to take the values from
    :param on: str or lst of str, column names to be used to match rows between dataframes
    :return: df1 updated with values from df2
    """
    for c in df2.columns:
        if c not in df1.columns:
            df1[c] = ''

    df1.set_index(on, inplace=True)
    df2.set_index(on, inplace=True)

    df1.update(df2)
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    return df1
