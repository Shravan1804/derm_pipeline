import os
import shutil

src = '/home/shravan/deep-learning/data/ppp_test_study_raw_sample'
if src.endswith('/'):
    src = src[:-1]
out = src + '_processed'
if os.path.exists(out):
    raise Exception(f'Destination directory already exists: {out}')
else:
    os.makedirs(out)

def flatten_dir(prefix, directory, out):
    for d in os.listdir(directory):
        if d.startswith('.'):   # ignore hidden files
            continue
        path = os.path.join(directory, d)
        if os.path.isfile(path):
            shutil.copy(path, os.path.join(out, prefix + '_' + d))
        elif os.path.isdir(path):
            print("going deeper")
            flatten_dir(prefix + '_' + d, path, out)
        else:
            raise Exception(f'Path is neither a directory or a file: {path}')

flatten_dir('', src, out)