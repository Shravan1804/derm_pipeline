import os
import cv2
import time
import common
import datetime
import concurrency
import numpy as np
from shutil import move
import multiprocessing as mp
from functools import partial


import fastai
from radam import *


def load_and_prepare_img(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    t = fastai.vision.pil2tensor(im, dtype=im.dtype)  # converts to numpy tensor
    return fastai.vision.Image(t.float() / 255.)  # Convert to float


def correct_labels(args, task):
    logs = ""
    start = time.time()
    idx, batch = task
    model = common.load_fastai_model(args.model_path) if args.std_proc else mp.current_process().model
    for img_path in batch:
        cat = os.path.basename(os.path.dirname(img_path))
        pred = str(model.predict(load_and_prepare_img(img_path))[0])
        if cat != pred:
            f = os.path.basename(img_path)
            move(img_path, os.path.join(args.data, pred, f))
            logs += f"{f};{cat};{pred}\n"
    if idx % 100 == 0:
        print(f"Process {os.getpid()} completed task {idx} in {datetime.timedelta(seconds=time.time() - start)}.")
    return logs


def main(args, ctx=None):
    start = time.time()
    all_dirs = common.list_dirs(args.data)
    tasks = concurrency.batch_files_in_dirs(args.data, all_dirs, bs=args.proc_bs)
    pool = mp.Pool(processes=args.workers) if ctx is None else ctx.Pool(processes=args.workers)
    with open(args.log, 'w') as logger:
        logger.write('file;old_label;new_label\n')
        for correction in pool.imap_unordered(partial(correct_labels, args), zip(range(len(tasks)), tasks)):
            logger.write(f'{correction}\n')
    pool.close()
    pool.join()
    print(f"Work completed in {datetime.timedelta(seconds=time.time() - start)}.")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Correct dataset labels")
    parser.add_argument('--data', type=str, required=True, help="source dataset to correct")
    parser.add_argument('--model', type=str, required=True, help="model path")

    parser.add_argument('--device', type=str, default="cpu", help="cpu or gpu")
    parser.add_argument('--proc_bs', type=int, default=100, help="Number of imgs per process")
    parser.add_argument('--workers', type=int, help="Number of process, will adjust to chosen device if needed")
    parser.add_argument('--log', default="log.txt", type=str, help="log file where label corrections will be tracked")
    parser.add_argument('--std-proc', action='store_true', help="Use standard process instead of custom process")
    parser.add_argument('--seed', default=42, type=int, help="batch size")
    args = parser.parse_args()

    common.check_args(args)
    common.set_seeds(args.seed, np=np)

    assert os.path.exists(args.model), f"Provided model path do not exist: {args.model}"

    if args.device == "gpu" and torch.cuda.is_available():
        fastai.torch_core.defaults.device = 'gpu'
        if args.workers is None:
            args.workers = torch.cuda.device_count()
        else:
            args.workers = min(torch.cuda.device_count(), args.workers)
    else:
        args.device = "cpu"
        fastai.torch_core.defaults.device = 'cpu'
        if args.workers is None:
            args.workers = mp.cpu_count()

    if args.std_proc:
        mp_ctx = None
    else:
        # https://stackoverflow.com/questions/740844/python-multiprocessing-pool-of-custom-processes
        mp_ctx = mp.get_context()  # get the default context

        class CustomProcess(mp_ctx.Process):
            def __init__(self, *argv, **kwargs):
                self.model = common.load_fastai_model(args.model)
                super().__init__(*argv, **kwargs)
                print(f"Custom process {os.getpid()} created.")

        mp_ctx.Process = CustomProcess  # override the context's Process

    main(args, mp_ctx)


