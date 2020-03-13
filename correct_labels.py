import os
import time
import common
import datetime
import concurrency
from shutil import move
import multiprocessing as mp
from functools import partial


def correct_labels(args, task):
    logs = ""
    start = time.time()
    idx, batch = task
    model = common.fastai_load_model(args.model) if args.std_proc else mp.current_process().model
    for img_path in batch:
        cat = os.path.basename(os.path.dirname(img_path))
        pred = str(model.predict(common.fastai_load_and_prepare_img(img_path))[0])
        if cat != pred:
            f = os.path.basename(img_path)
            move(img_path, os.path.join(args.data, pred, f))
            logs += f"{f};{cat};{pred}\n"
    if idx % 100 == 0:
        pcid = str(os.getpid()) if args.std_proc else f"{mp.current_process().pcid} (pid {os.getpid()})"
        print(f"Process {pcid} completed task {idx} in {datetime.timedelta(seconds=time.time() - start)}.")
    return logs


def main(args, ctx=None):
    start = time.time()
    all_dirs = common.list_dirs(args.data)
    tasks = concurrency.batch_files_in_dirs(args.data, all_dirs, bs=args.proc_bs)
    pool = mp.Pool(processes=args.workers) if ctx is None else ctx.Pool(processes=args.workers)
    with open(args.log, 'w') as logger:
        logger.write('file;old_label;new_label\n')
        for correction in pool.imap_unordered(partial(correct_labels, args), zip(range(len(tasks)), tasks)):
            logger.write(correction)
    pool.close()
    pool.join()
    pool.terminate()
    print(f"Work completed in {datetime.timedelta(seconds=time.time() - start)}.")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Correct dataset labels")
    parser.add_argument('--data', type=str, required=True, help="source dataset to correct")
    parser.add_argument('--model', type=str, required=True, help="model path")

    parser.add_argument('--device', type=str, default="cpu", help="cpu or gpu")
    parser.add_argument('--proc_bs', type=int, default=100, help="Number of imgs per process")
    parser.add_argument('--workers', type=int, help="Number of process, will adjust to chosen device if needed")
    parser.add_argument('--log', type=str, help="log file where label corrections will be tracked")
    parser.add_argument('--std-proc', action='store_true', help="Use standard process instead of custom process")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    args = parser.parse_args()

    if args.device == "gpu":
        import torch
        if True or not torch.cuda.is_available():
            raise Exception("GPU mode not supported.")
        # TODO: each process use one GPU
        if args.workers is None:
            args.workers = torch.cuda.device_count()
        else:
            args.workers = min(torch.cuda.device_count(), args.workers)
    else:
        args.device = 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if args.workers is None:
            args.workers = mp.cpu_count()

    common.check_args(args)
    common.set_seeds(args.seed)
    if args.log is None:
        args.log = os.path.join(args.data, f'{datetime.datetime.now().strftime("%Y%m%d_%H%M")}_changes.txt')

    assert os.path.exists(args.model), f"Provided model path do not exist: {args.model}"

    if args.std_proc:
        mp_ctx = None
    else:
        # https://stackoverflow.com/questions/740844/python-multiprocessing-pool-of-custom-processes
        mp_ctx = mp.get_context()  # get the default context

        class CustomProcess(mp_ctx.Process):
            proc_id = 0

            def __init__(self, *argv, **kwargs):
                self.pcid = CustomProcess.proc_id
                self.model = common.fastai_load_model(args.model)
                super().__init__(*argv, **kwargs)
                CustomProcess.proc_id += 1
                print(f"Custom process {CustomProcess.proc_id} created.")

        mp_ctx.Process = CustomProcess  # override the context's Process

    main(args, mp_ctx)


