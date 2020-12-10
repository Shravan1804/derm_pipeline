import os
import time
import common
import datetime
import numpy as np
import concurrency
from shutil import move
import multiprocessing as mp
from functools import partial


def maybe_move_file(data_path, img_path, cat, pred):
    if cat != pred:
        f = os.path.basename(img_path)
        move(img_path, os.path.join(data_path, pred, f))
        return f'{f};{cat};{pred}\n'
    return ''


def gpu_correct_labels(data_path, learner, log_path=None):
    if log_path is None:
        log_path = default_logfile(data_path)
    import fastai.vision as fv
    file_labels = [(str(p), os.path.basename(os.path.dirname(p))) for p in learner.data.test_ds.items]
    preds, _ = learner.get_preds(ds_type=fv.DatasetType.Test)
    preds = [learner.data.classes[p] for p in np.argmax(preds.numpy(), 1)]
    changes = [maybe_move_file(data_path, file_labels[i][0], file_labels[i][1], p) for i, p in enumerate(preds)]
    log_changes(log_path, changes)


def cpu_correct_labels(args, task):
    logs = ""
    start = time.time()
    idx, batch = task
    p = {'path': os.path.dirname(args.model), 'file': os.path.basename(args.model)}
    model = common.fastai_load_model(p) if args.std_proc else mp.current_process().model
    for img_path in batch:
        cat = os.path.basename(os.path.dirname(img_path))
        pred = str(model.predict(common.fastai_load_and_prepare_img(img_path))[0])
        logs += maybe_move_file(args.data, img_path, cat, pred)
    if idx % 100 == 0:
        pcid = str(os.getpid()) if args.std_proc else f"{mp.current_process().pcid} (pid {os.getpid()})"
        print(f"Process {pcid} completed task {idx} in {datetime.timedelta(seconds=time.time() - start)}.")
    return logs


def default_logfile(data_path):
    return os.path.join(data_path, f'{datetime.datetime.now().strftime("%Y%m%d_%H%M")}_changes.txt')


def log_changes(log_file_path, changes):
    with open(log_file_path, 'w') as logger:
        logger.write('file;old_label;new_label\n')
        for correction in changes:
            logger.write(correction)


def main(args, ctx=None):
    start = time.time()
    if args.device == "gpu":
        import fastai.vision as fv
        p = {'path': os.path.dirname(args.model), 'file': os.path.basename(args.model), 'bs': args.gpu_bs * args.ngpus,
             'test': fv.ImageList.from_folder(args.model)}
        learner = common.fastai_load_model(p)
        if args.ngpus > 1:
            learner.model = torch.nn.DataParallel(learner.model, device_ids=list(range(args.ngpus)))
        gpu_correct_labels(args.data, learner, args.log)
    else:
        files = common.list_files(args.data, full_path=True, recursion=True, max_rec_level=1)
        _, _, tasks = concurrency.batch_lst(files, bs=args.proc_bs)
        pool = mp.Pool(processes=args.workers) if ctx is None else ctx.Pool(processes=args.workers)
        log_changes(args.log, pool.imap_unordered(partial(cpu_correct_labels, args), zip(range(len(tasks)), tasks)))
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
    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")
    parser.add_argument('--ngpus', type=int, default=1, help="Number of gpus per machines")
    parser.add_argument('--gpu_bs', type=int, default=64, help="Number of imgs per gpu")
    parser.add_argument('--proc_bs', type=int, default=100, help="Number of imgs per process")
    parser.add_argument('--workers', type=int, help="Number of process, will adjust to chosen device if needed")
    parser.add_argument('--log', type=str, help="log file where label corrections will be tracked")
    parser.add_argument('--std-proc', action='store_true', help="Use standard process instead of custom process")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    args = parser.parse_args()

    if args.device == "gpu":
        import torch
        assert torch.cuda.is_available(), "CUDA not available, please run on CPU"
        common.maybe_set_gpu(args.gpuid, args.ngpus)
    else:
        args.device = 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if args.workers is None:
            args.workers = mp.cpu_count()

    common.check_args(args)
    common.set_seeds(args.seed)
    if args.log is None:
        args.log = default_logfile(args.data)

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
                self.model = common.fastai_load_model({'path': os.path.dirname(args.model),
                                                       'file': os.path.basename(args.model)})
                super().__init__(*argv, **kwargs)
                CustomProcess.proc_id += 1
                print(f"Custom process {CustomProcess.proc_id} created.")

        mp_ctx.Process = CustomProcess  # override the context's Process

    main(args, mp_ctx)

