import os
import math
import multiprocessing as mp
import time
import queue

def batch_list(lst, bs):
    return [lst[i:min(len(lst), i + bs)] for i in range(0, len(lst), bs)]

def batch_dirs(all_dirs, workers=None):
    workers = min(mp.cpu_count(), len(all_dirs)) if workers is None else workers
    batch_size = math.ceil(len(all_dirs) / workers)
    return workers, batch_size, batch_list(all_dirs, batch_size)


def batch_files_in_dirs(root, dirs, bs):
    fs = [os.path.join(root, d, f) for d in dirs for f in os.listdir(os.path.join(root, d))
          if os.path.isfile(os.path.join(root, d, f))]
    return batch_list(fs, bs)


def unload_mpqueue(pmq, processes):
    # https://stackoverflow.com/questions/31708646/process-join-and-queue-dont-work-with-large-numbers
    pms = []
    liveprocs = list(processes)
    while liveprocs:
        try:
            while 1:
                pms.extend(pmq.get(False))
        except queue.Empty:
            pass
        time.sleep(.5)  # Give tasks a chance to put more data in
        if not pmq.empty():
            continue
        liveprocs = [p for p in liveprocs if p.is_alive()]
    return pms
