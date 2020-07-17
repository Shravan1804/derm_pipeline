import common
import math
import multiprocessing as mp
import time
import queue


def batch_lst(files, bs=None, workers=None):
    workers = min(mp.cpu_count(), len(files)) if workers is None else workers
    bs = math.ceil(len(files) / workers) if bs is None else bs
    return workers, bs, common.batch_list(files, bs)


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
