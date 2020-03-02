import math
import multiprocessing as mp
import time
import queue

def batch_dirs(all_dirs):
    workers = min(mp.cpu_count(), len(all_dirs))
    batch_size = math.ceil(len(all_dirs) / workers)
    batched_dirs = [all_dirs[i:min(len(all_dirs), i + batch_size)] for i in range(0, len(all_dirs), batch_size)]
    return workers, batch_size, batched_dirs


def unload_mpqueue(pmq, processes):
    #https://stackoverflow.com/questions/31708646/process-join-and-queue-dont-work-with-large-numbers
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
