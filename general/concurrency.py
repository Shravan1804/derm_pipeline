#!/usr/bin/env python

"""concurrency.py: File regrouping useful methods for concurrency tasks"""

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
import multiprocessing as mp
import time
import queue

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common


def add_multi_proc_args(parser):
    """Adds concurrency args: workers and bs
    :param parser: arg parser
    :return: modified argparser
    """
    parser.add_argument('--workers', type=int, default=8, help="Number of workers to use")
    parser.add_argument('--bs', type=int, help="Batch size per worker")
    return parser


def batch_lst(files, bs=None, workers=None):
    """Batch list according to the specified batch size or number of workers
    :param files: list, elements to be batched
    :param bs: int, batch size
    :param workers: int, number of workers, defaults to the number of cpus - 2
    :return: tuple, (nb workers, batch size, batched list)
    """
    n = len(files)
    if workers is None: workers = min(mp.cpu_count() - 2, n)
    if bs is None or bs * workers < n: bs = math.ceil(n / workers)
    return workers, bs, common.batch_list(files, bs)


def unload_mpqueue(pmq, processes):
    """Empties multiprocess queue
    source: https://stackoverflow.com/questions/31708646/process-join-and-queue-dont-work-with-large-numbers
    :param pmq: mpqueue, multiprocess queue to be emptied
    :param processes: list, processes
    :return: list, with elements from mp queue
    """
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


def multi_process_fn(workers, batches, fn):
    """Dispatches function to workers with batches as args
    :param workers: int, number of workers
    :param batches: list of batches
    :param fn: function, takes as input process id and batch
    """
    jobs = []
    for i, batch in zip(range(workers), batches):
        jobs.append(mp.Process(target=fn, args=(i, batch)))
        jobs[i].start()
    for j in jobs:
        j.join()

