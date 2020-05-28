import os
import cv2
import math
import numpy as np
import multiprocessing as mp
import concurrency
import common


class Preprocessor:
    def __init__(self, source, dest, workers=None):
        self.source = source
        self.dest = dest
        self.workers = mp.cpu_count() - 2 if workers is None else workers
        self.items = self.collect_items()
        self.batches = self.create_batches()

    def apply(self, args):
        pmq, jobs = mp.Queue(), []
        for i, batch in zip(np.resize(range(self.workers), len(self.batches)), self.batches):
            jobs.append(mp.Process(target=self.preprocess, args=(i, pmq, batch, *args)))
            jobs[i].start()
        pms = concurrency.unload_mpqueue(pmq, jobs)
        for j in jobs:
            j.join()
        return pms

    def create_batches(self):
        """Divides items equally among workers. Number of batches = number of workers"""
        batch_size = math.ceil(len(self.items) / self.workers)
        return common.batch_list(self.items, batch_size)

    def preprocess(self, pid, pmq, batch, args):
        raise NotImplementedError

    def collect_items(self):
        raise NotImplementedError


class TestPreprocessor(Preprocessor):
    def __init__(self, w):
        super().__init__('NA', 'NA', w)

    def collect_items(self):
        return ["a", "b", "c"] * 30

    def preprocess(self, pid, pmq, batch):
        import time
        res = [b.upper() for b in batch]
        time.sleep(1*pid)
        pmq.put({pid: res})



if __name__ == '__main__':
    print("Hello World")
    test = TestPreprocessor(10)
    print(test.apply([]))
