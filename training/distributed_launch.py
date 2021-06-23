#!/usr/bin/env python

"""distributed_launch.py: Script used to launch training in distributed mode"""

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
import subprocess
from fastai.basics import *
from fastcore.script import *

@call_parse
def main(
    gpus:Param("The GPUs to use for distributed training", str)='all',
    encrypted:Param("Is the data encrypted", store_true)=False,
    script:Param("Script to run", str, opt=False)='',
    args:Param("Args to pass to script", nargs='...', opt=False)=''
):

    "PyTorch distributed training launch helper that spawns multiple distributed processes"
    # Loosely based on torch.distributed.launch
    current_env = os.environ.copy()
    gpus = list(range(torch.cuda.device_count())) if gpus=='all' else list(gpus)
    current_env["WORLD_SIZE"] = str(len(gpus))
    current_env["MASTER_ADDR"] = '127.0.0.1'
    current_env["MASTER_PORT"] = '29500'

    if encrypted:
        print("Please provide datasets encryption key")
        current_env["CRYPTO_KEY"] = input()
        args.append('--encrypted')

    processes = []
    gpus = [str(gpu) for gpu in gpus]
    for i, gpu in enumerate(gpus):
        current_env["RANK"] = str(i)
        cmd = [sys.executable, "-u", script, "--proc-gpu", gpu, "--gpu-ids", *gpus] + args
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes: process.wait()

