import os
import sys
import subprocess
from fastai.basics import *
from fastcore.script import *

@call_parse
def main(
    gpus:Param("The GPUs to use for distributed training", str)='all',
    encrypted:Param("Is the data encrypted", store_true)=False,
    debug:Param("All subprocess stdout will be ", store_true)=False,
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
    for i, gpu in enumerate(gpus):
        current_env["RANK"] = str(i)
        cmd = [sys.executable, "-u", script, f"--proc-gpu={gpu}", f"--num-gpus={gpus}"] + args
        process = subprocess.Popen(cmd, env=current_env,
                                   stdout=subprocess.DEVNULL if i != 0 and not debug else None,
                                   stderr=subprocess.DEVNULL if i != 0 and not debug else None)
        processes.append(process)

    for process in processes: process.wait()

