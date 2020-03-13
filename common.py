import os
import datetime
from pathlib import Path

def flatten(lst):
    """Flattens lst of lsts"""
    return [elem for sublst in lst for elem in sublst]


def maybe_create(*d):
    path = os.path.join(*d)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def list_dirs(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def get_exp_logdir(args):
    return f'{datetime.datetime.now().strftime("%Y%m%d_%H%M")}_{os.path.basename(args.data)}_{args.model}_lr{args.lr}' \
        + f'_bs{args.batch_size}_epo{args.epochs}_seed{args.seed}_world{args.nmachines * args.ngpus}_wd{args.wd}' \
        + f'_{args.exp_name}'

def get_root_logdir(logdir):
    if os.path.exists(logdir) and os.path.isdir(logdir):
        return logdir
    else:
        return os.path.join(str(Path.home()), 'logs')


def set_seeds(seed, cuda_seeded=False):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_seeded and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def check_args(args):
    args.data = args.data.rstrip('/')
    assert os.path.exists(args.data) and os.path.isdir(args.data), f"Provided dataset dir {args.data} invalid."


def set_gpu(gpuid):
    import torch
    if torch.cuda.is_available() and gpuid is not None:
        torch.cuda.set_device(gpuid)


def add_common_train_args(parser, lr=None, b=None, model=None):
    parser.add_argument('-name', '--exp-name', required=True, help='Custom string to append to experiment log dir')
    parser.add_argument('--lr', type=float, default=lr, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=26, help='Number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=b, type=int, help="Batch size")
    parser.add_argument('--logdir', type=str, help="Root directory where logs will be saved")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")
    parser.add_argument('--ngpus', type=int, default=1, help="Number of gpus per machines")
    parser.add_argument('--nmachines', type=int, default=1, help="Number of machines")
    parser.add_argument('--model', type=str, default=model, help="Model name")
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')


def add_obj_detec_args(parser):
    parser.add_argument('--obj-detec', action='store_true', help="if dataset is obj detec dataset")
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images if --obj-detec")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")


def fastai_load_model(model_path, radam=True):
    import fastai.vision as fvision
    if radam:
        from radam import RAdam
    return fvision.load_learner(os.path.dirname(model_path), os.path.basename(model_path))

def fastai_load_and_prepare_img(img_path):
    import fastai.vision as fvision
    import cv2
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    t = fvision.pil2tensor(im, dtype=im.dtype)  # converts to numpy tensor
    return fvision.Image(t.float() / 255.)  # Convert to float
