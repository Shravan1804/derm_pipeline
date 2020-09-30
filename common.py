import os
import cv2
import time
import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def flatten(lst):
    """Flattens lst of lsts"""
    return [elem for sublst in lst for elem in sublst]


def most_common(arr, top=3, return_index=False, return_counts=False):
    u, c = np.unique(arr, return_counts=True)
    sorted_c = c.argsort()[::-1]
    res = u[sorted_c[:top]]
    if return_index:
        res = res, sorted_c[:top]
    if return_counts:
        res = *res, c[res[-1]]
    return res

def maybe_create(*d):
    """Receives arbitrary number of dirnames, joins them and create them if they don't exist. Returns joined path."""
    path = os.path.join(*d)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def reproduce_dir_structure(source, dest):
    """Reproduce dir structure of source in dest. Raise exception if source invalid. Creates dest if needed."""
    check_dir_valid(source)
    maybe_create(dest)
    for d in list_dirs(source):
        reproduce_dir_structure(os.path.join(source, d), os.path.join(dest, d))


def list_files(root, full_path=False, posix_path=False, recursion=False, max_rec_level=-1):
    """Return the list of files, if recursion stops at max_rec_level unless negative then goes all levels."""
    lf = [os.path.join(root, f) if full_path else f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    if recursion and max_rec_level != 0:
        for d in list_dirs(root, full_path, posix_path, recursion, max_rec_level-1):
            full_d = d if full_path else os.path.join(root, d)
            lf.extend([os.path.join(d, f) for f in os.listdir(full_d) if os.path.isfile(os.path.join(full_d, f))])
    return [Path(i) for i in sorted(lf)] if posix_path else sorted(lf)


def list_dirs(root, full_path=False, posix_path=False, recursion=False, max_rec_level=-1, rec_level=0):
    """Return the list of dirs, if recursion stops at max_rec_level unless negative then goes all levels.
    rec_level is used for the recursion and should not be set"""
    if recursion and rec_level > max_rec_level >= 0:
        return []
    ld = [os.path.join(root, d) if full_path else d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if recursion:
        ld_rec = []
        for d in ld:
            new_root = d if full_path else os.path.join(root, d)
            children = list_dirs(new_root, full_path, posix_path, recursion, max_rec_level, rec_level+1)
            ld_rec.extend([c if full_path else os.path.join(d, c) for c in children])
        ld += ld_rec
    return [Path(d) for d in sorted(ld)] if posix_path else sorted(ld)


def list_files_in_dirs(root, full_path=False, posix_path=False):
    lf = [os.path.join(root, d, f) if full_path else f for d in list_dirs(root, full_path)
            for f in list_files(os.path.join(root, d))]
    return [Path(i) for i in lf] if posix_path else lf


def batch_list(lst, bs):
    return [lst[i:min(len(lst), i + bs)] for i in range(0, len(lst), bs)]


def now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def get_exp_logdir(args, custom=''):
    ws = args.num_machines * args.num_gpus
    d = f'{now()}_{custom}_{args.model}_bs{args.batch_size}'
    if args.lr:
        d += f'_lr{args.lr}'
    if args.wd:
        d += f'_wd{args.wd}'
    d += f'_epo{args.epochs}_seed{args.seed}_world{ws}_{args.exp_name}'
    return d

def get_root_logdir(logdir):
    if logdir is not None and os.path.exists(logdir) and os.path.isdir(logdir):
        return logdir
    else:
        return os.path.join(str(Path.home()), 'logs')


def set_seeds(seed, cuda_seeded=False):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_seeded and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def check_file_valid(filepath):
    assert os.path.exists(filepath) and os.path.isfile(filepath), f"Provided file {filepath} invalid."


def check_dir_valid(dirpath):
    assert os.path.exists(dirpath) and os.path.isdir(dirpath), f"Provided dir {dirpath} invalid."


def maybe_set_gpu(gpuid, num_gpus):
    if num_gpus != 1:
        print("Warning cannot fix more than 1 gpus, requested", num_gpus)
    else:
        import torch
        if torch.cuda.is_available() and gpuid is not None:
            torch.cuda.set_device(gpuid)


def add_multi_gpus_args(parser):
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)


def add_common_train_args(parser, pdefaults=dict(), phelp=dict()):
    parser.add_argument('-name', '--exp-name', required=True, help='Custom string to append to experiment log dir')
    parser.add_argument('--logdir', type=str, default=pdefaults.get('logdir', get_root_logdir(None)), help="Root directory where logs will be saved, default to $HOME/logs")
    parser.add_argument('--exp-logdir', type=str, help="Experiment logdir, will be created in root log dir")
    parser.add_argument('--model', type=str, default=pdefaults.get('model', None), help=phelp.get('model', "Model name"))

    parser.add_argument('--seed', type=int, default=pdefaults.get('seed', 42), help="Random seed")
    parser.add_argument('--epochs', type=int, default=pdefaults.get('epo', 26), help='Number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=pdefaults.get('bs', 6), type=int, help="Batch size")
    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")

    parser.add_argument('--wd', default=pdefaults.get('wd', None), type=float, help='weight decay')
    parser.add_argument('--lr', type=float, default=pdefaults.get('lr', None), help=phelp.get('lr', 'Learning rate'))
    parser.add_argument('--lr-steps', default=pdefaults.get('lr-steps', [8, 11]), nargs='+', type=int, help='decrease lr every step-size epochs')


def add_classif_args(parser):
    parser.add_argument('--classif', action='store_true', help="if dataset is classif dataset")


def add_obj_detec_args(parser):
    parser.add_argument('--obj-detec', action='store_true', help="if dataset is obj detec dataset")
    parser.add_argument('--img-dir', type=str, default='images', help="dir containing images if --obj-detec")
    parser.add_argument('--mext', type=str, default='.png', help="masks file extension")


def add_multi_proc_args(parser):
    parser.add_argument('--workers', type=int, help="Number of workers to use")
    parser.add_argument('--bs', type=int, help="Batch size per worker")


def load_custom_pretrained_weights(model, weights_path):
    import torch
    new_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
    model_state_dict = model.state_dict()
    for name, param in model_state_dict.items():
        if name in new_state_dict:
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                print('Shape mismatch at:', name, 'skipping')
        else:
            print(f'{name} weight of the model not in pretrained weights')
    model.load_state_dict(model_state_dict)


def fastai_load_model(model_params, radam=True):
    import fastai.vision as fvision
    if radam:
        from radam import RAdam
    return fvision.load_learner(**model_params)

def fastai_load_and_prepare_img(img_path):
    import fastai.vision as fvision
    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    t = fvision.pil2tensor(im, dtype=im.dtype)  # converts to numpy tensor
    return fvision.Image(t.float() / 255.)  # Convert to float


def time_method(m, args, *d):
    start = time.time()
    m(args, *d)
    print(f"Work completed in {datetime.timedelta(seconds=time.time() - start)}.")


def plt_save_fig(path, dpi=300, close=True):
    plt.savefig(path, dpi=dpi)
    if close:
        plt.close()


def load_rgb_img(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)


def plt_show_img(im, title, show=True, save_path=None):
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(title)
    if save_path is not None:
        plt_save_fig(save_path, close=False)
    if show:
        fig.show()
