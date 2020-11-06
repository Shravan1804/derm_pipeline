import os
import cv2
import time
import datetime
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit


body_loc_trad = {'arme': 'Arm', 'beine': 'Leg', 'fusse': 'Feet', 'hande': 'Hand', 'kopf': 'Head', 'other': 'Other',
                 'stamm': 'Trunk', 'mean': 'Mean'}


def get_cmap(n, name='Dark2'):
    ''' source https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def grouped_barplot(ax, vars_vals, var_labels, group_labels, barWidth=.25):
    """vars_vals: lst of variables values (lst, one item per group)"""
    assert len(vars_vals) == len(var_labels)
    assert np.array([len(v) == len(group_labels) for v in vars_vals]).all()
    ax.axis('on')
    cmap = get_cmap(len(vars_vals))
    pos = np.arange(len(group_labels))
    for var_vals, var_label, c in zip(vars_vals, var_labels, [cmap(i) for i in range(len(vars_vals))]):
        ax.bar(pos, var_vals, color=c, width=barWidth, edgecolor='white', label=var_label)
        pos = [x + barWidth for x in pos]
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Percentage', fontweight='bold')
    ax.set_xticks([r + barWidth for r in range(len(group_labels))])
    ax.set_xticklabels(group_labels)
    ax.legend()


def flatten(lst):
    """Flattens lst of lsts"""
    return [elem for sublst in lst for elem in sublst]


def merge(lst, cond_fn, merge_fn):
    """Applies merge_fn on pairs of lst items if cond_fn is satisfied"""
    for i, item_a in enumerate(lst):
        sub_lst = lst[:i] + lst[i+1:]
        for j, item_b in enumerate(sub_lst):
            if cond_fn(item_a, item_b):
                new_lst = sub_lst[:j] + sub_lst[j+1:] + [merge_fn(item_a, item_b)]
                return merge(new_lst, cond_fn, merge_fn)
    return lst


def int_to_bins(n, bins, rand=False):
    """Divides integer into bins. If rand true, random split else equal split"""
    if n <= 0 or bins <= 0:
        return np.array([])
    if rand:
        temp = np.concatenate([np.zeros(n, dtype=np.bool), np.ones(bins-1, dtype=np.bool)])
        np.random.shuffle(temp)
        return np.array([(~t).sum() for t in np.split(temp, temp.nonzero()[0])])
    else:
        return np.arange(n+bins-1, n-1, -1) // bins


def most_common(arr, top=3, return_index=False, return_counts=False):
    """Returns most common elements in array"""
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


def get_exp_logdir(args, custom='', show_train_hyp=False):
    ws = args.num_machines * args.num_gpus
    d = f'{now()}_{custom}_{args.model}_bs{args.bs}'
    if args.cross_val:
        d += f'_CV{args.nfolds}'
    if show_train_hyp:
        if args.lr:
            d += f'_lr{args.lr}'
        if args.wd:
            d += f'_wd{args.wd}'
    d += f'_epo{args.epochs}_seed{args.seed}_world{ws}_{args.exp_name}'
    return d


def zero_pad(it, max_it):
    return str(it).zfill(len(str(max_it)) + 1)


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
    if gpuid is not None and num_gpus != 1:
        print("Warning cannot fix more than 1 gpus, requested", num_gpus)
    else:
        import torch
        if torch.cuda.is_available() and gpuid is not None:
            torch.cuda.set_device(gpuid)


def add_multi_gpus_args(parser):
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)


def add_common_train_args(parser, pdef=dict(), phelp=dict()):
    parser.add_argument('-name', '--exp-name', required=True, help='Custom string to append to experiment log dir')
    parser.add_argument('--logdir', type=str, default=pdef.get('--logdir', get_root_logdir(None)),
                        help=phelp.get('--logdir', "Root directory where logs will be saved, default to $HOME/logs"))
    parser.add_argument('--exp-logdir', type=str, help="Experiment logdir, will be created in root log dir")
    parser.add_argument('--model', type=str, default=pdef.get('--model', None), help=phelp.get('--model', "Model name"))

    parser.add_argument('--seed', type=int, default=pdef.get('--seed', 42), help="Random seed")
    parser.add_argument('--epochs', type=int, default=pdef.get('--epochs', 26), help='Number of total epochs to run')
    parser.add_argument('--bs', default=pdef.get('--bs', 6), type=int, help="Batch size")
    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")

    parser.add_argument('--wd', default=pdef.get('--wd', None), type=float, help='weight decay')
    parser.add_argument('--lr', type=float, default=pdef.get('--lr', None), help=phelp.get('--lr', 'Learning rate'))
    parser.add_argument('--lr-steps', default=pdef.get('--lr-steps', [8, 11]), nargs='+', type=int,
                        help='decrease lr every step-size epochs')


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


def get_data_fn(args, full_img_sep, stratify):
    get_splits = partial(split_data, full_img_sep=full_img_sep, stratify=stratify, seed=args.seed,
                         cross_val=args.cross_val, nfolds=args.nfolds, valid_size=args.valid_size)
    get_dls = partial(progressive_resizing_dls, bs=args.bs, input_size=args.input_size, num_gpus=args.num_gpus,
                      progr_size=args.progr_size, factors=args.factors)
    return get_splits, get_dls


def get_full_img_dict(images, sep):
    """Returns a dict with keys the full images names and values the lst of corresponding images.
    sep is the string which separates the full img names"""
    full_images_dict = {}
    for fpath in images:
        cls = os.path.basename(os.path.dirname(fpath))
        file, ext = os.path.splitext(os.path.basename(fpath))
        fi = os.path.join(cls, f'{file.split(sep)[0] if sep in file else file}{ext}')
        if fi in full_images_dict:
            full_images_dict[fi].append(fpath)
        else:
            full_images_dict[fi] = [fpath]
    return full_images_dict


def split_data(images, full_img_sep, stratify, seed=42, cross_val=False, nfolds=5, valid_size=.2):
    np.random.seed(seed)

    full_images_dict = get_full_img_dict(images, full_img_sep, with_cls=stratify)
    full_images = np.array(list(full_images_dict.keys()))
    full_images_cls = np.array([os.path.dirname(f) for f in full_images])

    cv_splitter, no_cv_splitter = (StratifiedKFold, StratifiedShuffleSplit) if stratify else (KFold, ShuffleSplit)
    splitter = cv_splitter(n_splits=nfolds, shuffle=True, random_state=seed) if cross_val else \
        no_cv_splitter(n_splits=1, test_size=valid_size, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(full_images, full_images_cls)):
        if cross_val:
            print("FOLD:", fold)
        train_images = [i for fi in full_images[train_idx] for i in full_images_dict[fi]]
        valid_images = [i for fi in full_images[valid_idx] for i in full_images_dict[fi]]
        np.random.shuffle(train_images)
        np.random.shuffle(valid_images)
        yield fold, train_images, valid_images


def progressive_resizing_dls(dls_fn, max_bs, bs, input_size, num_gpus, progr_size, factors):
    input_sizes = [int(input_size * f) for f in factors] if progr_size else [input_size]
    batch_sizes = [max(1, min(int(bs / f / f) * num_gpus, max_bs) // 2 * 2) for f in factors] if progr_size else [bs]
    for it, (bs, size) in enumerate(zip(batch_sizes, input_sizes)):
        run = f'{zero_pad(it, len(batch_sizes))}_{size}px_bs{bs}'
        print(f"Iteration {it}: running {run}")
        yield it, run, dls_fn(bs, size)


def img_bgr_to_rgb(im):
    if len(im.shape) != 3:
        raise Exception(f"Error cannot convert from bgr to rgb, im shape is {im.shape}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_rgb_img(path):
    return img_bgr_to_rgb(cv2.imread(path, cv2.IMREAD_UNCHANGED))


def prepare_img_axs(h_w_ratio, nrows, ncols, figsize_fact=8, no_axis=True):
    base_figsize = (ncols*figsize_fact, nrows*figsize_fact*h_w_ratio)
    fig, axs = plt.subplots(nrows, ncols, figsize=base_figsize)
    axs = axs.flatten()
    if no_axis:
        for ax in axs:
            ax.axis('off')
    return fig, axs


def img_on_ax(im, ax, title=None):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis('off')


def plt_show_img(im, title, show=True, save_path=None):
    fig, ax = plt.subplots()
    img_on_ax(im, ax, title=title)
    if save_path is not None:
        plt_save_fig(save_path, close=False)
    if show:
        fig.show()


def acc(TP, TN, FP, FN, epsilon=1e-8):
    return (TP + TN) / (TP + TN + FP + FN + epsilon)


def prec(TP, TN, FP, FN, epsilon=1e-8):
    return TP / (TP + FP + epsilon)


def rec(TP, TN, FP, FN, epsilon=1e-8):
    return TP / (TP + FN + epsilon)
