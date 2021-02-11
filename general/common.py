import os
import sys
import cv2
import math
import datetime
import itertools
import contextlib
from timeit import default_timer
from string import ascii_uppercase
from collections import defaultdict
from pathlib import Path, PosixPath


import numpy as np
import PIL.Image as PImage
import matplotlib.pyplot as plt


body_loc_trad = {'arme': 'Arm', 'beine': 'Leg', 'fusse': 'Feet', 'hande': 'Hand', 'kopf': 'Head', 'other': 'Other',
                 'stamm': 'Trunk', 'mean': 'Mean'}


def get_cmap(n, name='Dark2'):
    ''' source https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def clip_err(vals, err, bounds=(0, 1)):
    """If err not None, will return 2D err array (low_err and up_err) clipped between specified bounds"""
    if err is None: return np.zeros((2, *vals.shape), dtype=vals.dtype)
    return (np.vstack([vals - err, vals + err]).clip(*bounds) - np.vstack([vals, vals])) * np.array([[-1], [1]])


def get_error_display_params():
    return {'elinewidth': .8, 'capsize': 2, 'capthick': .5}


def show_graph_values(ax, values, pos_x, pos_y=None, yerr=None, kwargs={"fontsize": 6}):
    pos_y = values if pos_y is None else pos_y
    yerr = np.zeros(values.shape) if yerr is None else yerr
    for x, y, v, ye in zip(pos_x, pos_y, values, yerr):
        ax.text(x, y + ye, f'{v:.2f}', **kwargs)


def plot_lines_with_err(ax, xs, ys, errs, labels, show_val=False, err_bounds=(0, 1), legend_loc="lower center"):
    for x, y, err, label in zip(xs, ys, errs, labels):
        err = clip_err(y, err, err_bounds)
        ax.errorbar(x, y, yerr=err, label=label, **get_error_display_params())
        if show_val: show_graph_values(ax, y, x, yerr=err[1])
    ax.legend(loc=legend_loc)


def grouped_barplot_with_err(ax, stats, groupLabels, xlabel=None, ylabel=None, title=None, barwidth=.35, show_val=False,
                             title_loc=None, legend_loc="lower center", err_bounds=(0, 1)):
    """Stats is a dict with keys the bar name (e.g. accuracy) and values the values for each group (e.g. categories)"""
    nbars_per_group, ngroups = len(stats.keys()), len(groupLabels)
    group_width = nbars_per_group * barwidth * 4/3
    positions = np.arange(0, ngroups * group_width, group_width)
    offsets = [(i - nbars_per_group / 2) * barwidth + barwidth / 2 for i in range(nbars_per_group)]

    ekw = get_error_display_params()
    cmap = get_cmap(nbars_per_group)
    for offset, (key, (vals, err)), c in zip(offsets, stats.items(), [cmap(i) for i in range(nbars_per_group)]):
        err = clip_err(vals, err, err_bounds)
        ax.bar(positions + offset, vals, color=c, width=barwidth, yerr=err, label=key, error_kw=ekw, edgecolor='white')
        if show_val: show_graph_values(ax, vals, positions + offset - barwidth/2, yerr=err[1])

    ax.set_ylabel("Performance" if ylabel is None else ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(groupLabels, rotation=45)
    ax.legend(loc=legend_loc)
    if title is not None:
        ax.set_title(title, loc=title_loc)
    ax.grid(False)


def plot_confusion_matrix(ax, cm_std, labels, title=None, title_loc=None, ):
    cm, std = cm_std
    ax.imshow(cm, interpolation='nearest', cmap="Blues")

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, rotation=0)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        coeff = f'{cm[i, j]:.{2}f} \u00B1 {std[i, j]:.{2}f}'
        ax.text(j, i-.15, f'{cm[i, j]:.{2}f}', horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)
        ax.text(j, i+.15, f'\u00B1 {std[i, j]:.{2}f}', horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=6)

    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    if title is not None:
        ax.set_title(title, loc=title_loc)
    ax.grid(False)


def recursive_default_dict():
    return lambda: defaultdict(recursive_default_dict)


def flatten(lst):
    """Flattens lst of lsts"""
    return [elem for sublst in lst for elem in sublst]


def is_path(item):
    return type(item) in (str, PosixPath, Path)


def merge(lst, cond_fn, merge_fn):
    """Applies merge_fn on pairs of lst items if cond_fn is satisfied"""
    for i, item_a in enumerate(lst):
        sub_lst = lst[:i] + lst[i+1:]
        for j, item_b in enumerate(sub_lst):
            if cond_fn(item_a, item_b):
                new_lst = sub_lst[:j] + sub_lst[j+1:] + [merge_fn(item_a, item_b)]
                return merge(new_lst, cond_fn, merge_fn)
    return lst


def decimal_to_base(n, base, ndigits=-1):
    """Converts n from base 10 to specified base"""
    res, count = [], 0
    while(n > 0):
        rem = int(n % base)
        res = [rem] + res
        n = (n-rem)/base
        count += 1
    if ndigits != -1:
        assert count <= ndigits, f"{int(n)} cannot be converted in base {base} with only {ndigits} digits."
        while count+1 <= ndigits:
            res = [0] + res
            count += 1
    elif count == 0: res = [0]
    return res


def generate_codes(ncodes, alphabet=ascii_uppercase, size=None):
    n_symbols = len(set(ascii_uppercase))
    min_symbols = int(math.log(ncodes, len(alphabet))) + 1
    if size is None: size = min_symbols
    if size != -1:
        error_msg = f"Error, requested {ncodes} different codes but an alphabet with only {n_symbols}" \
                    f"different symbols was provided (require at least {min_symbols} different symbols)."
        assert size >= min_symbols, error_msg
    return ["".join([alphabet[i] for i in decimal_to_base(c, len(alphabet), ndigits=size)]) for c in range(ncodes)]


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


def list_images(root, **kwargs):
    return [f for f in list_files(root, **kwargs) if os.path.splitext(f)[1] in ('.jpg', '.png', '.tiff')]


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
    lf = [os.path.join(d, f) for d in list_dirs(root, full_path)
          for f in list_files(d if full_path else os.path.join(root, d))]
    return [Path(i) for i in lf] if posix_path else lf


def batch_list(lst, bs):
    return [lst[i:min(len(lst), i + bs)] for i in range(0, len(lst), bs)]


def now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def zero_pad(it, max_it):
    return str(it).zfill(len(str(max_it)) + 1)


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


@contextlib.contextmanager
def mynullcontext(enter_result=None):     # 3.6 alternative to contextlib.nullcontext()
    yield enter_result


@contextlib.contextmanager
def elapsed_timer():
    # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python?page=2&tab=active#tab-top
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def time_method(m, *args, **kwargs):
    with elapsed_timer() as elapsed:
        m(*args, **kwargs)
        print(f"Work completed in {datetime.timedelta(seconds=elapsed())}.")


class PrintPrepender:
    # https://stackoverflow.com/questions/58866481/how-could-i-override-pythons-print-function-to-prepend-some-arbitrary-text-to-e
    stdout = sys.stdout

    def __init__(self, text_to_prepend):
        self.text_to_prepend = text_to_prepend
        self.buffer = [self.text_to_prepend]

    def write(self, text):
        lines = text.splitlines(keepends=True)
        for line in lines:
            self.buffer.append(line)
            self.flush()
            if line.endswith(os.linesep):
                self.buffer.append(self.text_to_prepend)

    def flush(self, *args):
        self.stdout.write(''.join(self.buffer))
        self.stdout.flush()
        self.buffer.clear()


def stdout_prepend(f, pre_msg, *args):
    with contextlib.redirect_stdout(PrintPrepender(pre_msg)):
        f(*args)


def plt_save_fig(path, fig=None, close=True, **kwargs):
    if fig is None:
        plt.savefig(path, **kwargs)
        if close: plt.close(plt.gcf())
    else:
        fig.savefig(path, **kwargs)
        if close: plt.close(fig)


def quick_img_size(img_path):
    """Returns height, width of img, quicker than cv2 since it does not load the image in memory"""
    width, height = PImage.open(img_path).size
    return height, width


def img_bgr_to_rgb(im):
    if len(im.shape) != 3:
        raise Exception(f"Error cannot convert from bgr to rgb, im shape is {im.shape}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_img(path):
    if is_path(path) and type(path) is not str: path = str(path)
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img_bgr_to_rgb(im) if len(im.shape) > 2 else im


def save_img(im, path):
    """If im has more than two channels, assumes it is an rgb image thus will convert it to bgr before using imwrite"""
    cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if len(im.shape) > 2 else im)


def prepare_img_axs(h_w_ratio, nrows, ncols, figsize_fact=8, no_axis=True, flatten=True, title=None):
    base_figsize = (ncols*figsize_fact, nrows*figsize_fact*h_w_ratio)
    plt.rcParams['font.size'] = max(base_figsize) * .8
    fig, axs = plt.subplots(nrows, ncols, figsize=base_figsize)
    nd = nrows > 1 or ncols > 1
    if no_axis:
        for ax in axs.flatten() if nd else [axs]:
            ax.axis('off')
    if nd and flatten: axs = axs.flatten()
    if title is not None: fig.suptitle(title, fontsize=base_figsize[0] * 1.4)
    return fig, axs


def img_on_ax(im, ax, title=None):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis('off')


def plt_show_img(im, title="", show=True, save_path=None):
    fig, ax = plt.subplots()
    img_on_ax(im, ax, title=title)
    if save_path is not None:
        plt_save_fig(save_path, close=False)
    if show:
        fig.show()

