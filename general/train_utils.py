import os
import sys
from pathlib import Path
from functools import partial

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

import torch
import fastai.vision.all as fv
import fastai.distributed as fd   # needed for fastai multi gpu
import fastai.callback.tensorboard as fc

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto, img_utils


def common_train_args(parser, pdef=dict(), phelp=dict()):
    parser.add_argument('--data', type=str, required=True, help="Root dataset dir")
    parser.add_argument('--use-wl', action='store_true', help="Data dir contains wl and sl data")
    parser.add_argument('--wl-train', type=str, default='weak_labels', help="weak labels (wl) dir")
    parser.add_argument('--sl-train', type=str, default='strong_labels_train', help="strong labels (sl) dir")
    parser.add_argument('--sl-test', type=str, default='strong_labels_test', help="sl test dir")
    parser.add_argument('--cats', type=str, nargs='+', default=pdef.get('--cats', None),
                        help=phelp.get('--cats', "Categories"))

    parser.add_argument('-name', '--exp-name', required=True, help='Custom string to append to experiment log dir')
    parser.add_argument('--logdir', type=str, default=pdef.get('--logdir', get_root_logdir(None)),
                        help=phelp.get('--logdir', "Root directory where logs will be saved, default to $HOME/logs"))
    parser.add_argument('--exp-logdir', type=str, help="Experiment logdir, will be created in root log dir")

    parser.add_argument('--encrypted', action='store_true', help="Data is encrypted")
    parser.add_argument('--user-key', type=str, help="Data encryption key")

    parser.add_argument('--cross-val', action='store_true', help="Perform 5-fold cross validation on sl train set")
    parser.add_argument('--nfolds', default=5, type=int, help="Number of folds for cross val")
    parser.add_argument('--valid-size', default=.2, type=float, help='If no cross val, splits train set with this %')

    parser.add_argument('--model', type=str, default=pdef.get('--model', None), help=phelp.get('--model', "Model name"))
    parser.add_argument('--bs', default=pdef.get('--bs', 6), type=int, help="Batch size")
    parser.add_argument('--epochs', type=int, default=pdef.get('--epochs', 26), help='Number of total epochs to run')

    parser.add_argument('--no-norm', action='store_true', help="Do not normalizes images to imagenet stats")
    parser.add_argument('--full-precision', action='store_true', help="Train with full precision (more gpu memory)")
    parser.add_argument('--early-stop', action='store_true', help="Early stopping during training")

    parser.add_argument('--gpu', type=int, required=True, help="Id of gpu to be used by script")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="number of machines")

    parser.add_argument('--seed', type=int, default=pdef.get('--seed', 42), help="Random seed")


def common_img_args(parser, pdef=dict(), phelp=dict()):
    parser.add_argument('--input-size', default=pdef.get('--input-size', 512), type=int,
                        help=phelp.get('--input-size', "Model input will be resized to this value"))
    parser.add_argument('--progr-size', action='store_true',
                        help=phelp.get('--progr-size', "Applies progressive resizing"))
    parser.add_argument('--size-facts', default=pdef.get('--size-facts', [.25, .5, .75,  1]), nargs='+', type=float,
                        help=phelp.get('--size-facts', 'Increase progressive size factors'))


def get_exp_logdir(args, image_data):
    ws = args.num_machines * args.num_gpus
    d = f'{common.now()}_{args.model}_bs{args.bs}_epo{args.epochs}_seed{args.seed}_world{ws}'
    d += '' if args.no_norm else '_normed'
    d += '_fp32' if args.full_precision else '_fp16'
    d += f'_CV{args.nfolds}' if args.cross_val else f'_noCV_valid{args.valid_size}'
    d += '_WL' if args.use_wl else '_SL'
    if image_data:
        d += f'_input{args.input_size}'
        d += f'_progr-size{"_".join(map(str, args.size_facts))}' if args.progr_size else ""
    d += f'_{args.exp_name}'
    return d


def get_root_logdir(logdir):
    if logdir is not None and os.path.exists(logdir) and os.path.isdir(logdir):
        return logdir
    else:
        return os.path.join(str(Path.home()), 'logs')


def get_data_path(args, weak_labels=False):
    if args.use_wl:
        return os.path.join(args.data, args.wl_train if weak_labels else args.sl_train)
    else:
        return args.data


def get_data_fn(args, full_img_sep, stratify):
    get_splits = partial(split_data, full_img_sep=full_img_sep, stratify=stratify, seed=args.seed,
                         cross_val=args.cross_val, nfolds=args.nfolds, valid_size=args.valid_size)
    get_dls = partial(progressive_resizing_dls, bs=args.bs, input_size=args.input_size,
                      progr_size=args.progr_size, factors=args.size_facts)
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
    images = images if type(images) is np.ndarray else np.array(images)
    full_images_dict = get_full_img_dict(images, full_img_sep)
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


def create_dls_from_lst(blocks, get_y, bs, size, tr, val, args):
    tfms = fv.aug_transforms(size=size)
    if not args.no_norm:
        tfms.append(fv.Normalize.from_stats(*fv.imagenet_stats))
    data = fv.DataBlock(blocks=blocks,
                        get_items=lambda x: tr + val,
                        get_x=lambda x: crypto.decrypt_img(x, args.user_key) if args.encrypted else x,
                        get_y=get_y,
                        splitter=fv.IndexSplitter(list(range(len(tr), len(tr) + len(val)))),
                        item_tfms=fv.Resize(args.input_size),
                        batch_tfms=tfms)
    return data.dataloaders(args.data, bs=bs)


def progressive_resizing_dls(dls_fn, sl_data, max_bs, bs, input_size, progr_size, factors):
    input_sizes = [int(input_size * f) for f in factors] if progr_size else [input_size]
    batch_sizes = [max(1, min(int(bs / f / f), max_bs) // 2 * 2) for f in factors] if progr_size else [bs]
    for it, (bs, size) in enumerate(zip(batch_sizes, input_sizes)):
        run = f'{common.zero_pad(it, len(batch_sizes))}_{"SL" if sl_data else "WL"}_{size}px_bs{bs}'
        print(f"Iteration {it}: running {run}")
        yield it, run, dls_fn(bs, size)


class CustomTensorBoardCallback(fc.TensorBoardBaseCallback):
    def __init__(self, log_dir, run_name, grouped_metrics):
        super().__init__()
        self.log_dir = log_dir
        self.run_name = run_name
        self.grouped_metrics = grouped_metrics

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds") \
                   and int(os.environ.get('RANK', 0)) == 0
        if not self.run: return
        self._setup_writer()

    def after_batch(self):
        self.writer.add_scalar(f'{self.run_name}_Loss/train_loss', self.smooth_loss, self.train_iter)
        for i, h in enumerate(self.opt.hypers):
            for k, v in h.items(): self.writer.add_scalar(f'{self.run_name}_Opt_hyper/{k}_{i}', v, self.train_iter)

    def after_epoch(self):
        grouped = {}
        for n, v in zip(self.recorder.metric_names[2:-1], self.recorder.log[2:-1]):
            if n in self.grouped_metrics:
                perf = n.split('_')[1]
                if perf in grouped:
                    grouped[perf][n] = v
                else:
                    grouped[perf] = {n: v}
            else:
                log_group = 'Loss' if "loss" in n else 'Metrics'
                self.writer.add_scalar(f'{self.run_name}_{log_group}/{n}', v, self.train_iter)
        for n, v in grouped.items():
            self.writer.add_scalars(f'{self.run_name}_Metrics/{n}', v, self.train_iter)


def setup_tensorboard(learn, logdir, run_name, grouped_metrics):
    learn.remove_cb(CustomTensorBoardCallback)
    return CustomTensorBoardCallback(logdir, run_name, grouped_metrics)


def save_learner(learn, is_fp16, save_path):
    learn.remove_cb(CustomTensorBoardCallback)
    if is_fp16:
        learn.to_fp32()
    learn.save(save_path)
    if is_fp16:
        learn.to_fp16()


def split_model(model, splits):
    """Inspired from fastai 1, splits model on requested top level children"""
    top_children = list(model.children())
    idxs = [top_children.index(split) for split in splits]
    assert idxs == sorted(idxs), f"Provided splits ({splits}) are not sorted."
    assert len(idxs) > 0, f"Provided splits ({splits}) not found in top level children: {top_children}"
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(top_children): idxs.append(len(top_children))
    return [torch.nn.Sequential(*top_children[i:j]) for i, j in zip(idxs[:-1], idxs[1:])]


def prepare_training(args, image_data):
    common.check_dir_valid(args.data)
    common.set_seeds(args.seed)

    assert torch.cuda.is_available(), "Cannot run without CUDA device"
    torch.cuda.set_device(args.gpu)

    if args.encrypted:
        args.user_key = os.environ.get('CRYPTO_KEY', "").encode()
        args.user_key = crypto.request_key(args.data, args.user_key)

    if args.exp_logdir is None:
        args.exp_logdir = common.maybe_create(args.logdir, get_exp_logdir(args, image_data))
    print("Creation of log directory: ", args.exp_logdir)


def prepare_learner(args, learn):
    if not args.full_precision:
        learn.to_fp16()
    return learn


def basic_train(args, learn, fold, run, dls, metrics_names, save_model=True):
    learn.dls = dls
    cbT = setup_tensorboard(learn, args.exp_logdir, run, metrics_names)
    with learn.distrib_ctx():
        learn.fine_tune(args.epochs, cbs=[cbT])
    if save_model:
        save_path = os.path.join(args.exp_logdir, f'f{common.zero_pad(fold, args.nfolds)}__{run}_model')
        save_learner(learn, is_fp16=(not args.full_precision), save_path=save_path)


def train_model(args, get_splits, sl_images, get_dls, create_dls, create_learner,
                metrics_names, metrics_fn, correct_wl, wl_images):
    for fold, tr, val in get_splits(sl_images):
        for it, run, dls in get_dls(partial(create_dls, tr=tr, val=val, args=args), sl_data=True, max_bs=len(tr)):
            if it == 0:
                learn = create_learner(args, dls, metrics_fn)
            basic_train(args, learn, fold, run, dls, metrics_names)

        if args.use_wl:
            wl_classif_dls_fn = partial(create_dls, tr=wl_images, val=val, args=args)
            for it, run, dls in get_dls(wl_classif_dls_fn, sl_data=False, max_bs=len(wl_images)):
                correct_wl(args)
                basic_train(args, learn, fold, run, dls, metrics_names)


class FastaiTrainer:
    def __init__(self, args, stratify):
        self.args = args
        self.stratify = stratify
        self.metrics_names, self.metrics_fn = self.get_metrics()

    def get_metrics(self):
        raise not NotImplementedError

    def get_items(self):
        raise not NotImplementedError

    def create_learner(self):
        raise not NotImplementedError

    def create_dls(self):
        raise not NotImplementedError

    def correct_wl(self):
        raise not NotImplementedError

    def train_model(self):
        raise not NotImplementedError

    def split_data(self, items: np.ndarray, items_cls: np.ndarray):
        np.random.seed(self.args.seed)

        if self.stratify:
            cv_splitter, no_cv_splitter = StratifiedKFold, StratifiedShuffleSplit
        else:
            cv_splitter, no_cv_splitter = KFold, ShuffleSplit

        if self.args.cross_val:
            splitter = cv_splitter(n_splits=self.args.nfolds, shuffle=True, random_state=self.args.seed)
        else:
            splitter = no_cv_splitter(n_splits=1, test_size=self.args.valid_size, random_state=self.args.seed)

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(items, items_cls)):
            if self.args.cross_val:
                print("FOLD:", fold)
            yield fold, items[train_idx], items_cls[train_idx], items[valid_idx], items_cls[valid_idx]

    def get_data_path(self, weak_labels=False):
        if self.args.use_wl:
            return os.path.join(self.args.data, self.args.wl_train if weak_labels else self.args.sl_train)
        else:
            return self.args.data

    def basic_train(self, learn, fold, run, dls):
        learn.dls = dls
        cbT = setup_tensorboard(learn, self.args.exp_logdir, run, self.metrics_names)
        with learn.distrib_ctx():
            learn.fine_tune(self.args.epochs, cbs=[cbT])
        save_path = os.path.join(self.args.exp_logdir, f'f{common.zero_pad(fold, self.args.nfolds)}__{run}_model')
        save_learner(learn, is_fp16=(not self.args.full_precision), save_path=save_path)

    def prepare_learner(self, learn):
        if not self.args.full_precision:
            learn.to_fp16()
        return learn


class ImageTrainer(FastaiTrainer):
    def __init__(self, args, stratify, full_img_sep):
        super().__init__(args, stratify)
        self.full_img_sep = full_img_sep

    def get_image_cls(self, img_paths):
        return np.array([os.path.basename(os.path.dirname(img_path)) for img_path in img_paths])

    def split_data(self, items: np.ndarray, items_cls=None):
        full_images_dict = img_utils.get_full_img_dict(items, self.full_img_sep)
        full_images = np.array(list(full_images_dict.keys()))

        for fold, tr, _, val, _ in super().split_data(full_images, self.get_image_cls(full_images)):
            train_images = np.array([i for fi in tr for i in full_images_dict[fi]])
            valid_images = np.array([i for fi in val for i in full_images_dict[fi]])
            np.random.shuffle(train_images)
            np.random.shuffle(valid_images)
            yield fold, train_images, self.get_image_cls(train_images), valid_images, self.get_image_cls(valid_images)

    def progressive_resizing(self, tr, val, sl_data):
        if self.args.progr_size:
            input_sizes = [int(self.args.input_size * f) for f in self.args.factors]
            batch_sizes = [max(1, min(int(self.args.bs / f / f), tr.size) // 2 * 2) for f in self.args.factors]
        else:
            input_sizes = [self.args.input_size]
            batch_sizes = [self.args.bs]

        for it, (bs, size) in enumerate(zip(batch_sizes, input_sizes)):
            run = f'{common.zero_pad(it, len(batch_sizes))}_{"SL" if sl_data else "WL"}_{size}px_bs{bs}'
            print(f"Iteration {it}: running {run}")
            yield it, run, self.create_dls(bs, size, tr, val)

    def train_model(self):
        print("Running script with args:", self.args)
        sl_images, wl_images = self.get_items()
        for fold, tr, _, val, _ in self.split_data(sl_images):
            for it, run, dls in self.progressive_resizing(tr, val, sl_data=True):
                if it == 0:
                    learn = self.create_learner(dls)
                self.basic_train(learn, fold, run, dls)

            if self.args.use_wl:
                for it, run, dls in self.progressive_resizing(wl_images, val, sl_data=False, max_bs=len(wl_images)):
                    self.correct_wl()
                    self.basic_train(learn, fold, run, dls)


