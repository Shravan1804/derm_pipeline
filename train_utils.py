import os
from pathlib import Path
from functools import partial

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

import torch
import fastai.distributed   # needed for fastai multi gpu
import fastai.vision.all as fv
import fastai.callback.tensorboard as fc

import common
import crypto


def maybe_set_gpu(gpuid, num_gpus):
    if gpuid is not None:
        assert num_gpus == 1, f"Warning cannot fix more than 1 gpus, requested {num_gpus}"
        if torch.cuda.is_available() and gpuid is not None:
            torch.cuda.set_device(gpuid)


def common_train_args(parser, pdef=dict(), phelp=dict()):
    parser.add_argument('--data', type=str, required=True, help="Root dataset dir")
    parser.add_argument('--use-wl', action='store_true', help="Data dir contains wl and sl data")
    parser.add_argument('--wl-train', type=str, default='weak_labels', help="weak labels (wl) dir")
    parser.add_argument('--sl-train', type=str, default='strong_labels_train', help="strong labels (sl) dir")
    parser.add_argument('--sl-test', type=str, default='strong_labels_test', help="sl test dir")

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

    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)

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
    get_dls = partial(progressive_resizing_dls, bs=args.bs, input_size=args.input_size, num_gpus=args.num_gpus,
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


def progressive_resizing_dls(dls_fn, max_bs, bs, input_size, num_gpus, progr_size, factors):
    bs *= num_gpus
    input_sizes = [int(input_size * f) for f in factors] if progr_size else [input_size]
    batch_sizes = [max(1, min(int(bs / f / f), max_bs) // 2 * 2) for f in factors] if progr_size else [bs]
    for it, (bs, size) in enumerate(zip(batch_sizes, input_sizes)):
        run = f'{common.zero_pad(it, len(batch_sizes))}_{size}px_bs{bs}'
        print(f"Iteration {it}: running {run}")
        yield it, run, dls_fn(bs, size)


class CustomTensorBoardCallback(fc.TensorBoardBaseCallback):
    def __init__(self, log_dir, grouped_metrics):
        super().__init__()
        self.log_dir = log_dir
        self.grouped_metrics = grouped_metrics

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds") \
                   and int(os.environ.get('RANK', 0)) == 0
        if not self.run: return
        self._setup_writer()

    def after_batch(self):
        self.writer.add_scalar('Loss/train_loss', self.smooth_loss, self.train_iter)
        for i, h in enumerate(self.opt.hypers):
            for k, v in h.items(): self.writer.add_scalar(f'Opt_hyper/{k}_{i}', v, self.train_iter)

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
                self.writer.add_scalar(f'{log_group}/{n}', v, self.train_iter)
        for n, v in grouped.items():
            self.writer.add_scalars(f'Metrics/{n}', v, self.train_iter)


def setup_tensorboard(learn, logdir, run_name, grouped_metrics):
    learn.remove_cb(CustomTensorBoardCallback)
    logdir = common.maybe_create(logdir, run_name)
    learn.add_cb(CustomTensorBoardCallback(logdir, grouped_metrics))


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

    maybe_set_gpu(args.gpuid, args.num_gpus)

    if args.encrypted:
        args.user_key = crypto.request_key(args.data, args.user_key)

    if args.exp_logdir is None:
        args.exp_logdir = common.maybe_create(args.logdir, get_exp_logdir(args, image_data))
    print("Creation of log directory: ", args.exp_logdir)


def prepare_learner(args, learn):
    if not args.full_precision:
        learn.to_fp16()
    if args.num_gpus > 1:
        learn.to_parallel(device_ids=list(range(args.num_gpus)))
    return learn

