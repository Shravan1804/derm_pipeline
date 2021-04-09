import os
import sys
import collections
from functools import partial

import numpy as np
import sklearn.metrics as skm

import torch
import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import train_utils, train_utils_img
from general.PatchExtractor import PatchExtractor
import classification.classification_utils as classif_utils


class ImageClassificationTrainer(train_utils_img.ImageTrainer):
    @staticmethod
    def get_argparser(desc="Fastai image classification trainer arguments", pdef=dict(), phelp=dict()):
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        psr = super(ImageClassificationTrainer, ImageClassificationTrainer).get_argparser(desc, pdef, phelp)
        psr.add_argument('--oversample', action='store_true', help="Uses weighted dls based on class distrib")
        return psr

    @staticmethod
    def get_exp_logdir(args, custom=""):
        d = ''
        if args.oversample: d += '_oversample'
        custom = f'{d}_{custom}'
        return super(ImageClassificationTrainer, ImageClassificationTrainer).get_exp_logdir(args, custom=custom)

    @staticmethod
    def prepare_training(args):
        args.exp_name = "img_classif_"+args.exp_name
        if args.cats is None: args.cats = common.list_dirs(os.path.join(args.data, args.sl_train[0]), full_path=False)
        if args.exp_logdir is None:
            args.exp_logdir = os.path.join(args.logdir, ImageClassificationTrainer.get_exp_logdir(args))
        super(ImageClassificationTrainer, ImageClassificationTrainer).prepare_training(args)

    def __init__(self, args, stratify=True, full_img_sep=PatchExtractor.SEP, **kwargs):
        super().__init__(args, stratify, full_img_sep, **kwargs)

    def load_items(self, set_dir):
        path = os.path.join(self.args.data, set_dir)
        images = common.list_files_in_dirs(path, full_path=True, posix_path=True)
        return fv.L(images), fv.L([classif_utils.get_image_cls(img_path) for img_path in images])

    def get_full_img_cls(self, img_path): return classif_utils.get_image_cls(img_path)

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        cat_perf = partial(classif_utils.cls_perf, cats=self.args.cats)
        signature = f'{self.get_cat_metric_name(perf_fn, cat)}(inp, targ)'
        code = f"def {signature}: return cat_perf(train_utils.{perf_fn}, inp, targ, {cat_id}).to(inp.device)"
        exec(code, {"cat_perf": cat_perf, 'train_utils': train_utils}, metrics_fn)

    def ordered_test_perfs_per_cats(self):
        return [([self.get_cat_metric_name(f, c) for c in self.get_cats_with_all()], f) for f in self.args.metrics_fns]


def compute_conf_mat(self, targs, preds): return classif_utils.conf_mat(targs, preds, self.args.cats)

    def create_dls(self, tr, val, bs, size):
        blocks = fv.ImageBlock, fv.CategoryBlock(vocab=self.args.cats)
        if self.args.oversample:
            kwargs = {'dl_type': fv.WeightedDL, 'wgts': self.get_train_items_weights(list(zip(*tr))),
                      'dl_kwargs': [{}, {'cls': fv.TfmdDL}]}
        else: kwargs = {}
        return self.create_dls_from_lst(blocks, tr, val, bs, size, **kwargs)

    def get_class_weights(self, train_items):
        counts = collections.Counter([x[1] for x in train_items])
        class_counts = np.array([counts[c] for c in self.args.cats])
        return torch.FloatTensor(class_counts.max() / class_counts)

    def get_train_items_weights(self, train_items):
        labels, class_weights = [x[1] for x in train_items], self.get_class_weights(train_items).numpy()
        return class_weights[fv.CategoryMap(self.args.cats).map_objs(labels)]

    def create_learner(self, dls):
        learn_kwargs = self.get_learner_kwargs(dls)
        metrics = list(self.cust_metrics.values())
        metrics += [fv.Precision(average='micro'), fv.Recall(average='micro')] + [fv.accuracy]  # acc for early stop cb
        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained(self.args.model)
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            msplitter = lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params)
            learn = fv.Learner(dls, model, metrics=metrics, splitter=msplitter, **learn_kwargs)
        else:
            model = getattr(fv, self.args.model)
            learn = fv.cnn_learner(dls, model, metrics=metrics, **learn_kwargs)
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, preds):
        wl_items, labels = wl_items_with_labels
        preds = np.array(self.args.cats)[preds.numpy()]
        corr = np.array(labels) != preds
        labels[corr] = np.array([p for p in preds[corr]])
        return wl_items, fv.L(labels.tolist())

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=3)

    def process_test_preds(self, interp):
        interp = super().process_test_preds(interp)
        d, t = fv.flatten_check(interp.decoded, interp.targs)
        print(skm.classification_report(t, d, labels=list(interp.dl.vocab.o2i.values()),
                                        target_names=[str(v) for v in interp.dl.vocab]))
        return interp


def main(args):
    classif = ImageClassificationTrainer(args)
    classif.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256}
    parser = ImageClassificationTrainer.get_argparser(desc="Fastai image classification", pdef=defaults)
    args = parser.parse_args()

    ImageClassificationTrainer.prepare_training(args)

    common.time_method(main, args)

