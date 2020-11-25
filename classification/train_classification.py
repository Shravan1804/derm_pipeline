import os
import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import torch
import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils
from general.PatchExtractor import PatchExtractor
import classification.classification_utils as classif_utils


class ImageClassificationTrainer(train_utils.ImageTrainer):
    def load_data(self, path):
        images = common.list_files_in_dirs(path, full_path=True, posix_path=True)
        return np.array(images), self.get_images_cls(images)

    def get_images_cls(self, images):
        return np.array([os.path.basename(os.path.dirname(img_path)) for img_path in images])

    def get_metrics(self):
        metrics_fn = {}
        cat_perf = partial(classif_utils.cls_perf, cats=self.args.cats)
        for cat_id, cat in zip([*range(len(self.args.cats))] + [None], self.args.cats + [self.ALL_CATS]):
            for perf_fn in self.BASIC_PERF_FNS:
                code = f"def {cat}_{perf_fn}(inp, targ):" \
                       f"return cat_perf(train_utils.{perf_fn}, inp, targ, {cat_id}).to(inp.device)"
                exec(code, {"cat_perf": cat_perf, 'train_utils': train_utils}, metrics_fn)
        return metrics_fn

    def create_dls(self, tr, val, bs, size):
        return self.create_dls_from_lst((fv.ImageBlock, fv.CategoryBlock),
                                        tr.tolist(), val.tolist(), lambda x: x[1], bs, size)

    def create_learner(self, dls):
        metrics = list(self.cats_metrics.values()) + [fv.accuracy]  # for early stop callback
        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = EfficientNet.from_pretrained(self.args.model)
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            model_splitter = lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params)
            learn = fv.Learner(dls, model, metrics=metrics, splitter=model_splitter)
        else:
            learn = fv.cnn_learner(dls, getattr(fv, self.args.model), metrics=metrics)
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, preds):
        wl_items, labels = wl_items_with_labels
        preds = np.array(self.args.cats)[preds.numpy()]
        corr = labels != preds
        changelog = "\n".join([f'{i};{ic};{p}' for i, ic, p in zip(wl_items[corr], labels[corr], preds[corr])])
        labels[corr] = np.array([p for p in preds[corr]])
        return (wl_items, labels), changelog

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=3)

    def process_preds(self, interp):
        interp.cm = classif_utils.conf_mat(self.args.cats, interp.decoded, interp.targs)
        return interp

    def aggregate_test_performance(self, folds_res):
        agg = super().aggregate_test_performance(folds_res)
        agg["cm"] = tuple([s.numpy() for s in train_utils.tensors_mean_std([interp.cm for interp in folds_res])])
        return agg

    def plot_test_performance(self, test_path, run, agg_run_perf):
        for show_val in [False, True]:
            save_path = os.path.join(test_path, f'{run}{"_show_val" if show_val else ""}.jpg')
            fig, axs = plt.subplots(1, 2, figsize=self.args.test_figsize)
            bar_perf = {p: cat_vals for p, cat_vals in agg_run_perf.items() if p != 'cm'}
            bar_cats = self.args.cats + [self.ALL_CATS]
            common.grouped_barplot_with_err(axs[0], bar_perf, bar_cats, xlabel='Classes', show_val=show_val)
            common.plot_confusion_matrix(axs[1], agg_run_perf['cm'], self.args.cats)
            fig.tight_layout(pad=.2)
            plt.savefig(save_path, dpi=400)


def main(args):
    classif = ImageClassificationTrainer(args, stratify=True, full_img_sep=PatchExtractor.SEP)
    classif.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256}
    parser = ImageClassificationTrainer.get_argparser(desc="Fastai image classification", pdef=defaults)
    args = parser.parse_args()

    args.exp_name = "img_classif_"+args.exp_name
    if args.cats is None:
        args.cats = common.list_dirs(os.path.join(args.data, args.sl_train), full_path=False)

    ImageClassificationTrainer.prepare_training(args)

    common.time_method(main, args, prepend=f"GPU {args.gpu} proc: ")
