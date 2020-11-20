import os
import sys
import argparse
from functools import partial

import matplotlib.pyplot as plt

import torch
import fastai.vision.all as fv
import fastai.distributed as fd
from fastai.callback.tracker import EarlyStoppingCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, train_utils
from general.PatchExtractor import PatchExtractor
import classification.classification_utils as classif_utils


class ImageClassificationTrainer(train_utils.ImageTrainer):
    def load_data(self, path):
        return common.list_files_in_dirs(path, full_path=True, posix_path=True)

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
                                        tr.tolist(), val.tolist(), fv.parent_label, bs, size)

    def create_learner(self, dls):
        metrics = list(self.cats_metrics.values()) + [fv.accuracy]  # for early stop callback
        if "efficientnet" in self.args.model:
            from efficientnet_pytorch import EfficientNet
            model = fd.rank0_first(lambda: EfficientNet.from_pretrained(self.args.model))
            model._fc = torch.nn.Linear(model._fc.in_features, dls.c)
            model_splitter = lambda m: fv.L(train_utils.split_model(m, [m._fc])).map(fv.params)
            learn = fd.rank0_first(lambda: fv.Learner(dls, model, metrics=metrics, splitter=model_splitter))
        else:
            learn = fd.rank0_first(lambda: fv.cnn_learner(dls, getattr(fv, self.args.model), metrics=metrics))
        return self.prepare_learner(learn)

    def early_stop_cb(self):
        return EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=3)

    def process_preds(self, interp):
        interp.cm = classif_utils.conf_mat(self.args.cats, interp.decoded, interp.targs)
        return interp

    def aggregate_test_performance(self, interp_lst):
        agg = super().aggregate_test_performance(interp_lst)
        agg["cm"] = train_utils.tensors_mean_std([interp.cm for interp in interp_lst])
        return agg

    def plot_test_performance(self, test_path, run, agg_run_perf):
        for show_val in [False, True]:
            save_path = os.path.join(test_path, f'{run}{"_show_val" if show_val else ""}.jpg')
            fig, axs = plt.subplots(1, 2, figsize=self.args.test_figsize)
            bar_perf = {k: (v[0].numpy(), v[1].numpy()) for k, v in agg_run_perf.items() if k != 'cm'}
            bar_cats = self.args.cats + [self.ALL_CATS]
            common.grouped_barplot_with_err(axs[0], bar_perf, bar_cats, xlabel='Classes', show_val=show_val)
            cm_perf = tuple(map(torch.Tensor.numpy, agg_run_perf['cm']))
            common.plot_confusion_matrix(axs[1], cm_perf, self.args.cats)
            fig.tight_layout(pad=.2)
            plt.savefig(save_path, dpi=400)


def main(args):
    classif = ImageClassificationTrainer(args, stratify=True, full_img_sep=PatchExtractor.SEP)
    classif.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastai classification")
    train_utils.common_train_args(parser, pdef={'--bs': 6, '--model': 'resnet34'})
    train_utils.common_img_args(parser, pdef={'--input-size': 256})
    args = parser.parse_args()

    train_utils.prepare_training(args, image_data=True, custom="classification")

    if args.cats is None:
        args.cats = common.list_dirs(os.path.join(args.data, args.sl_train), full_path=False)

    common.time_method(main, args, prepend=f"GPU {args.gpu} proc: ")
