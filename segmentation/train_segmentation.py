#!/usr/bin/env python

"""train_segmentation.py: Train segmentation model. Can be extended for more specific segmentation tasks."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

import os
import sys
import datetime
from functools import partial

import numpy as np
from tqdm import tqdm
import sklearn.metrics as skm

import torch
import fastai.vision.all as fv
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.vision.models.unet import DynamicUnet

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training import metrics, train_utils
from training.image_trainer import ImageTrainer
import segmentation.mask_utils as mask_utils
import segmentation.segmentation_utils as segm_utils
from segmentation.crop_to_thresh import SEP as CROP_SEP


class ImageSegmentationTrainer(ImageTrainer):
    """Class used to train image segmentation model. Can be extended for specific more tasks."""
    @staticmethod
    def get_argparser(desc="Fastai segmentation image trainer arguments", pdef=dict(), phelp=dict()):
        """Creates segmentation argparser
        :param desc: str, argparser description
        :param pdef: dict, default arguments values
        :param phelp: dict, argument help strings
        :return: argparser
        """
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(ImageSegmentationTrainer, ImageSegmentationTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument('--plot-no-bg', action='store_true', help="Include perfs without bg cat in plots")
        parser.add_argument('--coco-metrics', action='store_true', help="Also computes coco metrics")
        parser.add_argument('--rm-small-objs', action='store_true', help="Remove objs smaller than --min-size")
        parser.add_argument('--min-size', default=60, type=int, help="Objs below this size will be discarded")
        return segm_utils.common_segm_args(parser, pdef, phelp)

    @staticmethod
    def prepare_training(args):
        """Sets up training, check args validity.
        :param args: command line args
        """
        args.exp_name = "img_segm_" + args.exp_name
        super(ImageSegmentationTrainer, ImageSegmentationTrainer).prepare_training(args)

    def __init__(self, args, stratify=False, full_img_sep=CROP_SEP, **kwargs):
        """Creates classification trainer
        :param args: command line args
        :param stratify: bool, whether to stratify data when splitting
        :param full_img_sep: str, when splitting the data train/valid, will make sure to split on full images
        """
        self.NO_BG = '_no_bg'    # used to differentiate metrics ignoring background
        super().__init__(args, stratify, full_img_sep, **kwargs)
        self.loss_axis = 1
        if self.args.coco_metrics:
            # conditional import because object detection requires the icevision library
            from object_detection.object_detection_utils import CustomCocoEval, segm_dataset_to_coco_format

    def load_items(self, set_dir):
        """Loads training items from directory
        :param set_dir: str, directory containing training items
        :return: tuple of fastai L lists, first list contain items, second list contains labels (masks)
        """
        path = os.path.join(self.args.data, set_dir)
        images = common.list_images(os.path.join(path, self.args.img_dir), full_path=True, posix_path=True)
        return fv.L(images), fv.L([self.get_image_mask_path(img_path) for img_path in images])

    def get_image_mask_path(self, img_path):
        """Prepares corresponding mask path
        :param img_path: str, image path
        :return: str, mask path
        """
        return segm_utils.get_mask_path(img_path, self.args.img_dir, self.args.mask_dir, self.args.mext)

    def load_mask(self, item, load_mask_array=False):
        """Loads image mask
        :param item: str, mask path or array
        :param load_mask_array: bool, optional, force mask array loading in memory
        :return: array mask if encrypted else str mask path
        """
        if type(item) is np.ndarray: mask = item
        elif common.is_path(item): mask = self.load_image_item(item, load_im_array=load_mask_array)
        else: mask = mask_utils.rles_to_non_binary_mask(item)
        if self.args.rm_small_objs:
            if common.is_path(mask): mask = mask_utils.load_mask_array(mask)
            mask = mask_utils.rm_small_objs_from_non_bin_mask(mask, self.args.min_size, range(len(self.args.cats)), self.args.bg)
        return mask

    def get_cat_metric_name(self, perf_fn, cat, bg):
        """Gets category specific metric name
        :param perf_fn: str, base metric name (e.g. precision)
        :param cat: str, category name
        :param bg: int, background code, if None then metric will include background else not
        :return str, category specific metric function name
        """
        return f'{super().get_cat_metric_name(perf_fn, cat)}{"" if bg is None else self.NO_BG}'

    def create_cats_metrics(self, perf_fn, cat_id, cat, metrics_fn):
        """Generates metrics functions for the individual categories and background inclusion/exclusion
        :param perf_fn: function, metrics to apply, e.g. precision
        :param cat_id: int, category id for which to compute metric
        :param cat: str, label of category for which to compute metric
        :param metrics_fn: dict, contains generated metrics function names as keys and metrics functions as values
        """
        for bg in [None, self.args.bg]:
            cat_perf = partial(segm_utils.cls_perf, cls_idx=cat_id, cats=self.args.cats, bg=bg)
            signature = f'{self.get_cat_metric_name(perf_fn, cat, bg)}(inp, targ)'
            code = f"def {signature}: return cat_perf(metrics.{perf_fn}, inp, targ).to(inp.device)"
            exec(code, {"cat_perf": cat_perf, 'metrics': metrics}, metrics_fn)

    def ordered_test_perfs_per_cats(self):
        """Returns custom metrics ordered per category order and metrics type
        :return: list of tuples, each tuple is the list of category metrics with the metrics function name
        """
        ordered = []
        for perf_fn in self.args.metrics_base_fns:
            for bg in [None, self.args.bg]:
                mns = [self.get_cat_metric_name(perf_fn, cat, bg) for cat in self.get_cats_with_all()]
                ordered.append((mns, perf_fn + ("" if bg is None else self.NO_BG)))
        return ordered

    def compute_conf_mat(self, targs, preds):
        """Compute pixel-wise confusion matrix from flattened predicted masks
        :param targs: tensor, ground truth, size B x M x M
        :param decoded: tensor, decoded predictions, size B x M x M
        :return: tensor, confusion metrics N x N (with N categories)
        """
        return segm_utils.pixel_conf_mat(targs, preds, self.args.cats)

    def plot_custom_metrics(self, ax, agg_perf, show_val, title=None):
        """Plots aggregated metrics results.
        :param ax: axis
        :param agg_perf: dict, fold aggregated metrics results
        :param show_val: bool, print values on plot
        :param title: str, plot title
        """
        if not self.args.plot_no_bg:
            agg_perf = {k: v for k, v in agg_perf.items() if self.NO_BG not in k}
        super().plot_custom_metrics(ax, agg_perf, show_val, title)

    def compute_metrics(self, interp):
        """Apply metrics functions on test set predictions. If requested, will also compute object detection metrics
        :param interp: namespace with predictions, targs, decoded preds, test set predictions
        :return: same namespace but with metrics results dict
        """
        interp.targs = interp.targs.as_subclass(torch.Tensor)   # otherwise issues with fastai PILMask custom class
        interp = super().compute_metrics(interp)
        targs, dec = interp.targs.flatten(), interp.decoded.flatten()
        print(skm.classification_report(targs, dec, target_names=self.args.cats,
                                        labels=[i for i, v in enumerate(self.args.cats)]))
        if self.args.coco_metrics:
            to_coco = partial(segm_dataset_to_coco_format, cats=self.args.cats, bg=self.args.bg)
            with common.elapsed_timer() as elapsed:
                gt, dt = to_coco(interp.targs), to_coco(interp.decoded, scores=True)
                print(f"Segmentation dataset converted in {datetime.timedelta(seconds=elapsed())}.")
            cocoEval = CustomCocoEval(gt, dt, all_cats=self.ALL_CATS)
            cocoEval.eval_acc_and_summarize(verbose=False)
            self.coco_param_labels, areaRng, maxDets, stats = cocoEval.get_precision_recall_with_labels()
            interp.metrics['cocoeval'] = torch.Tensor(stats)
            interp.metrics['cocoeval_areaRng'] = torch.Tensor(areaRng)
            interp.metrics['cocoeval_maxDets'] = torch.Tensor(maxDets)
        return interp

    def plot_test_performance(self, test_path, run, agg_perf):
        """Plots aggregated performance
        :param test_path: str, test identifier
        :param run: str, run identifier
        :param agg_perf: dict, aggregated performance
        """
        super().plot_test_performance(test_path, run, agg_perf)
        if self.args.coco_metrics:
            figsize = self.args.test_figsize
            cocoeval = agg_perf['cocoeval']
            show_val = not self.args.no_plot_val
            save_path = self.plot_save_path(test_path, run, show_val, custom="_coco")
            CustomCocoEval.plot_coco_eval(self.coco_param_labels, cocoeval, figsize, save_path, show_val)

    def customize_datablock(self):
        """Provides experiment specific kwargs for DataBlock
        :return: dict with argnames and argvalues
        """
        return {
            'blocks': (fv.ImageBlock, fv.MaskBlock(self.args.cats)),
            'get_y': fv.Pipeline([fv.ItemGetter(1), self.load_mask]),
        }

    def get_class_weights(self, train_items):
        """Compute class weights based on train items labels pixel frequency
        :param train_items: tuple of fastai lists, (items, labels)
        :return: tensor, weight for each categories
        """
        print("Computing class weights")
        size = self.args.input_size, self.args.input_size
        ms = [mask_utils.resize_mask(self.load_mask(m, load_mask_array=True), size) for _, m in tqdm(train_items)]
        _, class_counts = np.unique(np.stack(ms), return_counts=True)
        assert class_counts.size == len(self.args.cats)
        # this one causes issues if imbalance too high
        #return torch.FloatTensor(class_counts.max() / class_counts)
        return torch.FloatTensor(1 / np.log(class_counts))

    def customize_learner(self, dls):
        """Provides experiment specific kwargs for Learner
        :return: kwargs dict
        """
        kwargs = super().customize_learner(dls)
        kwargs['metrics'].append(fv.foreground_acc)
        return kwargs

    def create_learner(self, dls):
        """Creates learner with callbacks
        :param dls: train/valid dataloaders
        :return: learner
        """
        learn_kwargs = self.customize_learner(dls)
        callbacks = []
        if self.args.mixup:
            callbacks += [MixUp()]
        if self.args.wandb:
            import wandb
            from fastai.callback.wandb import WandbCallback
            callbacks += [WandbCallback()]
            # update the name of the wandb run
            run_name = f'{self.args.model}-{wandb.run.name}'
            wandb.run.name = run_name
            wandb.run.save()

        if "ssl" in self.args.model:
            from self_supervised_dermatology.embedder import Embedder
            ssl_model = self.args.model.replace('ssl_', '')
            model, info = Embedder.load_pretrained(ssl_model, return_info=True)
            print(f'Loaded pretrained SSL model: {info}')
            # removing last layer (i.e. adaptive pooling)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            # infer the shape of the dataset
            size = dls.one_batch()[0].shape[-2:]
            # create the unet
            msplitter = lambda m: fv.L(
                train_utils.split_model(m, [m[0][:6], m[0][6:], m[1:]])).map(
                    fv.params)
            unet = DynamicUnet(model, n_out=dls.train.after_item.c, img_size=size)
            learn = fv.Learner(dls, unet, splitter=msplitter, cbs=callbacks, **learn_kwargs)
        else:
            learn = fv.unet_learner(dls, getattr(fv, self.args.model), cbs=callbacks, **learn_kwargs)
        # also log the training metrics
        # then train + valid metrics are reported
        learn.recorder.train_metrics = True
        return self.prepare_learner(learn)

    def correct_wl(self, wl_items_with_labels, preds):
        """Corrects weakly labeled data with model predictions. To save RAM, the masks are RLE encoded category wise.
        :param wl_items_with_labels: tuple of fastai lists, (items, labels)
        :param decoded: tensor, decoded predictions
        :return: tuple of fastai lists, (items, corrected labels)
        """
        wl_items, labels = wl_items_with_labels
        labels = [mask_utils.non_binary_mask_to_rles(self.load_image_item(pred.numpy())) for pred in preds]
        # preds size is self.args.input_size but wl_items are orig size => first item_tfms should resize the input
        return wl_items, fv.L(labels)

    @staticmethod
    def load_pretrained_backbone_weights(weights_path, model):
        """Closely ressemble train_utils version. Param order switched, + the layers are renamed.
        :param weights_path: str, path to .pth weights
        :param model: torch module
        """
        new_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
        model_state_dict = model.state_dict()
        for name, param in model_state_dict.items():
            new_name = name.replace('layers.', '')
            if new_name in new_state_dict:
                input_param = new_state_dict[new_name]
                if input_param.shape == param.shape:
                    param.copy_(input_param)
                else:
                    print('Shape mismatch at:', name, 'skipping')
            else:
                print(f'{name} weight of the model not in pretrained weights')
        model.load_state_dict(model_state_dict)


def main(args):
    """Creates segmentation trainer
    :param args: command line args
    """
    segm = ImageSegmentationTrainer(args)
    segm.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageSegmentationTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    ImageSegmentationTrainer.prepare_training(args)

    common.time_method(main, args)
