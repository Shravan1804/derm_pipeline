#!/usr/bin/env python

"""infer_images_classif.py: Applies classification model on new images. Uses classification trainer args"""

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
from types import SimpleNamespace

import cv2

import torch
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, common_plot as cplot
from general.PatchExtractor import PatchExtractor
from training.train_utils import GPUManager
from training.infer_images import ImageInference
from classification.train_classification import ImageClassificationTrainer


class ImageClassificationInference(ImageInference):
    """Class used to perform inference with classification model
    Based on classification trainer"""
    @staticmethod
    def get_argparser(parser):
        """Add classification inference specific args to argparser
        :param parser: argparser
        :return: argparser
        """
        parser = super(ImageClassificationInference, ImageClassificationInference).get_argparser(parser)
        parser.add_argument('--topk', type=int, default=2, help="Top probas to be written on img")
        parser.add_argument('--gradcam', action='store_true', help="Show gradcam of predictions")
        return parser

    @staticmethod
    def prepare_inference(args):
        """Calls classification trainer setup
        :param args: command line arguments
        """
        super(ImageClassificationInference, ImageClassificationInference).prepare_inference(args)
        ImageClassificationTrainer.prepare_training(args)

    def maybe_get_labels(self, impath):
        """Get label from item
        :param impath: str, image path
        :return: label
        TODO
        """
        return None

    def prepare_learner_input(self, inference_item):
        """Prepares learner input from inference item
        :param inference_item: tuple, image and label, label can be None
        :return: tuple, learner input, flag set to True if labels available, patch maps (may be None if no patching)
        """
        img_path, label = inference_item
        with_labels = label is not None
        if with_labels: raise NotImplementedError
        im_patches, pms = self.maybe_patch(img_path)
        linput = [(p,) for p in im_patches]
        return linput, with_labels, pms

    def learner_preds(self, learn, dl):
        """Computes learner predictions on inference items. Redefined here to be able to perform gradcam
        TODO improve, batch backward
        :param learn: learner
        :param dl: dataloader
        :return: namespace object with preds, targs, decoded preds
        """
        if self.args.gradcam:
            dl.bs = 1
            interp = SimpleNamespace()
            interp.preds, interp.targs, interp.decoded, interp.gradcam = [], [], [], []
            with GPUManager.running_context(learn, self.args.gpu_ids):
                for b in dl:
                    x = b[0]
                    if len(b) == 1: interp.targs = None
                    else: interp.targs.append(b[1])
                    with fv.Hook(learn.model[0], lambda m, i, o: o[0].detach().clone(), is_forward=False) as hookg:
                        with fv.Hook(learn.model[0], lambda m, i, o: o.detach().clone()) as hook:
                            output = learn.model.eval()(x.cuda() if torch.cuda.is_available() else x)
                            act = hook.stored
                        with torch.no_grad():
                            p = output.argmax(axis=1)[0]
                        output[0, p].backward()
                        interp.preds.append(output.cpu())
                        interp.decoded.append(p.cpu())
                        grad = hookg.stored
                    interp.gradcam.append((grad[0].mean(dim=[1,2], keepdim=True) * act[0]).sum(0).cpu())
            return interp
        else: return super().learner_preds(learn, dl)

    def process_results(self, inference_item, interp, save_dir):
        """Prepare inference results: rebuild predictions if patched, resize predictions and targs
        :param inference_item: tuple with image and its label
        :param interp: namespace, learner inference results
        :param save_dir: str, path where to save result plot
        :return: reconstructed predictions
        """
        img_path, gt = inference_item
        im = common.trainer.load_image_item(img_path)
        pred = interp.preds.topk(self.args.topk, axis=1)
        hmap = interp.gradcam.numpy()
        if interp.pms is None: hmap = cv2.resize(hmap, im.shape[:2], interpolation=cv2.INTER_LINEAR)
        else: hmap = PatchExtractor.rebuild_im_from_patches(interp.pms, hmap, im.shape[:2], interpol=cv2.INTER_LINEAR)
        interp.gradcam = hmap
        save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + "_preds.jpg")
        if not self.args.no_plot: self.plot_results(interp, im, gt, pred, save_path)
        return pred

    def plot_results(self, interp, im, gt, pred, save_path):
        """Plots inference results: original image with predicted labels, eventually heatmap, eventually metrics
        :param interp: namespace, learner inference results
        :param im: array, image
        :param gt: array, ground truth
        :param pred: array, decoded predictions
        :param save_path: str, path to save plot
        """
        topk_p, topk_idx = pred
        with_labels = gt is not None
        ncols = 2 if with_labels else 1
        if self.args.gradcam: ncols += 1
        fig, axs = cplot.prepare_img_axs(im.shape[0] / im.shape[1], 1, ncols, flatten=True)
        if ncols == 1: axs = [axs]
        cplot.img_on_ax(im, axs[0], title='Original image')
        pred_pos = [(0, 0)] if interp.pms is None else [(pm['h'], pm['w'])for pm in interp.pms]
        for (h, w), p, prob in zip(pred_pos, topk_idx, topk_p):
            axs[0].text(w+50, h + 50, f'{self.args.cats[p]}: {prob:.2f}')
        axi = 1
        if self.args.gradcam:
            cplot.img_on_ax(im, axs[axi], title='GradCAM')
            axs[axi].imshow(interp.gradcam, alpha=0.6, cmap='magma');
            axi += 1
        if with_labels:
            agg_perf = self.trainer.aggregate_test_performance([self.trainer.process_test_preds(interp)])
            self.trainer.plot_custom_metrics(axs[axi], agg_perf, show_val=True)
        if save_path is not None: cplot.plt_save_fig(save_path, fig=fig, dpi=150)


def main(args):
    """Performs classification inference based on provided args
    :param args: command line arguments
    :return:
    """
    classif = ImageClassificationInference(ImageClassificationTrainer(args))
    classif.inference()


if __name__ == '__main__':
    pdef = {'--bs': 6, '--model': 'resnet34', '--input-size': 256}
    parser = ImageClassificationInference.get_argparser(ImageClassificationTrainer.get_argparser(pdef=pdef))
    args = parser.parse_args()

    ImageClassificationInference.prepare_inference(args)

    common.time_method(main, args)

