#!/usr/bin/env python

"""infer_images.py: Base class to perform model inference on new images"""

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
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, common_img as cimg
from training.train_utils import GPUManager
from general.PatchExtractor import PatchExtractor


class ImageInference:
    """Base class for model inference. Should be extended to fit custom tasks requirements"""
    @staticmethod
    def get_argparser(parser):
        """Adds inference args to supplied arg parser
        :param parser: argparser
        :return: argparser
        """
        parser.add_argument('--mpath', type=str, help="Model weights path (use if single model inference required)")
        parser.add_argument('--mdir', type=str, help="Model weights dir path (multiple models inference)")
        parser.add_argument('--impath', type=str, help="path of image to apply inference on")
        parser.add_argument('--imdir', type=str, help="path of dir with images to apply inference on")

        parser.add_argument('--ps', type=int, help="Patch px side size to cut images")
        return parser

    @staticmethod
    def prepare_inference(args):
        """Checks provided arguments validity
        :param args: command line arguments
        :return: args after validation
        """
        args.inference = True

        if args.mpath is not None: common.check_file_valid(args.mpath)
        else:
            if args.mdir is None and args.exp_logdir is not None:
                exp_logdir_mdir = os.path.join(args.exp_logdir, 'models')
                if len(ImageInference.get_model_weights_files(exp_logdir_mdir)) > 0: args.mdir = exp_logdir_mdir
            common.check_dir_valid(args.mdir)

        if args.impath is not None: common.check_file_valid(args.impath)
        elif args.imdir is not None: common.check_dir_valid(args.imdir)
        else: assert len(args.sl_tests) > 0, "No image to apply inference on was provided"
        args.inference_on_test_dirs = args.impath is None and args.imdir is None

        return args

    @staticmethod
    def get_model_weights_files(dirpath):
        """Inference can be run directly on logdir from trainer pipeline.
        In this case, will use saved models for inference
        :param dirpath: str, trainer logs model dir path
        :return: list of model weights path
        """
        return [m for m in common.list_files(dirpath, full_path=True) if m.endswith(".pth")]

    def __init__(self, trainer):
        """Creates inference object
        :param trainer: trainer object
        """
        self.trainer = trainer
        self.args = self.trainer.args

    def get_result_save_path(self, model_info):
        """Creates dir where to save the inference results
        :param model_info: str, model weight filename without extension, contains all necessary run info
        :return: str, path of inference result dir
        """
        save_tag = "_" + os.path.basename(self.args.mdir) if self.args.mdir is not None else ""
        if self.args.ps is not None:
            save_tag += f'_ps{self.args.ps}px'
        return common.maybe_create(self.args.exp_logdir, f'preds{save_tag}_{self.args.exp_name}_{model_info}')

    def maybe_patch(self, img_path):
        """Patch image if patch size was specified
        :param img_path: str, path of image
        :return: tuple, list of patches (or with full image if no ps) and list of patch maps (or None)
        """
        if self.args.ps is None:
            return [cimg.load_img(img_path)], None
        else:
            return PatchExtractor.impath_to_patches(img_path, self.args.ps)

    def maybe_get_labels(self, img_path):
        """Get label from item. Should be overriden"""
        raise NotImplementedError

    def inference_items(self):
        """Creates list of inference items
        :return: list of tuples, each tuple contains an item with its label
        """
        if self.args.impath is not None:
            return [(Path(self.args.impath), self.maybe_get_labels(self.args.impath))]
        else:
            imgs = common.list_images(self.args.imdir, full_path=True, recursion=True, posix_path=True)
            return [(impath, self.maybe_get_labels(impath)) for impath in imgs]

    def prepare_learner_input(self, inference_item):
        """Converts inference item to learning input. Should be extended."""
        raise NotImplementedError

    def learner_inference(self, learn, save_dir):
        """Applies learner on inference items
        :param learn: learner object
        :param save_dir: str, path where to store results
        """
        for inference_item in self.inference_items():
            linput, with_labels, pms = self.prepare_learner_input(inference_item)
            dl = learn.dls.test_dl(linput, with_labels=with_labels)
            interp = self.learner_preds(learn, dl)
            GPUManager.clean_gpu_memory(dl)
            interp.pms = pms
            self.process_results(inference_item, interp, with_labels, save_dir)

    def learner_preds(self, learn, dl):
        """Compute learner predictions
        :param learn: learner object
        :param dl: inference dataloader
        :return: namespace with predictions, targets, decoded predictions
        """
        with GPUManager.running_context(learn, self.args.gpu_ids):
            interp = SimpleNamespace()
            interp.preds, interp.targs, interp.decoded = learn.get_preds(dl=dl, with_decoded=True)
        return interp

    def process_results(self, inference_item, interp, save_dir):
        """Process inference results. Should be extended."""
        raise NotImplementedError

    def inference(self):
        """Inference steps
        TODO: multiple models inference + results aggregation"""
        if self.args.mpath is not None: model_paths = [self.args.mpath]
        else: model_paths = ImageInference.get_model_weights_files(self.args.mdir)
        _, tr, val = next(self.trainer.split_data(*self.trainer.get_train_items()[0]))
        for mpath in model_paths:
            run_info = os.path.basename(mpath).split(self.trainer.MODEL_SUFFIX)[0]
            learn = self.trainer.load_learner_from_run_info(run_info, tr, val, mpath)
            if self.args.inference_on_test_dirs: self.trainer.evaluate_on_test_sets(learn, run_info)
            else: self.learner_inference(learn, self.get_result_save_path(run_info))
            GPUManager.clean_gpu_memory(learn.dls, learn)
        if self.args.inference_on_test_dirs: self.trainer.generate_tests_reports()

