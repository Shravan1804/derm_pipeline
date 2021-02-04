import os
import sys

import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training.train_utils import GPUManager
from general.PatchExtractor import PatchExtractor


class ImageInference:
    @staticmethod
    def get_argparser(parser):
        parser.add_argument('--mpath', type=str, help="Model weights path (use if single model inference required)")
        parser.add_argument('--mdir', type=str, help="Model weights dir path (multiple models inference)")
        parser.add_argument('--impath', type=str, help="path of image to apply inference on")
        parser.add_argument('--imdir', type=str, help="path of dir with images to apply inference on")

        parser.add_argument('--ps', type=int, help="Patch px side size to cut images")
        return parser

    @staticmethod
    def prepare_inference(args):
        args.inference = True

        if args.mpath is not None: common.check_file_valid(args.mpath)
        elif args.mdir is None and args.exp_logdir is not None:
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
        return [m for m in common.list_files(dirpath, full_path=True) if m.endswith(".pth")]

    def __init__(self, trainer):
        self.trainer = trainer
        self.args = self.trainer.args

    def learner_inference(self, learn): raise NotImplementedError

    def maybe_patch(self, img_path):
        if self.args.ps is None: return [common.load_rgb_img(img_path)], None
        else: return PatchExtractor.image_to_patches(img_path, self.args.ps)

    def infer_items(self, learn, items, with_labels):
        dl = learn.dls.test_dl(items, with_labels=with_labels)
        with GPUManager.running_context(learn, self.args.gpu_ids):
            res = fv.Interpretation.from_learner(learn, dl=dl)
        GPUManager.clean_gpu_memory(dl)
        return res

    def inference(self):
        if self.args.mpath is not None: model_paths = [self.args.mpath]
        else: model_paths = ImageInference.get_model_weights_files(self.args.mdir)
        _, tr, val = next(self.trainer.split_data(*self.trainer.get_train_items()[0]))
        for mpath in model_paths:
            run_info = os.path.basename(mpath).split(self.trainer.MODEL_SUFFIX)[0]
            learn = self.trainer.load_learner_from_run_info(run_info, tr, val, mpath)
            if self.args.inference_on_test_dirs: self.trainer.evaluate_on_test_sets(learn, run_info)
            else: self.learner_inference(learn)
            GPUManager.clean_gpu_memory(learn.dls, learn)
        if self.args.inference_on_test_dirs: self.trainer.generate_tests_reports()

