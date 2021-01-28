import os
import sys

import torch
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto
from training.train_utils import GPUManager


class ImageInference:
    @staticmethod
    def prepare_inference_args(parser):
        parser.add_argument('--mpath', type=str, help="Model weights path (use if single model inference required)")
        parser.add_argument('--mdir', type=str, help="Model weights dir path (multiple models inference)")
        args = parser.parse_args()
        args.inference = True

        if args.mpath is not None: common.check_file_valid(args.mpath)
        else: common.check_dir_valid(args.mdir)

        return args

    def __init__(self, trainer):
        self.trainer = trainer
        self.args = self.trainer.args

    def infer_items(self, learn, items, with_labels):
        return fv.Interpretation.from_learner(learn, dl=learn.dls.test_dl(items, with_labels=with_labels))

    def inference(self):
        # create learner from args
        if self.args.mpath is not None: model_paths = [self.args.mpath]
        else: model_paths = [m for m in common.list_files(self.args.mdir, full_path=True) if m.endswith(".pth")]
        _, tr, val = next(self.trainer.split_data(*self.trainer.get_train_items()[0]))
        for mpath in model_paths:
            run_info = os.path.basename(mpath).split(self.trainer.MODEL_SUFFIX)[0]
            learn = self.trainer.load_learner_from_run_info(run_info, tr, val, mpath)
            self.trainer.evaluate_on_test_sets(learn, run_info)
            GPUManager.clean_gpu_memory(learn.dls, learn)
        self.trainer.generate_tests_reports()

