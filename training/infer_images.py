import os
import sys

import torch
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common, crypto
from training import train_utils


class ImageInference:
    def __init__(self):
        self.trainer = None
        self.args = self.trainer.args
        pass

    def create_trainer(self):
        pass

    def create_base_dls(self):
        sl_data, wl_data = self.trainer.get_train_items()
        _, tr, val = next(self.trainer.split_data(*sl_data))
        return self.trainer.create_dls(tr, val, self.args.bs, self.args.input_size)

    def create_learner(self):
        learn = self.trainer.create_learner(self.create_base_dls())

    def infer_items(self, learn, items, with_labels):
        return fv.Interpretation.from_learner(learn, dl=learn.dls.test_dl(items, with_labels=with_labels))

    def inference(self):
        models = [m for m in common.list_files(self.args.exp_logdir, full_path=True) if m.endswith(".pth")]
        _, tr, val = next(self.split_data(*self.get_train_items()[0]))
        for mpath in models:
            run_info = os.path.basename(mpath).split(self.MODEL_SUFFIX)[0]
            learn = self.load_learner_from_run_info(run_info, tr, val, mpath)
            self.evaluate_on_test_sets(learn, run_info)
            GPUManager.clean_gpu_memory(learn.dls, learn)
        self.generate_tests_reports()

