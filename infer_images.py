import os

import torch
import fastai.vision.all as fv


class FastaiModel:
    def __init__(self, model_path, use_cpu, bs):
        fv.default_device(not use_cpu)
        self.device = torch.device('cuda') if not use_cpu and torch.cuda.is_available() else torch.device('cpu')
        self.learner = fv.load_learner(model_path)
        self.learner.dls.bs = bs

    def predict_imgs(self, ims):
        """ims is a lst of raw images"""
        dl = self.learner.dls.test_dl(ims, num_workers=0)
        preds = self.learner.get_preds(dl=dl, with_decoded=True)[-1]
        #preds = torch.randn((len(ims), *ims[0].shape[:2]))
        return preds
