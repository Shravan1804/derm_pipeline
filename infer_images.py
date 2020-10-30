import os

import numpy as np

import torch
import fastai.vision as fv

import common
from PatchExtractor import PatchExtractor


class FastaiModel:
    def __init__(self, model_path, use_cpu, bs):
        self.device = torch.device('cuda') if not use_cpu and torch.cuda.is_available() else torch.device('cpu')
        fv.defaults.device = torch.device(self.device)
        self.learner = fv.load_learner(os.path.dirname(model_path), os.path.basename(model_path))
        self.learner.data.batch_size = bs
        self.classes = self.learner.data.classes

    def prepare_img_for_inference(self, ims):
        ims = [fv.Image(fv.pil2tensor(im, np.float32).div_(255)) for im in ims]
        return torch.cat([self.learner.data.one_item(im)[0] for im in ims], dim=0)

