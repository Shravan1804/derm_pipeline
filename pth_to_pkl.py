import os
import argparse
from functools import partial

import radam
import torch
import fastai.vision as fv

import common


def load_custom_pretrained_weights(learner, model_path, device=None):
    if device is None:
        device = learner.data.device
    new_state_dict = torch.load(model_path, map_location=device)['model']
    learner_state_dict = learner.model.state_dict()
    for name, param in learner_state_dict.items():
        if name in new_state_dict:
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                print('Shape mismatch at:', name, 'skipping')
        else:
            print(f'{name} weight of the model not in pretrained weights')
    learner.model.load_state_dict(learner_state_dict)


def main(args):
    data = fv.ImageDataBunch.from_folder(path=args.data, train=args.train, valid=args.valid,
                                         bs=4, size=512, ds_tfms=None)
    learner = fv.cnn_learner(data, getattr(fv.models, args.model), pretrained=False, metrics=[fv.accuracy],
                             wd=1e-3, opt_func=partial(radam.RAdam), bn_wd=False, true_wd=True,
                             loss_func=fv.LabelSmoothingCrossEntropy(), callback_fns=[])
    load_custom_pretrained_weights(learner, args.pth)
    learner.export(args.pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert pth to fastai pkl export format")
    parser.add_argument('--data', type=str, default='/home/shravan/deep-learning/data/USZ_pipeline_sample', help="root dataset dir")
    parser.add_argument('--train', type=str, default='strong_labels_train', help="weak labels dir")
    parser.add_argument('--valid', type=str, default='strong_labels_test', help="strong labels dir")
    parser.add_argument('--model', type=str, default='resnet34', help="model name")
    parser.add_argument('--pth', type=str, default='/home/shravan/models/008_freeze_10_weak_labels_train_model.pth', help="model weights")
    parser.add_argument('--pkl', type=str, default='008_freeze_10_weak_labels_train_model.pkl', help="path where to save pkl")

    args = parser.parse_args()

    common.time_method(main, args)

