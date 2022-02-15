#!/usr/bin/env python

"""classification_with_image_and_metadata.py: Model arch for image features + metadata classification"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

from collections import OrderedDict

import torch
from torch import nn
import fastai.vision.all as fv


def dec2bin(x, bits):
    """Converts integer to binary tensor
    :param x: tensor, integer value in base 10 to be converted
    :param bits: integer, number of bits to be used in the conversion
    :return: tensor, float binary tensor
    """
    # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    """Converts binary tensor to base 10 integer
    :param b: tensor, float binary tensor
    :param bits: integer, number of bits in input tensor
    :return: tensor, float tensor with integer value
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def my_create_head(metadata_dim, nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, first_bn=True, bn_final=False,
                   lin_first=False, y_range=None):
    """Based on fastai create_head method. Adds metadata dimensions.
    """
    if concat_pool: nf *= 2
    lin_ftrs = [metadata_dim + nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    bns = [first_bn] + [True] * len(lin_ftrs[1:])
    ps = fv.L(ps)
    if len(ps) == 1: ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    pool = fv.AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, fv.Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni, no, bn, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
        layers += fv.LinBnDrop(ni, no, bn=bn, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(fv.SigmoidRange(*y_range))
    return nn.Sequential(*layers)


class ClassifWithMetadata(nn.Sequential):
    def __init__(self, metadata_dim, arch, n_out, lin_ftrs=None, embed_dim=64):
        super().__init__()
        self.model_meta = fv.model_meta[arch]
        self.body = fv.create_body(arch, n_in=3, pretrained=True, cut=self.meta['cut'])
        nf = fv.num_features_model(nn.Sequential(*self.body.children()))
        self.metadata_dim = metadata_dim
        self.embed_dim = embed_dim
        print(f"Metadata (#{metadata_dim}) will be embedded in {self.embed_dim} dimensions")
        self.embed = nn.Embedding(2 ** metadata_dim, self.embed_dim)
        self.head = my_create_head(self.embed_dim, nf, n_out, lin_ftrs=lin_ftrs)
        fv.apply_init(self.head, nn.init.kaiming_normal_)

        self.add_module("body", self.body)
        self.add_module("embed", self.embed)
        self.add_module("head", self.head)

    def forward(self, imtensor, metadata):
        imfeatures = self.body(imtensor)
        imfeatures = self.head[:2](imfeatures)
        embed_metadata = self.embed(bin2dec(metadata, self.metadata_dim).long())
        x = torch.cat([imfeatures, embed_metadata], 1)
        out = self.head[2:](x)
        return out

    def create_fastai_learner(self, dls, loss_func=None):
        learn = fv.Learner(dls, self, loss_func, splitter=self.model_meta['split'])
        if loss_func is None: learn.loss_func = learn.loss_func[-1]
        print("Using", learn.loss_func)
        learn.freeze()
        return learn

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return nn.Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return super().__getitem__(idx)


#learn = ClassifWithMetadata(metadata_dim, getattr(fv, model_arch), len(cats)).create_fastai_learner(dls)