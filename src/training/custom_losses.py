import math

import torch
import fastai.vision.all as fv

from ..training import fastai_monkey_patches as fmp


class FocalLossPlusCElossFlat(fmp.FixedFocalLossFlat):
    @fv.use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        super().__init__(*args, gamma=gamma, axis=axis, **kwargs)
        self.CEloss = fv.CrossEntropyLossFlat(*args, axis=axis, **kwargs)
    def __call__(self, inp, targ, **kwargs):
        return super().__call__(inp, targ, **kwargs) + self.CEloss(inp, targ, **kwargs)


class GeneralizedDiceLoss(fv.Module):
    y_int = True
    def __init__(self, axis=1, eps=1e-7): fv.store_attr()

    def forward(self, out, targs):
        num_classes = max(out.shape[self.axis], 2)
        t = torch.nn.functional.one_hot(targs, num_classes).permute(0, 3, 1, 2).float()
        o = torch.softmax(out, dim=self.axis).clip(self.eps, 1. - self.eps)

        w = 1 / ((torch.einsum("bkwh->bk", t) + self.eps) ** 2)
        inter = w * torch.einsum("bkwh,bkwh->bk", o, t)
        union = w * (torch.einsum("bkwh->bk", o) + torch.einsum("bkwh->bk", t))

        loss = 1 - 2 * (torch.einsum("bk->b", inter) + self.eps) / (torch.einsum("bk->b", union) + self.eps)
        return loss.mean()
    def activation(self, out): return fv.F.softmax(out, dim=self.axis)
    def decodes(self, out): return out.argmax(dim=self.axis)


class DiceLoss(fv.Module):
    y_int = True
    def __init__(self, axis=1, delta=.5, eps=1e-7, smooth=1e-6):
        fv.store_attr()
        self.coeff = 0
    def forward(self, out, targs):
        num_classes = max(out.shape[self.axis], 2)
        t = torch.nn.functional.one_hot(targs, num_classes).permute(0, 3, 1, 2).float()
        o = torch.softmax(out, dim=self.axis).clip(self.eps, 1. - self.eps)
        # axis are HxW
        TP = torch.sum(t * o, axis=[2, 3])
        FN = torch.sum(t * (1 - o), axis=[2, 3])
        FP = torch.sum((1 - t) * o, axis=[2, 3])
        self.coeff = (TP + self.smooth) / (TP + self.delta * FN + (1 - self.delta) * FP + self.smooth)
        return (1 - self.coeff).mean()
    def activation(self, out): return fv.F.softmax(out, dim=self.axis)
    def decodes(self, out): return out.argmax(dim=self.axis)


class FocalDiceLoss(DiceLoss):
    def __init__(self, axis=1, delta=.5, eps=1e-7, smooth=1e-6, gamma=.75):
        super().__init__(axis, delta, eps, smooth)
        self.gamma = gamma
    def forward(self, out, targs):
        super().forward(out, targs)
        return torch.pow(1 - self.coeff, self.gamma).mean()


class TverskyLoss(DiceLoss):
    def __init__(self, axis=1, delta=.7, eps=1e-7, smooth=1e-6): super().__init__(axis, delta, eps, smooth)



class FocalTverskyLoss(FocalDiceLoss):
    def __init__(self, axis=1, delta=.7, eps=1e-7, smooth=1e-6, gamma=.75): super().__init__(axis, delta, eps, smooth, gamma)


class CosineFocalTverskyLoss(FocalTverskyLoss):
    def __init__(self, axis=1, delta=.7, eps=1e-7, smooth=1e-6, gamma=.75): super().__init__(axis, delta, eps, smooth, gamma)
    def forward(self, out, targs):
        super().forward(out, targs)
        self.coeff.clip_(0., 1.)
        self.coeff = torch.cos(self.coeff * math.pi)
        return torch.pow(1 - self.coeff, self.gamma).mean()


class FocalLossPlusFocalDiceLoss(fmp.FixedFocalLossFlat):
    @fv.use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, alpha=.5, gamma_f=2, axis=1, gamma_d=.75, delta_d=.7, log_d=False, **kwargs):
        super().__init__(*args, gamma=gamma_f, axis=axis, **kwargs)
        fv.store_attr("alpha,gamma_d,delta_d,log_d")
        self.dice_loss = FocalDiceLoss(axis=axis, delta=delta_d, gamma=gamma_d)

    def __call__(self, inp, targ, **kwargs):
        f_loss = super().__call__(inp, targ, **kwargs)
        dice = self.dice_loss(inp, targ)
        if self.log_d: dice = -torch.log(dice)
        return (1 - self.alpha) * f_loss + self.alpha * dice