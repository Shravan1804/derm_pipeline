import torch
import fastai.vision.all as fv


# OFFICIAL VERSION DOES NOT SEEM TO SUPPORT THE WEIGHT ARGS + does not use provided axis
class FixedLabelSmoothingCrossEntropyFlat(fv.BaseLoss):
    "Same as `LabelSmoothingCrossEntropy`, but flattens input and target."
    y_int = True
    @fv.use_kwargs_dict(keep=True, weight=None, eps=0.1, reduction='mean')
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(FixedLabelSmoothingCrossEntropy, *args, axis=axis, **kwargs)
    def activation(self, out): return fv.F.softmax(out, dim=self.axis)
    def decodes(self, out):    return out.argmax(dim=self.axis)


# EXACT COPY, NEEDED OTHERWISE FIXED FLAT COMPLAINS OVER WEIGHT ARGS
class FixedLabelSmoothingCrossEntropy(fv.Module):
    y_int = True
    def __init__(self, eps:float=0.1, weight=None, reduction='mean'):
        fv.store_attr()

    def forward(self, output, target):
        c = output.size()[1]
        log_preds = fv.F.log_softmax(output, dim=1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1) #We divide by that size at the return line so sum and not mean
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * fv.F.nll_loss(log_preds, target.long(), weight=self.weight, reduction=self.reduction)

    def activation(self, out): return fv.F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)


# OFFICIAL VERSION NOT PART OF LATEST RELEASE YET
class FixedFocalLossFlat(fv.CrossEntropyLossFlat):
    """
    Same as CrossEntropyLossFlat but with focal paramter, `gamma`. Focal loss is introduced by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf. Note the class weighting factor in the paper, alpha, can be
    implemented through pytorch `weight` argument in nn.CrossEntropyLoss.
    """
    y_int = True
    @fv.use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        self.gamma = gamma
        self.reduce = kwargs.pop('reduction') if 'reduction' in kwargs else 'mean'
        super().__init__(*args, reduction='none', axis=axis, **kwargs)
    def __call__(self, inp, targ, **kwargs):
        ce_loss = super().__call__(inp, targ, **kwargs)
        pt = torch.exp(-ce_loss)
        fl_loss = (1-pt)**self.gamma * ce_loss
        return fl_loss.mean() if self.reduce == 'mean' else fl_loss.sum() if self.reduce == 'sum' else fl_loss

class FocalLossPlusCElossFlat(FixedFocalLossFlat):
    @fv.use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        super().__init__(*args, gamma=gamma, axis=axis, **kwargs)
        self.CEloss = fv.CrossEntropyLossFlat(*args, axis=axis, **kwargs)
    def __call__(self, inp, targ, **kwargs):
        return super().__call__(inp, targ, **kwargs) + self.CEloss(inp, targ, **kwargs)


class FocalLossPlusFocalDiceLoss(FixedFocalLossFlat):
    @fv.use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, alpha=.5, gamma_f=2, axis=-1, gamma_d=.75, delta_d=.7, log_d=False, **kwargs):
        super().__init__(*args, gamma=gamma_f, axis=axis, **kwargs)
        fv.store_attr("alpha,gamma_d,delta_d,log_d")

    def dice_multiclass_focal_loss(self, out, targs, eps=1e-7, smooth=1e-6):
        """
        out: BxNxHxW
        targs: BxHxW
        gamma: down-weights easy examples,
        delta: weight given to false positive and false negatives
        """
        # max for binary class case
        num_classes = max(out.shape[self.axis], 2)
        t = torch.nn.functional.one_hot(targs, num_classes).permute(0, 3, 1, 2).float()
        o = torch.softmax(out, dim=self.axis).clip(eps, 1. - eps)
        # axis are HxW
        TP = torch.sum(t * o, axis=[2, 3])
        FN = torch.sum(t * (1 - o), axis=[2, 3])
        FP = torch.sum((1 - t) * o, axis=[2, 3])
        dice = (TP + smooth) / (TP + self.delta_d * FN + (1 - self.delta_d) * FP + smooth)
        dice_loss = torch.pow(1 - dice, self.gamma_d).mean()
        return -torch.log(dice_loss) if self.log_d else dice_loss

    def __call__(self, inp, targ, **kwargs):
        f_loss = super().__call__(inp, targ, **kwargs)
        dice = self.dice_multiclass_focal_loss(inp, targ)
        return (1 - self.alpha) * f_loss + self.alpha * dice