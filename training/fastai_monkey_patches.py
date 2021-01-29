import fastai.vision.all as fv


class FixedLabelSmoothingCrossEntropyFlat(fv.BaseLoss):
    "Same as `LabelSmoothingCrossEntropy`, but flattens input and target."
    y_int = True
    @fv.use_kwargs_dict(keep=True, weight=None, eps=0.1, reduction='mean')
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(FixedLabelSmoothingCrossEntropy, *args, axis=axis, **kwargs)
    def activation(self, out): return fv.F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)


# Cell
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

