import fastai.vision.all as fv
import torch


# OFFICIAL VERSION DOES NOT SEEM TO SUPPORT THE WEIGHT ARGS + does not use provided axis
class FixedLabelSmoothingCrossEntropyFlat(fv.BaseLoss):
    """`LabelSmoothingCrossEntropy`, but flattens input and target."""

    y_int = True

    @fv.use_kwargs_dict(keep=True, weight=None, eps=0.1, reduction="mean")
    def __init__(self, *args, axis=-1, **kwargs):
        super().__init__(FixedLabelSmoothingCrossEntropy, *args, axis=axis, **kwargs)

    def activation(self, out):
        return fv.F.softmax(out, dim=self.axis)

    def decodes(self, out):
        return out.argmax(dim=self.axis)


# EXACT COPY, NEEDED OTHERWISE FIXED FLAT COMPLAINS OVER WEIGHT ARGS
class FixedLabelSmoothingCrossEntropy(fv.Module):
    y_int = True

    def __init__(self, eps: float = 0.1, weight=None, reduction="mean"):
        fv.store_attr()

    def forward(self, output, target):
        c = output.size()[1]
        log_preds = fv.F.log_softmax(output, dim=1)
        if self.reduction == "sum":
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(
                dim=1
            )  # We divide by that size at the return line so sum and not mean
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * fv.F.nll_loss(
            log_preds, target.long(), weight=self.weight, reduction=self.reduction
        )

    def activation(self, out):
        return fv.F.softmax(out, dim=-1)

    def decodes(self, out):
        return out.argmax(dim=-1)


# OFFICIAL VERSION NOT PART OF LATEST RELEASE YET
class FixedFocalLossFlat(fv.CrossEntropyLossFlat):
    """
    CrossEntropyLossFlat but with focal paramter, `gamma`.

    Focal loss is introduced by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf. Note the class weighting factor in the paper, alpha, can be
    implemented through pytorch `weight` argument in nn.CrossEntropyLoss.
    """

    y_int = True

    @fv.use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction="mean")
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        self.gamma = gamma
        self.reduce = kwargs.pop("reduction") if "reduction" in kwargs else "mean"
        super().__init__(*args, reduction="none", axis=axis, **kwargs)

    def __call__(self, inp, targ, **kwargs):
        ce_loss = super().__call__(inp, targ, **kwargs)
        pt = torch.exp(-ce_loss)
        fl_loss = (1 - pt) ** self.gamma * ce_loss
        return (
            fl_loss.mean()
            if self.reduce == "mean"
            else fl_loss.sum()
            if self.reduce == "sum"
            else fl_loss
        )
