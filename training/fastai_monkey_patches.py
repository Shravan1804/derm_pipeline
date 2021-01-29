import fastai.vision.all as fv


class FixedLabelSmoothingCrossEntropyFlat(fv.BaseLoss):
    "Same as `LabelSmoothingCrossEntropy`, but flattens input and target."
    y_int = True
    @use_kwargs_dict(keep=True, weight=None, eps=0.1, reduction='mean')
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(fv.LabelSmoothingCrossEntropy, *args, axis=axis, **kwargs)
    def activation(self, out): return fv.F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)