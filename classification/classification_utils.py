
import torch

import common


def cls_perf(perf, inp, targ, cls_idx, cats, axis=-1):
    if axis is not None:
        inp = inp.argmax(dim=axis)
    if cls_idx is None:
        res = [common.get_cls_TP_TN_FP_FN(targ == c, inp == c) for c in range(len(cats))]
        res = torch.cat([torch.tensor(r).unsqueeze(0) for r in res], dim=0).sum(axis=0).tolist()
        return torch.tensor(perf(*res))
    else:
        return torch.tensor(perf(*common.get_cls_TP_TN_FP_FN(targ == cls_idx, inp == cls_idx)))

