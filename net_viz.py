import math
from contextlib import ExitStack

import numpy as np
from matplotlib import pyplot as plt

import torch
import fastai.vision as fv


class Hook:
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)

    def hook_func(self, m, i, o):self.stored = o.detach().clone()

    def __enter__(self, *args): return self

    def __exit__(self, *args): self.hook.remove()


class HookBwd:
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)

    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()

    def __enter__(self, *args): return self

    def __exit__(self, *args): self.hook.remove()


def prepare_patch(patch, learn):
    x = fv.Image(fv.pil2tensor(patch, np.float32).div_(255))
    return learn.data.one_item(x, detach=False, denorm=False)[0]


def _grad_cam_top_out_acts_grad(learn, layers, patch, cls=-1, topk=3):
    x = prepare_patch(patch, learn)
    with ExitStack() as stack:
        hookg = [stack.enter_context(HookBwd(layer)) for layer in layers]
        hook = [stack.enter_context(Hook(layer)) for layer in layers]
        output = learn.model.eval()(x.cuda() if torch.cuda.is_available() else x)
        acts = [h.stored for h in hook]
        top_preds = output[0].cpu().detach().numpy().argsort()[::-1][:topk]
        cls = cls if cls != -1 else int(top_preds[0])
        output[0, cls].backward()
        grads = [h.stored for h in hookg]
    return top_preds, output, acts, grads


def grad_cam(learn, layers_with_names, patch, cls=-1, topk=3):
    layers, layer_names = zip(*layers_with_names)
    labs = learn.data.classes
    top_preds, output, acts, grads = _grad_cam_top_out_acts_grad(learn, layers, patch, cls, topk)
    print("Out:", ', '.join([f'{l} = {o.item():.{3}f}' for l, o in zip(labs, output[0])]))
    with torch.no_grad():
        ws = [grad[0].mean(dim=[1, 2], keepdim=True) for grad in grads]
        cam_maps = [(w * act[0]).sum(0) for w, act in zip(ws, acts)]
        print(f'Patch predicted as {labs[int(top_preds[0])]}, showing grad cam for class {labs[cls]}')
        plot_grad_cam(patch, layer_names, [c.detach().cpu() for c in cam_maps])


def plot_grad_cam(patch, layer_names, cam_maps, ncol=5):
    nrow = math.ceil(len(cam_maps) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
    axs = axs.flatten()
    axs[0].imshow(patch)
    axs[0].set_title('Patch')
    for cam_map, title, ax in zip(cam_maps, layer_names, axs[1:]):
        ax.imshow(patch)
        ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0, *patch.shape[:-1], 0),
                  interpolation='bilinear', cmap='magma');
        ax.set_title(title)
    for ax in axs:
        ax.axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.85])


#https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask