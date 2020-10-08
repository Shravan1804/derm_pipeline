import math
from contextlib import ExitStack

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import PIL
import torch
import fastai.vision as fv

import common


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


def _grad_cam(learn, layers, patches, cls, relu, to_cpu=True):
    """Computes grad cam for provided patches at provided layer for specified classes
    Returns:
        cls: the labels used for gradcam, shape bs
        top_preds: preds proba in descending order, shape bs x n_classes
        output: model output, shape bs x n_classes
        acts: layers activations, lst of n_layers items, each of shape bs x layer_n_features x 8 x 8
        grads: layers gradients, lst of n_layers items, each of shape bs x layer_n_features x 8 x 8
        cam_maps cam viz, lst of 1 (3 if relu is true) x n_layers items, each of shape bs x 8 x 8
    """
    x = torch.cat([prepare_patch(patch, learn) for patch in patches], dim=0)

    with ExitStack() as stack:
        hooks, hookgs = zip(*[(stack.enter_context(Hook(l)), stack.enter_context(HookBwd(l))) for l in layers])

        output = learn.model.eval()(x.cuda() if torch.cuda.is_available() else x)
        top_preds = output.detach().argsort(axis=1, descending=True)
        cls = cls if cls is not None else top_preds[:, 0]

        selected_logits = (output * torch.nn.functional.one_hot(cls, num_classes=learn.data.c)).sum(axis=1)
        selected_logits.backward(gradient=torch.ones_like(selected_logits))

        with torch.no_grad():
            # cannot use torch.cat since layers will have different nb of features
            acts, grads = [h.stored for h in hooks], [h.stored for h in hookgs]
            cam_maps = [(grad.mean(dim=[2, 3], keepdim=True) * act).sum(1) for act, grad in zip(acts, grads)]
            if relu:
                cam_maps = [c for cs in cam_maps for c in (cs, fv.F.relu(cs), fv.F.relu(-cs))]

        if to_cpu:
            cls = cls.cpu()
            top_preds = top_preds.cpu()
            output = output.cpu()
            acts, grads, cam_maps = (list(map(torch.Tensor.cpu, t)) for t in [acts, grads, cam_maps])
    return cls, top_preds, output, acts, grads, cam_maps


def grad_cam(learn, layers_with_names, patches, cls=None, relu=False, ret=False):
    if cls is not None:
        assert type(cls) is torch.Tensor, f"Please provide cls parameter as tensor"
        assert len(patches) == cls.size()[0], f"Provided classes {cls} do not match batch size {len(patches)}"

    layers, layer_names = zip(*layers_with_names)
    layer_names = layer_names if not relu else [f'{n} {r}' for n in layer_names for r in ['ALL', 'POS', 'NEG']]

    res = _grad_cam(learn, layers, patches, cls, relu)
    cls, top_preds, _, _, _, cam_maps = res

    labs = np.array(learn.data.classes)
    print(f'Prediction / GradCAM class: {[f"{labs[t]}/{labs[c]}" for t, c in zip(top_preds[:, 0], cls)]}')

    plot_grad_cam(patches, layer_names, cam_maps)

    return res if ret else None


def plot_grad_cam(patches, layer_names, cam_maps, ncols=5, cmap='magma'):
    nrows = math.ceil(((len(patches) + 1) * len(cam_maps)) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axs = axs.flatten()

    ax_i = 0
    for pi, patch in enumerate(patches):
        common.img_on_ax(patch, axs[ax_i], title='Patch')
        ax_i += 1
        for cam, title in zip(cam_maps, layer_names):
            common.img_on_ax(patch, axs[ax_i], title=title)
            axs[ax_i].imshow(cam[pi], alpha=0.6, extent=(0, *patch.shape[:-1], 0), interpolation='bilinear', cmap=cmap)
            ax_i += 1

    for ax in axs[ax_i:]:
        ax.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.85])


def create_heatmap(patch, grad_scaled_acts, cmap='magma'):
    """Heavy magic to reproduce matplotlib viz ... grad_scaled_acts np array, returns the patch heatmap"""
    my_cm = cm.get_cmap(cmap)
    normed_data = (grad_scaled_acts - np.min(grad_scaled_acts)) / (np.max(grad_scaled_acts) - np.min(grad_scaled_acts))
    mapped_data = my_cm(normed_data, bytes=True)
    acts_heat = PIL.Image.fromarray(mapped_data).resize(patch.shape[:-1], resample=PIL.Image.BILINEAR)
    acts_heat = PIL.Image.fromarray(np.array(acts_heat)[:, :, :3])

    return np.array(PIL.Image.blend(PIL.Image.fromarray(patch), acts_heat, 0.6))


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
