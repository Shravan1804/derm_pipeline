import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


import torch
import fastai.vision as fv


# https://towardsdatascience.com/bayesian-deep-learning-with-fastai-how-not-to-be-uncertain-about-your-uncertainty-6a99d1aa686e

class CustomDropout(torch.nn.Module):
    """Custom Dropout module to be used as a baseline for MC Dropout"""

    def __init__(self, p: float, activate=True):
        super().__init__()
        self.activate = activate
        self.p = p

    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, training=self.training or self.activate)

    def extra_repr(self):
        return f"p={self.p}, activate={self.activate}"


def switch_custom_dropout(m, activate: bool = True, verbose: bool = False):
    """Turn all Custom Dropouts training mode to true or false according to the variable activate"""
    for c in m.children():
        if isinstance(c, CustomDropout):
            print(f"Current active : {c.activate}")
            print(f"Switching to : {activate}")
            c.activate = activate
        else:
            switch_custom_dropout(c, activate=activate)


def convert_layers(model: torch.nn.Module, original: torch.nn.Module, replacement: torch.nn.Module, get_args=None,
                   additional_args: dict = {}):
    """Convert modules of type "original" to "replacement" inside the model

    get_args : a function to use on the original module to eventually get its arguements to pass to the new module
    additional_args : a dictionary to add more args to the new module
    """
    for child_name, child in model.named_children():

        if isinstance(child, original):
            # First we grab args from the child
            if get_args:
                original_args = get_args(child)
            else:
                original_args = {}

            # If we want to provide additional args
            if additional_args:
                args = {**original_args, **additional_args}
            else:
                args = original_args

            new_layer = replacement(**args)
            setattr(model, child_name, new_layer)
        else:
            convert_layers(child, original, replacement,
                           get_args, additional_args)


def entropy(probs, softmax=False):
    """Return the prediction of a T*N*C tensor with :
        - T : the number of samples
        - N : the batch size
        - C : the number of classes
    """
    probs = fv.to_np(probs)
    prob = probs.mean(axis=0)

    entrop = - (np.log(prob) * prob).sum(axis=1)
    return entrop


def uncertainty_best_probability(probs):
    """Return the standard deviation of the most probable class"""
    idx = probs.mean(axis=0).argmax(axis=1)

    std = probs[:, np.arange(len(idx)), idx].std(axis=0)

    return std


def BALD(probs):
    """Information Gain, distance between the entropy of averages and average of entropy"""
    entrop1 = entropy(probs)
    probs = fv.to_np(probs)

    entrop2 = - (np.log(probs) * probs).sum(axis=2)
    entrop2 = entrop2.mean(axis=0)

    ig = entrop1 - entrop2
    return ig


def top_k_uncertainty(s, k=5, reverse=True):
    """Return the top k indexes"""
    sorted_s = sorted(list(zip(np.arange(len(s)), s)),
                      key=lambda x: x[1], reverse=reverse)
    output = [sorted_s[i][0] for i in range(k)]
    return output


def get_preds_sample(learn, ds_type=fv.DatasetType.Valid, n_sample=10, reduce=None, activ=None, with_loss=False):
    """Get MC Dropout predictions from a learner, and eventually reduce the samples"""
    preds = []
    for i in range(n_sample):
        pred, y = learn.get_preds(ds_type=ds_type, activ=activ)
        pred = pred.view((1,) + pred.shape)
        preds.append(pred)
    preds = torch.cat(preds)
    if reduce == "mean":
        preds = preds.mean(dim=0)
    return preds, y


def plot_hist_groups(pred, y, metric, bins=None, figsize=(16, 16)):
    TP = fv.to_np((pred.mean(dim=0).argmax(dim=1) == y) & (y == 1))
    TN = fv.to_np((pred.mean(dim=0).argmax(dim=1) == y) & (y == 0))
    FP = fv.to_np((pred.mean(dim=0).argmax(dim=1) != y) & (y == 0))
    FN = fv.to_np((pred.mean(dim=0).argmax(dim=1) != y) & (y == 1))

    result = metric(pred)

    TP_result = result[TP]
    TN_result = result[TN]
    FP_result = result[FP]
    FN_result = result[FN]

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    sns.distplot(TP_result, ax=ax[0, 0], bins=bins)
    ax[0, 0].set_title(f"True positive")

    sns.distplot(TN_result, ax=ax[0, 1], bins=bins)
    ax[0, 1].set_title(f"True negative")

    sns.distplot(FP_result, ax=ax[1, 0], bins=bins)
    ax[1, 0].set_title(f"False positive")

    sns.distplot(FN_result, ax=ax[1, 1], bins=bins)
    ax[1, 1].set_title(f"False negative")


def predict_entropy(learner, img, n_times=10):
    pred = learner.predict_with_mc_dropout(img, n_times=n_times)
    probs = [prob[2].view((1, 1) + prob[2].shape) for prob in pred]
    probs = torch.cat(probs)
    e = entropy(probs)
    return e


def plot_img_with_entropy(learner, img, n_times=10):
    e = predict_entropy(learner, img, n_times=n_times)
    img = fv.to_np(img.data.permute(1, 2, 0))
    plt.imshow(img)
    plt.title(f"Entropy : {e[0]:.{3}f}")


def custom_plot_hist(ax, vals, bins, title, gaussian_proba_density):
    ax.set_title(title)
    if len(vals) > 0:
        if gaussian_proba_density:
            sns.distplot(vals, ax=ax, bins=bins, norm_hist=True)
        else:
            weights = np.ones_like(np.array(vals)) / float(len(np.array(vals)))
            ax.hist(vals, weights=weights, bins=bins)
        for c, l, q in zip(['r', 'g', 'b'], ['--', '-', '-.'], np.quantile(vals, [.25, .5, .75])):
            ax.axvline(q, color=c, linestyle=l)


def custom_plot_hist_groups(labels, preds, y, metric, bins=None, gauss_proba_density=False):
    predictions = preds.mean(dim=0).argmax(dim=1)
    result = metric(preds)

    ncols = 4
    nrows = len(labels) + 1  # + 1 for the entropy distribution over all correct/incorrect predictions
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), sharex='col', sharey='row')
    axs = axs.flatten()

    correct = fv.to_np(predictions == y)
    incorrect = fv.to_np(predictions != y)
    custom_plot_hist(axs[0], result[correct], bins, "Entropy correct preds", gauss_proba_density)
    custom_plot_hist(axs[1], result[incorrect], bins, "Entropy incorrect preds", gauss_proba_density)
    for i in range(2, ncols):
        axs[i].axis('off')

    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [10], color=c, linestyle=l, lw=2) for c, l in zip(['r', 'g', 'b'], ['--', '-', '-.'])]
    axs[ncols - 1].legend(legend_lines, ['Q1', 'Q2', 'Q3'], loc='center')

    for cls_idx, cls_label in enumerate(labels):
        counts = {}
        counts['TP'] = fv.to_np((predictions == cls_idx) & (y == cls_idx))
        counts['TN'] = fv.to_np((predictions != cls_idx) & (y != cls_idx))
        counts['FP'] = fv.to_np((predictions == cls_idx) & (y != cls_idx))
        counts['FN'] = fv.to_np((predictions != cls_idx) & (y == cls_idx))
        for ax, (idx_name, idx) in zip(axs[(1 + cls_idx) * ncols:], counts.items()):
            custom_plot_hist(ax, result[idx], bins, f"Entropy of {cls_label} {idx_name}", gauss_proba_density)


def custom_top_k_preds(p, k=3):
    # p is of shape n_times, n_images, n_classes
    p_mean = p.mean(axis=0)
    p_std = p.std(axis=0)
    topk_p, topk_idx = p_mean.topk(k, axis=1)
    return topk_idx, topk_p, p_std.gather(dim=1, index=topk_idx)


def convert_learner(learner):
    # Convert nn.Dropout to CustomDropout module
    # Needs to be turned on with switch_custom_dropout
    get_args = lambda dp: {"p": dp.p}
    convert_layers(learner.model, torch.nn.Dropout, CustomDropout, get_args)
    switch_custom_dropout(learner.model, False, verbose=False)

