import io
import itertools
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_distribution_summary(
    df: pd.DataFrame,
    x: str,
    row: str,
    col: str,
    row_order: Optional[list[str]] = None,
    col_order: Optional[list[str]] = None,
    use_row_as_color: bool = False,
    xlim: Optional[tuple[int, int]] = None,
    ylim: Optional[tuple[int, int]] = None,
    binwidth: Optional[float] = None,
    row_title: Optional[str] = None,
    col_title: Optional[str] = None,
    fig_title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    if row_title is None:
        row_title = row
    if col_title is None:
        col_title = col
    row_levels = row_order if row_order else df[row].unique()
    nrows = len(row_levels)
    col_levels = col_order if col_order else df[col].unique()
    ncols = len(col_levels)
    palette = sns.color_palette(n_colors=nrows if use_row_as_color else ncols)
    fig, axes = plt.subplots(
        nrows * 3,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [0.05, 0.85, 0.1] * nrows},
    )
    for r, row_level in enumerate(row_levels):
        for c, col_level in enumerate(col_levels):
            row_col_df = df[(df[row] == row_level) & (df[col] == col_level)]
            spacer_row, hist_row, box_row = range(r * 3, r * 3 + 3)
            color = palette[r if use_row_as_color else c]

            axes[spacer_row][c].axis("off")

            sns.histplot(
                data=row_col_df,
                x=x,
                ax=axes[hist_row][c],
                color=color,
                binwidth=binwidth,
            )
            axes[hist_row][c].tick_params(
                axis="x", which="both", bottom=True, top=False, labelbottom=True
            )
            axes[hist_row][c].set_title(
                f"{row_title} = {row_level}, {col_title}={col_level}"
            )
            if xlim:
                axes[hist_row][c].set_xlim(xlim)
            if ylim:
                axes[hist_row][c].set_ylim(ylim)

            sns.boxplot(
                row_col_df,
                x=x,
                orient="h",
                ax=axes[box_row][c],
                color=color,
                fliersize=1,
            )
            axes[box_row][c].set(xlabel="", ylabel="")
            axes[box_row][c].tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            if xlim:
                axes[box_row][c].set_xlim(xlim)
    if fig_title:
        fig.suptitle(fig_title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def zero_error_bars(vals):
    """
    Create (low, high) error bars for provided values arrays.

    :param vals: array, array of values
    :return array of error bars of shape (2, *vals.shape)
    """
    return np.zeros((2, *vals.shape), dtype=vals.dtype)


def init_and_clip_err(vals, err, bounds=(0, 1)):
    """
    Create err array (low_err and up_err) clipped between specified bounds also from 1d error array.

    :param vals: array, array of values
    :param err: array, error values, shape can be 1x... (will substract/add from values) or 2x... (will clip only)
    :return array, clipped error array of shape (2, *vals.shape)
    """
    if err is None:
        err = zero_error_bars(vals)
    if err.shape == (2, *vals.shape):
        return err.clip(*bounds)
    else:
        return np.abs(
            np.vstack([vals - err, vals + err]).clip(*bounds) - np.vstack([vals, vals])
        )


def get_error_display_params():
    """
    Return usual error display parameters.

    :return dict, error display parameters
    """
    return {"elinewidth": 0.8, "capsize": 2, "capthick": 0.5}


def show_graph_values(ax, values, pos_x, pos_y=None, yerr=None, fontsize=6):
    """
    Write values on provided axis.

    :param ax: axis
    :param values: array, size 1xN
    :param pos_x: array, x positions, size 1xN
    :param pos_y: array, optional, size 1xN
    :param yerr: array, optional y positions errors (for error range), size 1xN
    :param fontsize: int
    """
    pos_y = values if pos_y is None else pos_y
    yerr = np.zeros(values.shape) if yerr is None else yerr
    for x, y, v, ye in zip(pos_x, pos_y, values, yerr):
        ax.text(x, y + ye, v, fontsize=fontsize)


def plot_lines_with_err(
    ax,
    xs,
    ys,
    labels,
    yerrs=None,
    xerrs=None,
    show_vals=None,
    legend_loc="lower center",
    err_bounds=(0, 1),
):
    """
    Plot multiple lines on provided axis.

    :param ax: axis
    :param xs: array, x positions, size BxN
    :param ys: array, y positions, size BxN
    :param labels: list, line labels, size B
    :param yerrs: array, optional y positions errors, size 1xN or 2xN
    :param xerrs: array, 1d optional x positions errors, size 1xN or 2xN
    :param show_vals: bool, whether to print plotted values above points
    :param legend_loc: str, legend location
    :param err_bounds: tuple (int, int), error boundary
    """
    if yerrs is None:
        yerrs = [zero_error_bars(y) for y in ys]
    if xerrs is None:
        xerrs = [zero_error_bars(x) for x in xs]
    for x, y, label, yerr, xerr in zip(xs, ys, labels, yerrs, xerrs):
        yerr = init_and_clip_err(y, yerr, err_bounds)
        xerr = init_and_clip_err(x, xerr, err_bounds)
        ax.errorbar(
            x, y, yerr=yerr, xerr=xerr, label=label, **get_error_display_params()
        )
        if show_vals is not None:
            show_graph_values(ax, show_vals, x, pos_y=y, yerr=yerr[1])
    ax.legend(loc=legend_loc)


def grouped_barplot_with_err(
    ax,
    stats,
    groupLabels,
    xlabel=None,
    ylabel=None,
    title=None,
    barwidth=0.35,
    show_val=False,
    title_loc=None,
    legend_loc="lower center",
    err_bounds=(0, 1),
):
    """
    Create barplot with several groups.

    :param ax: axis
    :param stats: dict, B keys the bar name (e.g. accuracy) with values 1xN for each group (e.g. res for each cls)
    :param groupLabels: list, labels for each group, size B
    :param xlabel: list, optional, size N
    :param ylabel: list, optional, size N
    :param title: str, option, plot title
    :param barwidth: float
    :param show_vals: bool, whether to print plotted values
    :param title_loc: str, optional title location (left, right, center)
    :param legend_loc: str, legend location
    :param err_bounds: tuple (int, int), error boundary
    """
    nbars_per_group, ngroups = len(stats.keys()), len(groupLabels)
    group_width = nbars_per_group * barwidth * 4 / 3
    positions = np.arange(0, ngroups * group_width, group_width)
    offsets = [
        (i - nbars_per_group / 2) * barwidth + barwidth / 2
        for i in range(nbars_per_group)
    ]

    ekw = get_error_display_params()
    cmap = plt.cm.get_cmap("Dark2", nbars_per_group)
    for offset, (key, (vals, err)), c in zip(
        offsets, stats.items(), [cmap(i) for i in range(nbars_per_group)]
    ):
        err = init_and_clip_err(vals, err, err_bounds)
        ax.bar(
            positions + offset,
            vals,
            color=c,
            width=barwidth,
            yerr=err,
            label=key,
            error_kw=ekw,
            edgecolor="white",
        )
        if show_val:
            show_graph_values(
                ax,
                [f"{v:.2f}" for v in vals],
                positions + offset - barwidth / 2,
                pos_y=vals,
                yerr=err[1],
            )

    ax.set_ylabel("Performance" if ylabel is None else ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(groupLabels, rotation=35, ha="right")
    ax.legend(loc=legend_loc)
    if title is not None:
        ax.set_title(title, loc=title_loc)
    ax.grid(False)


def plot_confusion_matrix(ax, cm_std, labels, title=None, title_loc=None):
    """
    Plot confusion matrix with std values below.

    Code inspired from fastai library.
    :param ax: axis
    :param cm_std_ tuple (array, array), conf mat values and conf mat std size (NxN, NxN)
    :param labels: list, confusion matrix labels, size N
    :param title: str, optional, plot title
    :param title_loc: str, optional title location (left, right, center)
    """
    cm, std = cm_std
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, rotation=0)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # coeff = f'{cm[i, j]:.{2}f} \u00B1 {std[i, j]:.{2}f}'
        ax.text(
            j,
            i - 0.15,
            f"{cm[i, j]:.{2}f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=8,
        )
        ax.text(
            j,
            i + 0.15,
            f"\u00B1 {std[i, j]:.{2}f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=6,
        )

    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if title is not None:
        ax.set_title(title, loc=title_loc)
    ax.grid(False)


def new_fig_with_axs(nrows, ncols, base_fig_width, base_fig_height=None, **kwargs):
    """
    Create new fig according to provided params.

    :param nrows: int, number of rows
    :param ncols: int, number of columns
    :param base_fig_width: int, base width of fig in inches
    :param base_fig_height: int, base height of fig in inches
    :param kwargs: dict, keywords argument for plt.subplots
    :return: (fig, axs), output of plt.subplots
    """
    if base_fig_height is None:
        base_fig_height = base_fig_width
    return plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * base_fig_width, nrows * base_fig_height),
        **kwargs,
    )


def plt_fig_as_np_array(fig):
    """
    Convert matplotlib figure in image array.

    :param fig: plt figure
    :return: image array of matplotlib figure
    """
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    return img_arr


def show_im_with_zoom_region(
    ax, im, zoom_loc=[0.35, 0.5, 0.35, 0.5], inset_loc=[0.6, 0.6, 0.4, 0.4]
):
    """
    Plot image with zoom region.

    :param ax: axis
    :param im: array, image
    :param zoom_loc: list, image zoom location
    :param inset_loc: list, where to display zoom
    """
    ax.imshow(im, origin="lower")
    ax.axis("off")
    axins = ax.inset_axes(inset_loc)
    axins.imshow(im, origin="lower")
    h, w = im.shape[:2]
    x1, x2, y1, y2 = map(int, np.array(zoom_loc) * np.array([w, w, h, h]))
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels("")
    axins.set_xticks([])
    axins.set_yticklabels("")
    axins.set_yticks([])
    ax.indicate_inset_zoom(axins, edgecolor="black")


def plt_save_fig(path, fig=None, close=True, **kwargs):
    """
    Save matplotlib figure.

    :param path: str, save path
    :param fig: figure, optional if none will use current fig
    :param close: bool, close figure after saving
    :param kwargs: dict, keywords arguments for plt.savefig
    """
    if fig is None:
        plt.savefig(path, bbox_inches="tight", **kwargs)
        if close:
            plt.close(plt.gcf())
    else:
        fig.savefig(path, bbox_inches="tight", **kwargs)
        if close:
            plt.close(fig)


def prepare_img_axs(
    h_w_ratio, nrows, ncols, figsize_fact=8, no_axis=True, flatten=True, title=None
):
    """
    Create figure with size fitting provided h/w ratio.

    :param h_w_ratio: float, usually image height / width
    :param nrows: int, number of rows
    :param ncols: int, number of columns
    :param figsize_fact: float, factor to increase size from provided ratio
    :param no_axis: bool, hide axis
    :param flatten: bool, flatten subplots axes
    :param title: str, general title
    :return: fig, axs
    """
    fig, axs = new_fig_with_axs(nrows, ncols, figsize_fact)
    base_figsize = (ncols * figsize_fact, nrows * figsize_fact * h_w_ratio)
    plt.rcParams["font.size"] = max(base_figsize) * 0.4
    nd = nrows > 1 or ncols > 1
    if no_axis:
        for ax in axs.flatten() if nd else [axs]:
            ax.axis("off")
    if nd and flatten:
        axs = axs.flatten()
    if title is not None:
        fig.suptitle(title, fontsize=base_figsize[0] * 1.4)
    return fig, axs


def img_on_ax(im, ax, title=None):
    """
    Plot image on axis.

    :param im: array, image
    :param ax: axis
    :param title: str, title of fig
    """
    ax.imshow(im)
    ax.set_title(title)
    ax.axis("off")


def plt_show_img(im, title="", show=True, save_path=None):
    """
    Plot image.

    :param im: array, image
    :param title: str, title of fig
    :param show: bool, call plt.show()
    :param save_path: str, if not None, saves fig in path
    """
    fig, ax = plt.subplots()
    img_on_ax(im, ax, title=title)
    if save_path is not None:
        plt_save_fig(save_path, close=False)
    if show:
        fig.show()
