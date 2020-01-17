from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from remsen.plot.utils import configure_latex, get_colors
from remsen.training import Trainer, tensorboard_dataframe


def plot_training(
    names: List[str],
    splits: List[str] = ["validation"],
    metric: str = "iou",
    ylim: Optional[Tuple] = None,
    labels: Optional[Dict] = None,
    save: bool = False,
    scaler: float = 1.33,
) -> None:
    """Plot training sequence from TensorBoard for given model/split/metric."""
    configure_latex(scaler=scaler)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\mathrm{Epoch}$")

    if metric == "iou":
        latex_metric = r"$\mathrm{IoU}$"
        ax.set_ylabel(latex_metric)
    else:
        latex_metric = r"$\mathrm{" + metric.capitalize() + "}$"
        ax.set_ylabel(latex_metric)

    if not labels:
        labels = {name: name.replace("_", " ").capitalize() for name in names}

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    training_data = {name: tensorboard_dataframe(name=name) for name in names}

    # Clip training series to simulate same number of epochs
    min_epoch = 1e8
    for data in training_data.values():
        min_epoch = min(min_epoch, data.index.max())
    for data in training_data.values():
        data.drop(index=data.index[data.index >= min_epoch], inplace=True)

    for i, name in enumerate(names):
        if "train" in splits:
            metrics = training_data[name]["train_" + metric]
            ax.plot(
                metrics,
                color=colors[i],
                linestyle="--",
                dashes=(5, 1),
                alpha=0.75,
            )
        if "validation" in splits:
            metrics = training_data[name]["val_" + metric]
            ax.plot(
                metrics,
                color=colors[i],
                label=labels[name],
            )

            if metric == "loss":
                best_index = metrics.idxmin()
            else:
                best_index = metrics.idxmax()

            best_metric = metrics[best_index]
            ax.plot(
                best_index,
                best_metric,
                marker="o",
                color=colors[i],
                zorder=1000000,
                markerfacecolor=colors[i],
                markeredgecolor="xkcd:black",
                label=f"{latex_metric}: {best_metric:.4f}",
            )

    if ylim:
        ax.set_ylim(*ylim)

    # Customize legend to show color group and linestyle group
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="xkcd:black"))
    handles.append(Line2D([0], [0], color="xkcd:black", linestyle="--"))

    multiple_splits = len(splits) != 1
    if multiple_splits:
        labels.append("Validation")
        labels.append("Train")
        title = None
        ncol = len(names) + 1
    else:
        title = splits[0].capitalize() + " metric"
        ncol = len(names)

    ax.legend(
        handles,
        labels,
        ncol=ncol,
        loc="upper right" if metric == "loss" else "lower right",
        title=title,
    )

    # Save to deterministic filepath
    filename = (
        "+".join(sorted(names)) + "-" + "+".join(sorted(splits)) + "-" + metric
    )
    fig.tight_layout()
    if save:
        path = "/code/tex/img/metrics/" + filename + ".pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
    return fig, ax


def metric_correlation(
    x_model: str,
    y_model: str,
    metric: str = "iou",
    splits: Iterable[str] = ("train", "test"),
    labels: Optional[Tuple[str, str]] = None,
    minimum_building_area: float = 4,
    mask_color: bool = False,
    save: bool = False,
):
    # Enable LaTeX rendering of plots
    configure_latex(scaler=2)

    # Retrieve relevant data
    columns = ["cadastre", "tile", "split", "mask", metric]
    dfx = Trainer.evaluation_statistics(name=x_model)[columns]
    dfy = Trainer.evaluation_statistics(name=y_model)[columns]

    # Remove all rows with building area less than the given amount
    dfx = dfx[dfx["mask"] > minimum_building_area * 16]

    # Join the two model results
    metrics = dfx.merge(dfy, on=["cadastre", "tile", "split"], how="inner")

    # Indexable colors from the current color scheme
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, (axes,) = plt.subplots(
        1,
        len(splits),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for split, ax in zip(splits, axes):
        # Annotate the given split
        ax.set_title(r"\textbf{" + split.capitalize() + " split}")

        # Only use data from given split
        split_metrics = metrics[metrics.split == split]

        # Create scatter plot of x model performance vs. y model
        x = split_metrics[metric + "_x"]
        y = split_metrics[metric + "_y"]
        building_area = 100 * split_metrics["mask_x"] / (256 ** 2)
        color_plot = ax.scatter(
            x,
            y,
            c=building_area if mask_color else None,
            vmin=0,
            vmax=100,
            alpha=0.2,
            edgecolor="",
            rasterized=True,
        )

        # Plot dummy points in each corner in order to have correct
        # limits. Dirty hack, but easiest way to do it.
        ax.scatter([0, 0, 1, 1], [0, 1, 0, 1], s=0)

        # Both axes represents the same metric, so 1:1 aspect makes sense
        ax.set_aspect(1)

        # All further plotting should not change the axis limits
        ax.set_xlim(auto=False)
        ax.set_ylim(auto=False)

        # Plot the line where models perform equally
        ax.plot([-10, 10], [-10, 10], color="black")

        # How often is the x model better than the y model?
        x_better = x > y
        x_better = 100 * x_better.sum() / len(x_better)

        # Annotate split in across x = y with arrows
        ax.arrow(x=0.2, y=0.2, dx=0.1, dy=-0.1, color=colors[2], head_width=0.025)
        ax.arrow(x=0.2, y=0.2, dx=-0.1, dy=0.1, color=colors[1], head_width=0.025)

        # Annotate the proportion of when x is better than y
        ax.text(
            x=0.2 + 0.07,
            y=0.2 - 0.05,
            s=r"$\mathbf{" + str(round(x_better, 1)) + r"\%}$",
            color=colors[2],
            fontsize=12,
        )

        # Annotate the proportion of when y is better than x
        ax.text(
            x=0.2 - 0.06,
            y=0.2 - 0.0,
            s=r"$\mathbf{" + str(round(100 - x_better, 1)) + r"\%}$",
            color=colors[1],
            fontsize=12,
            horizontalalignment='right',
        )

        # Annotate mean IoU of each model
        x_mean = x.mean()
        y_mean = y.mean()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.plot(
            [x_min, y_mean],
            [y_mean, y_mean],
            color=colors[1],
            linestyle="--",
        )
        ax.vlines(
            x=x_mean,
            ymin=y_min,
            ymax=x_mean,
            color=colors[2],
            linestyle="--",
        )
        ax.text(
            x=x_min + 0.03,
            # y=y_mean - 0.02,
            y=y_mean + 0.5 * (y_max - y_mean),
            s=r"$\mathbf{\overline{IoU} = " + f"{y_mean:0.3f}" + r"}$",
            color=colors[1],
            fontsize=12,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.text(
            x=x_mean + 0.5 * (x_max - x_mean),
            y=y_min + 0.1,
            s=r"$\mathbf{\overline{IoU} = " + f"{x_mean:0.3f}" + r"}$",
            color=colors[2],
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation=-90,
        )

    # Apply labels if provided
    if labels:
        fig.text(0.53, 0.06, labels[0].replace("IoU", r"[$\mathrm{IoU}$]"), ha="center")
        axes[0].set_ylabel(labels[1].replace("IoU", r"[$\mathrm{IoU}$]"))

    # Reduce padding between subplots
    fig.tight_layout()
    if mask_color:
        # [left, bottom, width, height]
        colorbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
        fig.colorbar(
            color_plot,
            cax=colorbar_ax,
            format=PercentFormatter(),
        )

    # Semantic path for this plot
    save_path = Path(
        "tex/img/metric_correlation/"
        f"{x_model}+{y_model}"
        f"+{metric}{'+color' if mask_color else ''}.pdf"
    )

    # We have used rasterized=True in scatter, so we need to increase DPI
    # before saving.
    mpl.rcParams['savefig.dpi'] = 300
    if save:
        fig.savefig(save_path)


def plot_test_iou_summary(model: str, save: bool = False):
    # Configure plotting environment
    configure_latex(scaler=1.33)
    colors = get_colors()
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Test IoU")
    ax.set_ylabel("Number of tiles")

    # Filter down to only test tiles
    df = Trainer.evaluation_statistics(name=model)
    df = df[df.split == "test"]

    # Clip the left tail of the distribution in order to "zoom in" the histogram
    plot_df = df.copy()
    plot_df.iou.clip(lower=0.80, inplace=True)
    density, bins, patches = ax.hist(plot_df.iou, bins=50)

    # Indicate that the distribution has been clipped
    fraction = (df.iou <= 0.8).sum() / len(df)
    x, y = bins[1], density[0]
    ax.annotate(
        s=r"$\mathbf{IoU \leq 0.8}$" + f"\n$({100 * fraction:2.0f}" + r"\%)$",
        xy=(x + 0.0025, y / 2),
        horizontalalignment="left",
        color=colors[3],
    )
    patches[0].set_color(colors[3])

    # Plot interquartile range
    median = df.iou.median()
    ax.axvline(x=median, color=colors[1], label=f"Median = ${median:0.3f}$")
    lower_quantile, upper_quantile = df.iou.quantile(0.25), df.iou.quantile(0.75)
    ax.axvline(x=lower_quantile, color=colors[1], linestyle="--")
    ax.axvline(
        x=upper_quantile,
        color=colors[1],
        linestyle="--",
        label=f"IQR = $[{lower_quantile:0.2f}, {upper_quantile:0.3f}]$",
    )

    # Plot mean
    mean = df.iou.mean()
    ax.axvline(x=mean, color=colors[2], label=f"Mean = ${mean:0.3f}$")

    # Indicate left tail being clipped on x-axis labels
    xtickslabels = [f"{value:0.2f}" for value in ax.get_xticks()]
    xtickslabels[1] = r"$\leq 0.8$"
    ax.set_xticklabels(xtickslabels)

    ax.legend()
    fig.tight_layout()
    if save:
        save_dir = Path("tex/img/iou_distribution")
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / f"{model}.pdf"
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    return fig, ax
