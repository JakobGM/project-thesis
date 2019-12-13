from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from remsen.plot.utils import configure_latex
from remsen.training import Trainer, tensorboard_dataframe


def plot_training(
    names: List[str],
    splits: List[str] = ["validation"],
    metric: str = "iou",
    ylim: Optional[Tuple] = None,
    labels: Optional[Dict] = None,
) -> None:
    """Plot training sequence from TensorBoard for given model/split/metric."""
    configure_latex(scaler=1.75)
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
    comparison = len(splits) > 1
    if multiple_splits:
        labels.append("Validation")
        labels.append("Train")
        title = None
    else:
        title = splits[0].capitalize() + " metric"

    ax.legend(
        handles,
        labels,
        ncol=3 if not comparison and multiple_splits else 2,
        loc="upper right" if metric == "loss" else "lower right",
        title=title,
    )

    # Save to deterministic filepath
    filename = (
        "+".join(sorted(names)) + "-" + "+".join(sorted(splits)) + "-" + metric
    )
    path = "/code/tex/img/metrics/" + filename + ".pdf"
    fig.tight_layout()
    fig.savefig(path)
    return fig, ax


def metric_correlation(
    x_model: str,
    y_model: str,
    metric: str = "iou",
    splits: Iterable[str] = ("train", "test"),
    labels: Optional[Tuple[str, str]] = None,
    minimum_building_area: float = 4,
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
        ax.scatter(
            x,
            y,
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

    # Semantic path for this plot
    save_path = Path(
        "tex/img/metric_correlation/"
        f"{x_model}+{y_model}"
        f"+{metric}.pdf"
    )

    # We have used rasterized=True in scatter, so we need to increase DPI
    # before saving.
    mpl.rcParams['savefig.dpi'] = 300
    fig.savefig(save_path)
