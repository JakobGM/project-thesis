from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from remsen.plot.utils import configure_latex
from remsen.training import tensorboard_dataframe


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
    labels.append("Validation")
    labels.append("Train")
    ax.legend(
        handles,
        labels,
        ncol=3,
        loc="upper right" if metric == "loss" else "lower right",
    )

    # Save to deterministic filepath
    filename = (
        "+".join(sorted(names)) + "-" + "+".join(sorted(splits)) + "-" + metric
    )
    path = "/code/tex/img/metrics/" + filename + ".pdf"
    fig.tight_layout()
    fig.savefig(path)
    return fig, ax
