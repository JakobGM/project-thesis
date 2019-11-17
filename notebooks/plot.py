from pathlib import Path

from matplotlib import colors, pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from remsen.plot import configure_latex


def plot_conditions(
    ax,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    cutoff: float = 0.5,
) -> None:
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)

    predicted_mask = (y_pred > cutoff).astype("uint8")
    tp = np.logical_and(predicted_mask == 1, y_true == 1)
    tn = np.logical_and(predicted_mask == 0, y_true == 0)
    fp = np.logical_and(predicted_mask == 1, y_true == 0)
    fn = np.logical_and(predicted_mask == 0, y_true == 1)
    confusion_matrix = tp + 2 * tn + 3 * fp + 4 * fn

    cmap = colors.ListedColormap(
        ['#001F3F', '#DDDDDD', '#2ECC40', '#FF4136']
    )
    bounds = [0, 1.5, 2.5, 3.5, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.imshow(confusion_matrix, cmap=cmap, norm=norm)

    # Add TP/TN/FP/FN legend to plot
    legend_elements = [
        Patch(facecolor='#001F3F', edgecolor="white", label='TP'),
        Patch(facecolor='#DDDDDD', edgecolor="white", label='TN'),
        Patch(facecolor='#2ECC40', edgecolor="white", label='FP'),
        Patch(facecolor='#FF4136', edgecolor="white", label='FN'),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.125),
        frameon=False,
        handlelength=1.3,
        handleheight=1.5,
    )

    plt.tight_layout()
    plt.show()


def plot_prediction_vs_ground_truth():
    configure_latex()
    mask_width, mask_height = 256, 256
    num_classes = 1

    y_true = np.zeros(
        shape=(mask_height, mask_width, num_classes),
        dtype="uint8",
    )
    y_pred = np.zeros(
        shape=(mask_height, mask_width, num_classes),
        dtype="uint8",
    )

    y_true[80:200, 120:200] = 1
    y_pred[50:170, 90:170] = 1

    fig, (truth_ax, pred_ax, metric_ax) = plt.subplots(
        1,
        3,
        sharex=True,
        sharey=True,
        figsize=(0.7 * 15, 0.7 * 5),
    )

    truth_ax.set_title("Ground Truth")
    pred_ax.set_title("Predicted Mask")

    truth_ax.imshow(np.squeeze(y_true), cmap="binary")
    pred_ax.imshow(np.squeeze(y_pred), cmap="binary")

    for axis in (truth_ax, pred_ax):
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

    plot_conditions(ax=metric_ax, y_pred=y_pred, y_true=y_true)
    path = Path(__file__).parents[1] / "tex" / "img" / "confusions.pdf"
    fig.tight_layout()
    fig.savefig(str(path))
    return fig, (truth_ax, pred_ax)
