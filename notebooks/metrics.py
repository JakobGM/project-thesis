from typing import Tuple

from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import numpy as np

import pandas as pd


def conditions(y_true: np.ndarray) -> Tuple[int, int]:
    positives = int(np.sum(y_true))
    negatives = y_true.size - positives
    return positives, negatives


def confusions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[int, int, int, int]:
    tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
    tn = int(np.logical_and(y_pred == 0, y_true == 0).sum())
    fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
    fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())
    return tp, tn, fp, fn


def confusion_matrix(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[int, int, int, int]:
    tp, tn, fp, fn = confusions(y_pred=y_pred, y_true=y_true)
    return pd.DataFrame(
        data=[[tp, fp], [fn, tn]],
        index=["Positive Prediction", "Negative Prediction"],
        columns=["Condition Positive", "Condition Negative"],
    )


def sensitivity_and_specificity(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float]:
    tp, tn, fp, fn = confusions(y_pred=y_pred, y_true=y_true)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[int, int]:
    tp, tn, fp, fn = confusions(y_pred=y_pred, y_true=y_true)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def iou(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[int, int]:
    tp, tn, fp, fn = confusions(y_pred=y_pred, y_true=y_true)
    iou = tp / (tp + fp + fn)
    return iou


def plot_conditions(
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

    fig, ax = plt.subplots(1, 1)
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
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.125),
        frameon=False,
        handlelength=1.3,
        handleheight=1.5,
    )

    plt.tight_layout()
    plt.show()
