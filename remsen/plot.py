from typing import List, Tuple

from matplotlib import cm, pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.backend_bases import register_backend

import numpy as np

from remsen.training import tensorboard_dataframe
from remsen import utils


def configure_latex(scaler: float = 1.5):
    """Configure matplotlib for LaTeX figure output."""
    register_backend('pdf', FigureCanvasPgf)
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
            ],
            "figure.figsize": [scaler * 3.39, scaler * 2.0951352218621437],
            "text.usetex": True,
            "backend": "ps",
        }
    )


def plot_training(
    names: List[str],
    splits: List[str] = ["validation"],
    metric: str = "iou",
) -> None:
    """Plot training sequence from TensorBoard for given model/split/metric."""
    configure_latex()
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\mathrm{Epoch}$")

    if metric == "iou":
        ax.set_ylabel(r"$\mathrm{IoU}$")
    else:
        ax.set_ylabel(metric)

    for name in names:
        for split in splits:
            training_data = tensorboard_dataframe(name=name, split=split)
            metric_series = training_data["val_iou"]
            ax.plot(metric_series)

    fig.tight_layout()
    filename = (
        "+".join(sorted(names)) + "-" + "+".join(sorted(splits)) + "-" + metric
    )
    path = "/code/tex/img/metrics/" + filename + ".pdf"
    fig.savefig(path)
    return fig, ax


def imshow_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    ax,
    cmap: str = None,
    edge_color: Tuple[int, int, int, int] = (255, 0, 0, 180),
):
    """Plot image with overlayed mask."""
    image = image.copy()
    grayscale = image.shape[-1] != 3
    if grayscale:
        image = cm.ScalarMappable(cmap=cmap).to_rgba(image, bytes=True)
    else:
        image = np.concatenate(
            [image, 255 * np.ones_like(image[:, :, 0:1])],
            axis=2,
        )

    # Plot the original image
    ax.imshow(image)

    # Plot edge of mask with given RGBA value
    edges = utils.edge_pixels(np.squeeze(mask))
    image[:, :, :][edges] = edge_color
    ax.imshow(image)
