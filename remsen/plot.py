from typing import List, Tuple

from matplotlib import cm, pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.backend_bases import register_backend
from matplotlib.ticker import PercentFormatter

import numpy as np

import pandas as pd

from remsen.training import tensorboard_dataframe
from remsen import utils


def configure_latex(scaler: float = 1.5, width_scaler: float = 1):
    """Configure matplotlib for LaTeX figure output."""
    register_backend('pdf', FigureCanvasPgf)
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
            ],
            "figure.figsize": [width_scaler * scaler * 3.39, scaler * 2.0951352218621437],
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


def plot_bbox_distribution() -> None:
    configure_latex(width_scaler=1.6)

    # Fetch pre-computed bbox stats from QGIS dump
    stats = pd.read_csv("data/bbox_stats.csv")

    # Calculate max(width, height) of each bbox
    stats["max"] = stats.loc[:, ["width", "height"]].max(axis=1)

    # Calculate 90 percentile axis cut-off
    percentile = stats["max"].quantile(q=0.9)

    fig, (width_ax, height_ax, max_ax) = plt.subplots(
        1, 3,
        sharex=True,
        sharey=True,
    )
    for dimension, axis in [
        ("width", width_ax),
        ("height", height_ax),
        ("max", max_ax),
    ]:
        # Plot distribution of dimension extent
        axis.hist(
            stats[dimension],
            bins=100,
            range=(0, percentile),
            density=True,
        )

        # Show 64 meter cut-off
        axis.axvline(x=64, color="r", linestyle=":")

        # Plot arrow pointing to left indicating sum under curve to the left
        axis.annotate(
            "",
            xy=(64, 0.03),
            xytext=(44, 0.03),
            arrowprops=dict(arrowstyle="<-", color="red"),
        )

        # Calculate sum under curve to left of cut-off
        proportion = 100 * (stats[dimension] < 64).sum() / len(stats)

        # Annotate sum under curve
        axis.annotate(
            f"${proportion:.1f}" + r"\%$",
            xy=(47.5, 0.031),
        )
        axis.set_xlabel(dimension.capitalize() + f" ${dimension[0]}$ [$m$]")

    # Show percentages on y-axis
    width_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    width_ax.set_ylabel("Fraction of bounding boxes")
    fig.tight_layout()
    fig.savefig("/code/tex/img/bbox_stats.pdf")


def plot_gdalinfo_histogram(path, bands=("red", "green", "blue")):
    """Plot the result of gdalinfo -hist ${raster_file} > ${path}"""
    lines = path.read_text().split("\n")
    band_data = {}
    for i, line in enumerate(lines):
        if " buckets from " in line:
            buckets = lines[i].split()
            color_interp = lines[i - 3].split("=")[-1].lower()
            band_data[color_interp] = {
                "bin_num": int(buckets[0]),
                "bin_min": float(buckets[3]),
                "bin_max": float(buckets[5][:-1]),
                "frequency": np.array(list(map(int, lines[i + 1].split())))
            }

    fig, ax = plt.subplots()
    for band in bands:
        data = band_data[band]
        frequency = data["frequency"]
        print(frequency.sum())
        frequency = frequency / frequency.sum()
        bin_edges = np.linspace(
            start=data["bin_min"],
            stop=data["bin_max"],
            num=data["bin_num"] + 1,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(bin_centers, frequency)
        ax.set_ylim(0, 0.0085)

    return fig, ax
