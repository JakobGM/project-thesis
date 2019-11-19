from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ipypb import track

from matplotlib import cm, pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.backend_bases import register_backend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

import numpy as np

import pandas as pd

from scipy import ndimage

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
    configure_latex()
    rgb = len(bands) == 3
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
    for band in reversed(bands):
        data = band_data[band]
        frequency = data["frequency"]
        frequency = frequency / frequency.sum()
        # Fix binning error in gdalinfo
        if rgb:
            frequency[64] /= 2
            frequency[191] /= 2
            frequency[-3] /= 2
        else:
            frequency[frequency[1:].argmax() + 1] /= 2

        bin_edges = np.linspace(
            start=data["bin_min"],
            stop=data["bin_max"],
            num=data["bin_num"] + 1,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        frequency = frequency / (bin_centers[10] - bin_centers[9])
        ax.plot(
            bin_centers[:-1],
            frequency[:-1],
            color=None if not rgb else band,
            label="Elevation" if not rgb else band.capitalize(),
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Channel')

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    ax.set_ylabel("Frequency")

    img_path = Path("/code/tex/img")
    if rgb:
        ax.set_xlabel(r"Channel value")
        fig.savefig(str(img_path / "rgb-density.pdf"))
    else:
        ax.set_ylim(0, 0.004)
        ax.set_xlabel(r"Elevation [m]")
        fig.savefig(str(img_path / "elevation-density.pdf"))

    return fig, ax


def plot_raster_spread(cache):
    minima, mean, maxima = [], [], []
    for index in track(cache.cadastre_indeces):
        if index > 10_000:
            break
        tiles = cache.lidar_tiles(cadastre_index=index)
        for tile in tiles:
            tile = tile.flatten()
            if tile.sum() <= 64:
                continue
            minima.append(tile.min())
            mean.append(tile.mean())
            maxima.append(tile.max())

    minima = np.array(minima)
    mean = np.array(mean)
    maxima = np.array(maxima)

    sorter = mean.argsort()
    minima = minima[sorter]
    mean = mean[sorter]
    maxima = maxima[sorter]

    configure_latex(width_scaler=2)
    fig, (sorted_ax, dist_ax) = plt.subplots(1, 2)
    x = np.arange(len(minima))
    sorted_ax.fill_between(
        x,
        mean,
        minima,
        label="Minima",
        color="g",
        facecolor="none",
        alpha=0.5,
    )
    sorted_ax.fill_between(
        x,
        mean,
        maxima,
        label="Maxima",
        color="r",
        facecolor="none",
        alpha=0.5,
    )
    sorted_ax.plot(x, mean, label="Mean", color="xkcd:black", linewidth=1.5)

    sorted_ax.legend(loc="upper left")
    sorted_ax.set_xlabel("Tiles sorted by mean elevation")
    sorted_ax.set_ylabel("Tile elevation [m]")

    range = maxima - minima
    dist_ax.hist(
        range,
        bins=200,
        density=True,
        label=(
            r"\textbf{Mean:} " f"{range.mean():.2f}m, "
            r"\textbf{SD:} " f"{range.std():.2f}m"
        )
    )
    dist_ax.set_xlabel(r"Range ($\max - \min$) [m]")
    dist_ax.legend()

    dist_ax.set_ylabel(r"Frequency")
    dist_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    sorted_ax.get_xaxis().set_ticks([])
    fig.savefig("/code/tex/img/elevation-spread.pdf")
    return fig, sorted_ax


def plot_segmentation_types(
    cache,
    cadastre_index: int = 28,
    tile_index: int = 0,
    alpha: int = 164,
):
    """Plot the different types of object detection types."""
    configure_latex(scaler=3)
    cache.change_dataset()
    rgb_tile = cache.rgb_tiles(cadastre_index=cadastre_index)[tile_index]
    mask_tile = cache.mask_tiles(cadastre_index=cadastre_index)[tile_index]

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    bbox_ax, semseg_ax, instance_ax = ax

    # Plot aerial photography as background image and remove ticks
    for axis in ax:
        axis.imshow(np.squeeze(rgb_tile))
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

    # Plot building outline bounding boxes
    labeled_array, num_features = ndimage.label(np.squeeze(mask_tile))
    object_slices = ndimage.find_objects(labeled_array)
    for object_slice in object_slices:
        indexer = np.zeros((256, 256), dtype="bool")
        indexer[object_slice] = 1
        indexer = np.where(indexer != 0)
        ymin, ymax, xmin, xmax = (
            np.min(indexer[0]),
            np.max(indexer[0]),
            np.min(indexer[1]),
            np.max(indexer[1]),
        )
        rectangle = Rectangle(
            xy=(xmin, ymin),
            width=xmax - xmin,
            height=ymax - ymin,
            fill=False,
            color="red",
            linewidth=3,
        )
        bbox_ax.add_artist(rectangle)
    bbox_ax.set_title("Bounding Box Regression")

    # Plot semantic segmentation masks
    rgb_tile = np.concatenate([rgb_tile, np.zeros((256, 256, 1), dtype="uint8")], axis=2)
    semseg_tile = rgb_tile.copy()
    indexer = np.squeeze(mask_tile == 1)
    semseg_tile[indexer] = (255, 0, 0, alpha)
    # Add edge outline
    edges = utils.edge_pixels(indexer)
    semseg_tile[edges] = (255, 0, 0, 255)
    semseg_ax.imshow(semseg_tile)
    semseg_ax.set_title("Semantic Segmentation")

    # Plot instance segmentation masks
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    instance_tile = rgb_tile.copy()
    for index, object_slice in enumerate(object_slices):
        color = colors[index].strip("#")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)

        indexer = np.zeros((256, 256), dtype="bool")
        indexer[object_slice] = True
        indexer = np.logical_and(indexer, np.squeeze(mask_tile.astype("bool")))
        instance_tile[indexer] = rgb

        # Add edge outline
        edges = utils.edge_pixels(indexer)
        instance_tile[edges] = rgb[:3] + (255,)
    instance_ax.imshow(instance_tile)
    instance_ax.set_title("Instance Segmentation")

    fig.tight_layout()
    fig.savefig("/code/tex/img/segmentation-types.pdf")
    return fig, ax
