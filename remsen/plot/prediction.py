from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib as mpl
from matplotlib import colors, patheffects, pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from tensorflow.keras.models import Model

from remsen.data import Dataset
from remsen.plot.data import imshow_with_mask
from remsen.plot.utils import configure_latex
from remsen.training import Trainer


def plot_prediction(
    cadastre_index: int = 0,
    tile_index: int = 0,
    dataset: Optional[Dataset] = None,
    model: Union[str, Model] = "without_rgb",
    save: bool = False,
):
    configure_latex()
    if not dataset:
        dataset = Dataset()

    if isinstance(model, str):
        model_name = model
        trainer = Trainer(
            name=model_name,
            model=None,
            dataset=dataset,
        )
        model = trainer.model
    else:
        model_name = model

    # Get all the data we could possibly need
    tiles = dataset.tiles(
        cadastre_index=cadastre_index,
        with_tile_dimensions=False,
        with_rgb=True,
    )
    lidar_tile = tiles["lidar"][tile_index]
    rgb_tile = tiles["rgb"][tile_index]
    building_tile = tiles["mask"][tile_index]
    input_tile = dataset.input_tile_normalizer(lidar_tile.copy())[0]

    # Determine the type of data required by the model
    num_channels = model.input.shape[-1]
    with_rgb = num_channels in (3, 4)
    with_lidar = num_channels in (1, 4)

    if with_rgb and with_lidar:
        input_tile = np.concatenate(
            [input_tile, rgb_tile / 255],
            axis=2,
        )
    elif with_rgb:
        input_tile = rgb_tile / 255
    else:
        input_tile = input_tile

    # Construct prediction for given input data
    input_tile = input_tile.reshape(1, 256, 256, -1)
    prediction = model.predict(input_tile)

    # (2, 2) plot if RGB+LiDAR, otherwise (1, 3)
    multiple_inputs = with_rgb and with_lidar
    fig, axes = plt.subplots(
        nrows=2 if multiple_inputs else 1,
        ncols=2 if multiple_inputs else 3,
        figsize=(15, 15),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    if with_rgb and with_lidar:
        lidar_ax, rgb_ax, prediction_ax, metric_ax = axes.flatten()
    elif with_lidar:
        lidar_ax, prediction_ax, metric_ax = axes.flatten()
    else:
        rgb_ax, prediction_ax, metric_ax = axes.flatten()

    # Add input data to subplots
    if with_lidar:
        decorate_lidar(
            ax=lidar_ax,
            lidar_tile=lidar_tile,
            building_tile=building_tile,
            cadastre_index=cadastre_index,
        )
    if with_rgb:
        decorate_rgb(
            ax=rgb_ax,
            rgb_tile=rgb_tile,
            building_tile=building_tile,
        )

    # Add prediction visualization to remaining subplots
    decorate_prediction(
        ax=prediction_ax,
        prediction=prediction,
        building_tile=building_tile,
    )
    decorate_confusions(
        ax=metric_ax,
        prediction=prediction,
        building_tile=building_tile,
        multiple_inputs=multiple_inputs,
    )
    decorate_metrics(
        ax=prediction_ax,
        x=input_tile,
        y=building_tile,
        model=model,
    )

    # Remove all ticks
    for axis in axes.flatten():
        axis.tick_params(labelbottom=False, labelleft=False, width=0.0)
    plt.tight_layout()
    if save:
        figure_id = f"{model_name}-{cadastre_index}-{tile_index}"
        save_path = f"tex/img/predictions/{figure_id}.pdf"
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

    return fig, axes


def decorate_lidar(ax, lidar_tile, building_tile, cadastre_index=None):
    """Plot LiDAR data on given axis."""
    ax.set_title(r"\textbf{LiDAR input}", fontsize=20)
    lidar_tile = lidar_tile.reshape(256, 256)
    ax = imshow_with_mask(image=lidar_tile, mask=building_tile, ax=ax)
    if cadastre_index is None:
        return ax

    cadastre_text = ax.annotate(
        f'Cadastre\n{cadastre_index}',
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        size=14,
        ha='right',
        va='top',
        color="white",
        weight="bold",
        alpha=0.8,
    )
    cadastre_text.set_path_effects(
        [patheffects.withStroke(linewidth=2, foreground='black', alpha=0.3)],
    )
    return ax


def decorate_rgb(ax, rgb_tile, building_tile):
    """Plot RGB data to given axis."""
    ax.set_title(r"\textbf{RGB input}", fontsize=20)
    imshow_with_mask(image=rgb_tile, mask=building_tile, ax=ax)


def decorate_prediction(ax, prediction, building_tile):
    """Plot prediction probabilities on given axis."""
    ax.set_title(r"\textbf{Sigmoid activations}", fontsize=20)
    prediction = prediction.reshape(256, 256)
    ax.imshow(
        prediction,
        cmap="RdYlBu",
        vmin=0,
        vmax=1,
    )
    imshow_with_mask(
        image=prediction,
        mask=building_tile,
        cmap="RdYlBu",
        ax=ax,
    )
    return ax


def decorate_confusions(ax, prediction, building_tile, multiple_inputs):
    """Plot confusion categories on given axis."""
    prediction = prediction.reshape(256, 256)
    predicted_mask = (prediction > 0.5).astype("uint8")
    building_tile = np.squeeze(building_tile)
    tp = np.logical_and(predicted_mask == 1, building_tile == 1)
    tn = np.logical_and(predicted_mask == 0, building_tile == 0)
    fp = np.logical_and(predicted_mask == 1, building_tile == 0)
    fn = np.logical_and(predicted_mask == 0, building_tile == 1)
    confusion_matrix = tp + 2 * tn + 3 * fp + 4 * fn

    cmap = colors.ListedColormap(
        ['#001F3F', '#DDDDDD', '#2ECC40', '#FF4136']
    )
    bounds = [0, 1.5, 2.5, 3.5, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(confusion_matrix, cmap=cmap, norm=norm)

    # Add TP/TN/FP/FN legend to plot
    legend_elements = [
        Patch(facecolor='#001F3F', edgecolor="white", label='TP'),
        Patch(facecolor='#DDDDDD', edgecolor="white", label='TN'),
        Patch(facecolor='#2ECC40', edgecolor="white", label='FP'),
        Patch(facecolor='#FF4136', edgecolor="white", label='FN'),
    ]
    # Quick hack to prevent misalignment of legend
    if multiple_inputs:
        legend_position = (0.5, -0.09)
    else:
        legend_position = (0.5, -0.15)

    ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=legend_position,
        frameon=False,
        handlelength=1.3,
        handleheight=1.5,
        prop={'size': 17},
    )
    ax.set_title(r"\textbf{TP, TN, FP, FN} (cut-off $ = 0.5$)", fontsize=20)
    return ax


def decorate_metrics(ax, x, y, model):
    """Add model prediction metrics to given axis."""
    # Add metrics to plot
    building_tile = np.expand_dims(y, 0)
    building_tile = np.expand_dims(building_tile, -1)
    evaluation = model.evaluate(
        x=x,
        y=building_tile,
        verbose=0,
    )
    metrics = {
        name: value
        for name, value
        in zip(model.metrics_names, evaluation)
    }
    loss = metrics["loss"]
    iou = metrics["iou"]
    ax.set_xlabel(
        r"$\mathrm{Loss} = "
        f"{loss:.4f},~~~"
        r"\mathrm{IoU} = "
        f"{iou:0.4f}$",
        size=20,
        labelpad=10,
    )
    return ax


def plot_median_prediction(name, save=False):
    """Plot the median prediction for a given model."""
    df = Trainer.evaluation_statistics(name=name)
    df = df[df.split == "test"]
    df = df.sort_values(by="iou")
    median = df.iloc[len(df) // 2]
    plot_prediction(
        model=name,
        cadastre_index=median.cadastre,
        tile_index=median.tile,
        save=save,
    )


def plot_worst_prediction(
    name: str,
    save: bool = False,
    area_filter: float = 0,
    offset: int = 0,
):
    """Plot worst prediction for given model."""
    # Worst without filter due to small object
    df = Trainer.evaluation_statistics(name=name)
    df = df[df.split == "test"]
    df = df[df["mask"] > area_filter * (256 ** 2)]
    df = df.sort_values(by="iou")
    worst = df.iloc[offset]
    plot_prediction(
        model=name,
        cadastre_index=worst.cadastre,
        tile_index=worst.tile,
        save=save,
    )


def plot_prediction_comparison(
    worst_model: str = "only_rgb",
    best_model: str = "without_rgb",
    area_filter: float = 0,
    offset: int = 0,
    save: bool = False,
    names: Tuple[str, str] = ("RGB", "LiDAR"),
):
    dfx = Trainer.evaluation_statistics(name=best_model)
    dfy = Trainer.evaluation_statistics(name=worst_model)
    df = dfx.merge(dfy, on=["cadastre", "tile", "split", "mask"])
    df = df[df.split == "test"]
    df["iou_improvement"] = df.iou_x - df.iou_y
    df = df.sort_values(by="iou_improvement")
    df = df[df["mask"] > area_filter * 256 ** 2]

    best_improvement = df.iloc[-(offset + 1)]
    cadastre = best_improvement.cadastre
    tile = best_improvement.tile
    worst_fig, worst_axes = plot_prediction(
        cadastre_index=cadastre,
        tile_index=tile,
        save=False,
        model=worst_model,
    )
    worst_fig.tight_layout()
    worst_metric_ax = worst_axes.flatten()[-1]
    worst_metric_ax.get_legend().remove()
    worst_metric_ax.set_title("")
    worst_prediction_ax = worst_axes.flatten()[-2]
    worst_prediction_ax.set_title("")

    best_fig, best_axes = plot_prediction(
        cadastre_index=cadastre,
        tile_index=tile,
        save=False,
        model=best_model,
    )
    best_fig.tight_layout()
    best_metric_ax = best_axes.flatten()[-1]
    best_metric_ax.set_title("")
    best_prediction_ax = best_axes.flatten()[-2]
    best_prediction_ax.set_title("")

    best_fig.text(
        0.66,
        0.658,
        r"\textbf{" + f"{names[1]} model" + "}",
        ha="center",
        fontsize=20,
    )
    worst_fig.text(
        0.66,
        0.658,
        r"\textbf{" + f"{names[0]} model" + "}",
        ha="center",
        fontsize=20,
    )

    if save:
        save_dir = Path(
            "tex/img/prediction_improvement/"
            f"{worst_model}+{best_model}/"
            f"{cadastre}+{tile}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        best_path = save_dir / "best.pdf"
        worst_path = save_dir / "worst.pdf"

        best_fig.savefig(best_path, bbox_inches="tight", pad_inches=0)
        worst_fig.savefig(worst_path, bbox_inches="tight", pad_inches=0)


def plot_rgb_help(save: bool = False):
    configure_latex(scaler=2)
    lidar_df = Trainer.evaluation_statistics(name="without_rgb")
    rgb_df = Trainer.evaluation_statistics(name="only_rgb")
    both_df = Trainer.evaluation_statistics(name="with_rgb")
    indep_df = lidar_df.merge(
        rgb_df,
        on=["cadastre", "tile", "split"],
        suffixes=("_lidar", "_rgb"),
    )
    indep_df["rgb_improvement"] = indep_df.iou_rgb - indep_df.iou_lidar
    comparison = both_df.merge(indep_df, on=["cadastre", "tile", "split"])
    comparison["combo_improvement"] = comparison.iou - comparison.iou_lidar

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    ax.scatter(
        x=comparison.rgb_improvement,
        y=comparison.combo_improvement,
        alpha=0.2,
        edgecolor="",
        rasterized=True,
    )

    plt.axhline(0, color="black")
    plt.axvline(0, color="black")

    ax.set_xlabel(r"$\mathrm{IoU(RGB) - IoU(LiDAR)}$")
    ax.set_ylabel(r"$\mathrm{IoU(LiDAR + RGB) - IoU(LiDAR)}$")

    num_windows = 20
    windows = np.linspace(-1, 1, num_windows)
    window_width = 2 / num_windows
    x, means, medians, lower, upper = [], [], [], [], []
    for window in windows:
        index = np.logical_and(
            comparison.rgb_improvement >= window,
            comparison.rgb_improvement < window + window_width,
        )
        if not np.any(index):
            continue
        x.append(window + 0.5 * window_width)
        means.append(comparison.combo_improvement[index].mean())
        medians.append(comparison.combo_improvement[index].median())
        lower.append(comparison.combo_improvement[index].quantile(0.25))
        upper.append(comparison.combo_improvement[index].quantile(0.75))

    ax.step(
        x,
        medians,
        where="mid",
        color="xkcd:orange",
        label="Moving\ninterval\nmedian",
    )
    ax.legend(framealpha=1)

    mpl.rcParams['savefig.dpi'] = 300
    fig.tight_layout()
    if save:
        fig.savefig("tex/img/rgb-helps.pdf")
???
