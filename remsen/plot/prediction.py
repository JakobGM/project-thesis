from typing import Optional, Union

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
