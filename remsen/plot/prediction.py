from typing import Optional, Union

from matplotlib import colors, patheffects, pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from tensorflow.keras.models import Model

from remsen.data import Dataset
from remsen.plot.data import imshow_with_mask
from remsen.training import Trainer


def plot_prediction(
    cadastre_index=0,
    tile_index=0,
    dataset: Optional[Dataset] = None,
    model: Union[str, Model] = "without_rgb",
):
    if not dataset:
        dataset = Dataset()
    if isinstance(model, str):
        trainer = Trainer(
            name=model,
            model=None,
            dataset=dataset,
        )
        model = trainer.model

    # Prepare data for prediction
    with_rgb = model.input.shape[-1] == 4
    tiles = dataset.tiles(
        cadastre_index=cadastre_index,
        with_tile_dimensions=False,
        with_rgb=with_rgb,
    )
    lidar_tile = tiles["lidar"][tile_index]
    building_tile = tiles["mask"][tile_index]
    input_tile = dataset.input_tile_normalizer(lidar_tile.copy())[0]

    if with_rgb:
        rgb_tile = tiles["rgb"][tile_index]
        input_tile = np.concatenate(
            [input_tile, rgb_tile / 255],
            axis=2,
        )
    input_tile = np.expand_dims(input_tile, 0)

    # (2, 2) plot if with RGB, otherwise (1, 3)
    fig, axes = plt.subplots(
        nrows=2 if with_rgb else 1,
        ncols=2 if with_rgb else 3,
        figsize=(15, 15),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    if with_rgb:
        lidar_ax, rgb_ax, prediction_ax, metric_ax = axes.flatten()
        rgb_ax.title.set_text("RGB data")
        imshow_with_mask(image=rgb_tile, mask=building_tile, ax=rgb_ax)
    else:
        lidar_ax, prediction_ax, metric_ax = axes.flatten()

    lidar_ax.title.set_text("LiDAR data")
    prediction_ax.title.set_text("Prediction probabilities")
    metric_ax.title.set_text("TP / TN / FP / FN, cut-off = 0.5")

    cadastre_text = axes[0][0].annotate(
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

    for axis in axes.flatten():
        axis.tick_params(labelbottom=False, labelleft=False, width=0.0)

    lidar_tile = np.squeeze(lidar_tile)
    imshow_with_mask(image=lidar_tile, mask=building_tile, ax=lidar_ax)

    lidar_tile = np.expand_dims(lidar_tile, 0)
    lidar_tile = np.expand_dims(lidar_tile, -1)

    predicted_building_tile = model.predict(input_tile)
    predicted_building_tile = np.squeeze(predicted_building_tile)
    prediction_ax.imshow(
        predicted_building_tile,
        cmap="seismic",
        vmin=0,
        vmax=1,
    )
    imshow_with_mask(
        image=predicted_building_tile,
        mask=building_tile,
        cmap="seismic",
        ax=prediction_ax,
        edge_color=(0, 255, 0, 255),
    )

    predicted_mask = (predicted_building_tile > 0.5).astype("uint8")
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
    metric_ax.imshow(confusion_matrix, cmap=cmap, norm=norm)

    # Add TP/TN/FP/FN legend to plot
    legend_elements = [
        Patch(facecolor='#001F3F', edgecolor="white", label='TP'),
        Patch(facecolor='#DDDDDD', edgecolor="white", label='TN'),
        Patch(facecolor='#2ECC40', edgecolor="white", label='FP'),
        Patch(facecolor='#FF4136', edgecolor="white", label='FN'),
    ]
    metric_ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.075),
        frameon=False,
        handlelength=1.3,
        handleheight=1.5,
    )

    # Add metrics to plot
    building_tile = np.expand_dims(building_tile, 0)
    building_tile = np.expand_dims(building_tile, -1)
    evaluation = model.evaluate(
        x=input_tile,
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
    prediction_ax.set_xlabel(
        f"Loss = {loss:.4f},   IoU = {iou:0.4f}",
        size=13,
    )

    plt.tight_layout()
    plt.show()
