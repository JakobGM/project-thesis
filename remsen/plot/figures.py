from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

from scipy import ndimage

from remsen import utils
from remsen.plot.utils import configure_latex


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
