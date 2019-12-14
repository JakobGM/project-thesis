from matplotlib import cm, colors

import numpy as np

from remsen import utils


def imshow_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    ax,
    cmap: str = None,
    edge_color: str = "xkcd:black",
    alpha: int = 180,
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
    edge_color = (255 * np.array(colors.to_rgba(edge_color))).astype(int)
    edge_color[-1] = alpha
    edges = utils.edge_pixels(np.squeeze(mask))
    image[:, :, :][edges] = edge_color
    ax.imshow(image)
    return ax
