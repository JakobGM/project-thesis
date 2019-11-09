import numpy as np

from scipy import ndimage

from skimage.util import view_as_blocks


def split_images(batch, divisor):
    """Split images in batch by divisor, yielding divisor^2 as many images."""
    batch_size, img_height, img_width, channels = batch.shape
    new_shape = (
        (divisor ** 2) * batch_size,
        img_height // 2,
        img_width // 2,
        channels,
    )
    view_shape = (1,) + new_shape[1:]
    blocks = view_as_blocks(batch, view_shape)
    splits = blocks.reshape(new_shape)
    return splits


def edge_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Return edge mask from filled mask.

    Source: https://stackoverflow.com/a/33745971
    """
    # Rank 2 structure with full connectivity
    structure = ndimage.generate_binary_structure(rank=2, connectivity=2)
    erode = ndimage.binary_erosion(input=mask, structure=structure)
    # XOR operation
    edges = mask ^ erode
    return edges.astype(np.bool)
