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
