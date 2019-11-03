import numpy as np

from remsen import utils


def test_split_images():
    """Test if split reshaping occurs correctly."""
    images = np.random.normal(size=(2, 4, 4, 3))
    split_images = utils.split_images(batch=images, divisor=2)

    # We have 4 times as many images of half height and width
    assert split_images.shape == (8, 2, 2, 3)

    # First quadrant of first image
    np.testing.assert_array_equal(
        split_images[0],
        images[0, 0:2, 0:2, :],
    )

    # Second quadrant of first image
    np.testing.assert_array_equal(
        split_images[1],
        images[0, 0:2, 2:4, :],
    )

    # Fourth quadrant of first image
    np.testing.assert_array_equal(
        split_images[3],
        images[0, 2:4, 2:4, :],
    )

    # First quadrant of second image
    np.testing.assert_array_equal(
        split_images[4],
        images[1, 0:2, 0:2, :],
    )
