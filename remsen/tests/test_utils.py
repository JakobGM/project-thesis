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


def test_edge_pixels():
    """Test simple rectangle without interior hulls."""
    image = np.zeros([20, 20], dtype=np.bool)
    image[0:10, 0:10] = 1
    edges = utils.edge_pixels(image)

    # The number of edge pixels should be 36
    assert edges.flatten().sum() == 36

    # Left edge, right edge
    assert edges[0:10, 0].all()
    assert edges[0:10, 9].all()

    # Top edge, bottom edge
    assert edges[0, 0:10].all()
    assert edges[9, 0:10].all()


def test_edge_pixels_with_interior_hulls():
    """Test complex rectangle with interior hulls."""
    image = np.zeros([20, 20], dtype=np.bool)
    image[0:10, 0:10] = 1
    image[5, 5] = 0
    edges = utils.edge_pixels(image)

    # The number of edge pixels should be 36
    assert edges.flatten().sum() == 44

    # Left edge, right edge
    assert edges[0:10, 0].all()
    assert edges[0:10, 9].all()

    # Top edge, bottom edge
    assert edges[0, 0:10].all()
    assert edges[9, 0:10].all()

    # Interior hull
    assert edges[4, 4:7].all()
    assert edges[6, 4:7].all()
    assert edges[4:7, 4].all()
    assert edges[4:7, 6].all()


def test_edge_pixels_of_single_line():
    """Test edge creation of single line."""
    image = np.zeros([20, 20], dtype=np.bool)
    image[5, 0:10] = 1
    edges = utils.edge_pixels(image)
    assert (edges == image).all()
