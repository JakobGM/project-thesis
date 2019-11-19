from matplotlib import pyplot as plt

import numpy as np

from remsen import plot


def test_configure_latex(tmp_path):
    """Test if configure_latex() allows saving PDF figure."""
    plot.configure_latex()
    x = np.arange(0, 100)
    y = x.copy()
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel("$x$")
    fig_path = tmp_path.with_suffix(".pdf")
    fig.savefig(fig_path)
    assert fig_path.exists()


def test_importable_names():
    """Test that all the correct functions are importable."""
    functions = [
        "configure_latex",
        "plot_training",
        "imshow_with_mask",
        "plot_bbox_distribution",
        "plot_gdalinfo_histogram",
        "plot_raster_spread",
        "plot_segmentation_types",
    ]
    for function in functions:
        assert function in dir(plot)
