from pathlib import Path

import rasterio
from rasterio.enums import ColorInterp


def lidar_nodata_value(path: Path) -> float:
    """
    Return nodata value of lidar band in raster file.

    Supports files that contain RGB bands as well.
    """
    with rasterio.open(path, "r") as lidar_file:
        nodata_values = lidar_file.nodatavals
        if len(nodata_values) == 1:
            return nodata_values[0]

        color_interpretations = lidar_file.colorinterp
        assert set(color_interpretations) == {
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.undefined,
        }
        lidar_index = color_interpretations.index(ColorInterp.undefined)
        return nodata_values[lidar_index]
