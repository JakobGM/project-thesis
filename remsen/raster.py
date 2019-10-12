from pathlib import Path

import rasterio
from rasterio.enums import ColorInterp


def lidar_nodata_value(raster_path: Path) -> float:
    """
    Return nodata value of lidar band in raster file.

    Supports files that contain RGB bands as well.
    """
    with rasterio.open(raster_path, "r") as lidar_file:
        lidar_index = lidar_band_index(raster_path=raster_path)
        nodata_values = lidar_file.nodatavals
        return nodata_values[lidar_index]


def lidar_band_index(raster_path: Path) -> int:
    """
    Return lidar band index of given raster file.

    NB! The lidar band index is zero-indexed, while rasterio operates
    with one-indexed bands, requiring you to add one to the index
    if used in conjunction with the rasterio API.
    """
    with rasterio.open(raster_path, "r") as lidar_file:
        color_interpretations = lidar_file.colorinterp
        if len(color_interpretations) == 1:
            return 0

        assert set(color_interpretations) == {
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.undefined,
        }
        lidar_index = color_interpretations.index(ColorInterp.undefined)
        return lidar_index
