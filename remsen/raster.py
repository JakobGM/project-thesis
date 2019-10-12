from pathlib import Path
from typing import Tuple

import numpy as np

import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.mask import mask as rasterio_mask

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


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


def crop_and_mask(
    crop: Polygon,
    mask: MultiPolygon,
    raster_path: Path,
) -> Tuple[MemoryFile, MemoryFile]:
    """
    Crop and mask a given raster path.

    Supports single band LiDAR rasters and RGBZ rasters.
    """
    # Reduce mask size to crop area, all other masks are superflous
    mask = crop.intersection(mask)

    if isinstance(mask, GeometryCollection):
        mask = MultiPolygon([
            feature
            for feature
            in mask
            if not isinstance(feature, Point)
        ])

    with rasterio.open(raster_path) as src:
        assert str(src.crs["proj"]) == "utm" and int(src.crs["zone"]) == 32
        bands = src.count
        if bands == 1:
            # Raster file only contains one band; interpreting as LiDAR data
            cropped_lidar_data, affine_transformation = rasterio_mask(
                dataset=src,
                shapes=[crop],
                all_touched=True,
                crop=True,
                filled=False,
            )
        elif bands == 4:
            # Raster file contains four bands; interpreting as ZRGB data
            raster_bands = {1, 2, 3, 4}
            lidar_band = 1 + lidar_band_index(
                raster_path=raster_path,
            )
            rgb_bands = sorted(list(raster_bands - {lidar_band}))

            cropped_lidar_data, affine_transformation = rasterio_mask(
                dataset=src,
                shapes=[crop],
                all_touched=True,
                crop=True,
                filled=False,
                indexes=[lidar_band],
            )

            cropped_aerial_data, affine_aerial_transformation = rasterio_mask(
                dataset=src,
                shapes=[crop],
                all_touched=True,
                crop=True,
                filled=False,
                indexes=rgb_bands,
            )

            # Aerial data should have same shape as LiDAR data
            assert cropped_aerial_data.shape == (
                3,
                *cropped_lidar_data.shape[1:],
            )

            # RGB data should be in domain {0, 1, ..., 255}
            assert cropped_aerial_data.dtype == "uint8"

            # Transformations should be identical
            assert (
                affine_transformation == affine_aerial_transformation
            )
        else:
            raise NotImplementedError(f"Unsupported number of bands = {bands}")

        metadata = src.meta.copy()
        metadata.update(
            {
                "count": 1,
                "height": cropped_lidar_data.shape[1],
                "width": cropped_lidar_data.shape[2],
                "transform": affine_transformation,
                "driver": "GTiff",
                "dtype": cropped_lidar_data.dtype,
            }
        )

        cropped_lidar_file = MemoryFile()
        with rasterio.open(cropped_lidar_file, "w", **metadata) as file:
            file.write(cropped_lidar_data)
            if mask:
                mask_data, _ = rasterio_mask(
                    file,
                    shapes=[mask],
                    all_touched=True,
                    crop=False,
                )
                mask_data[mask_data > file.nodata] = 1
                mask_data[mask_data != 1] = 0
                mask_data = mask_data.astype("uint8", copy=False)
            else:
                mask_data = np.zeros(
                    cropped_lidar_data.shape,
                    dtype="uint8",
                )

        metadata = metadata.copy()
        metadata.update({"dtype": "uint8", "nodata": int(2 ** 8 - 1)})

        mask_file = MemoryFile()
        with rasterio.open(mask_file, "w", **metadata) as file:
            file.write(mask_data)

        return cropped_lidar_file, mask_file
