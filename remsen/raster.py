from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.mask import mask as rasterio_mask

from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Point,
    Polygon,
)

from skimage.util import view_as_blocks


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


def bands(raster_path: Path) -> int:
    """Return number of bands in given raster file."""
    with rasterio.open(raster_path, "r") as raster_file:
        return raster_file.count


def crop_and_mask(
    crop: Polygon,
    mask: MultiPolygon,
    raster_path: Path,
) -> Dict:
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

    result = {"shapely_mask": mask, "shapely_crop": crop}

    with rasterio.open(raster_path) as src:
        assert str(src.crs["proj"]) == "utm" and int(src.crs["zone"]) == 32
        original_metadata = src.meta.copy()

        bands = src.count
        if bands == 1:
            # Raster file only contains one band; interpreting as LiDAR data
            with_rgb = False
            cropped_lidar_data, affine_transformation = rasterio_mask(
                dataset=src,
                shapes=[crop],
                all_touched=True,
                crop=True,
                filled=False,
            )
            result["lidar_array"] = cropped_lidar_data
        elif bands == 4:
            # Raster file contains four bands; interpreting as ZRGB data
            with_rgb = True
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
            result["lidar_array"] = cropped_lidar_data
            result["rgb_array"] = cropped_aerial_data
        else:
            raise NotImplementedError(f"Unsupported number of bands = {bands}")

    lidar_metadata = original_metadata.copy()
    lidar_metadata.update(
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
    with rasterio.open(cropped_lidar_file, "w", **lidar_metadata) as file:
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

    mask_metadata = lidar_metadata.copy()
    mask_metadata.update({"dtype": "uint8", "nodata": int(2 ** 8 - 1)})
    mask_file = MemoryFile()
    with rasterio.open(mask_file, "w", **mask_metadata) as file:
        file.write(mask_data)

    if with_rgb:
        rgb_metadata = lidar_metadata.copy()
        rgb_metadata.update({
            "dtype": cropped_aerial_data.dtype,
            "count": 3,
        })
        rgb_file = MemoryFile()
        with rasterio.open(rgb_file, "w", **rgb_metadata) as file:
            file.write(cropped_aerial_data)
        result["rgb_file"] = rgb_file

    result["lidar_file"] = cropped_lidar_file
    result["mask_file"] = mask_file
    result["mask_array"] = mask_data
    return result


def tiles(
    bounds: Tuple[float, float, float, float],
    raster_path: Path,
    mask: MultiPolygon,
    max_num_tiles: Optional[int] = None,
) -> Dict:
    """
    Return 256 x 256 tiles covering the given bounds.

    Since we have pixels of size 0.25^2 meters, and we return tiles of size
    256 x 256, each tile represenents a 64m x 64m area.
    """
    min_x, min_y, max_x, max_y = bounds

    width = max_x - min_x
    height = max_y - min_y
    assert width > 0 and height > 0

    mid_x = min_x + 0.5 * width
    mid_y = min_y + 0.5 * height

    width_tiles = 1 if width <= 64 else width // 64 + 1
    height_tiles = 1 if height <= 64 else height // 64 + 1
    num_tiles = width_tiles * height_tiles
    if max_num_tiles and num_tiles > max_num_tiles:
        raise RuntimeError(
            f"Produced {num_tiles} > max_num_tiles={max_num_tiles}",
        )

    new_width = 64 * width_tiles
    new_height = 64 * height_tiles

    new_width -= 0.2
    new_height -= 0.2

    bounding_box = Polygon.from_bounds(
        xmin=mid_x - 0.5 * new_width,
        xmax=mid_x + 0.5 * new_width,
        ymin=mid_y - 0.5 * new_height,
        ymax=mid_y + 0.5 * new_height,
    )

    original_result = crop_and_mask(
        crop=bounding_box,
        mask=mask,
        raster_path=raster_path,
    )
    lidar_array = original_result["lidar_array"]
    mask_array = original_result["mask_array"]
    if "rgb_array" in original_result:
        with_rgb = True
        rgb_array = original_result["rgb_array"]
    else:
        # Discard rgb_array before returning, but this rgb_array placeholder
        # will reduce a lot of branching in the following code.
        with_rgb = False
        rgb_array = np.zeros((3, *lidar_array.shape[1:]), dtype="uint8")

    # Convert (CHANNELS, HEIGHT, WIDTH) -> (HEIGHT, WIDTH, CHANNELS),
    # which is the standard for everything besides rasterio.
    lidar_array = np.moveaxis(lidar_array, source=0, destination=2)
    mask_array = np.moveaxis(mask_array, source=0, destination=2)
    rgb_array = np.moveaxis(rgb_array, source=0, destination=2)

    # Trim last index if we have a small mismatch of the 256 multiplicity
    if lidar_array.shape[0] % 256 != 0:
        lidar_array = lidar_array[:-1, :, :]
        mask_array = mask_array[:-1, :, :]
        rgb_array = rgb_array[:-1, :, :]
    if lidar_array.shape[1] % 256 != 0:
        lidar_array = lidar_array[:, :-1, :]
        mask_array = mask_array[:, :-1, :]
        rgb_array = rgb_array[:, :-1, :]

    try:
        for array in (lidar_array, mask_array, rgb_array):
            assert array.shape[0] % 256 == 0
            assert array.shape[1] % 256 == 0
    except AssertionError:
        lidar_shape = lidar_array.shape
        mask_shape = mask_array.shape
        raise RuntimeError(
            f"LiDAR and/or mask could not be reshaped to a"
            "multiple of (256, 256). The resulting shape is: "
            f"lidar_shape={lidar_shape}, mask_shape={mask_shape}."
        )

    # Extract tiles from arrays
    lidar_tiles = view_as_blocks(lidar_array, (256, 256, 1))
    mask_tiles = view_as_blocks(mask_array, (256, 256, 1))
    rgb_tiles = view_as_blocks(rgb_array, (256, 256, 3))

    tile_dimensions = lidar_tiles.shape[:2]
    number_of_tiles = tile_dimensions[0] * tile_dimensions[1]

    # Convert to standard shape (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
    lidar_tiles = lidar_tiles.reshape(number_of_tiles, 256, 256, 1)
    mask_tiles = mask_tiles.reshape(number_of_tiles, 256, 256, 1)
    rgb_tiles = rgb_tiles.reshape(number_of_tiles, 256, 256, 3)

    result = {
        "lidar_tiles": lidar_tiles,
        "mask_tiles": mask_tiles,
        "tile_dimensions": tile_dimensions,
        "number_of_tiles": number_of_tiles,
    }
    if with_rgb:
        result["rgb_tiles"] = rgb_tiles

    original_result.update(result)
    return original_result
