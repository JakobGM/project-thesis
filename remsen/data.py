"""Module responsible for fetching, pre-processing, and preparing data."""
from pathlib import Path
from typing import Dict

import fiona

import numpy as np

import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask

from shapely.geometry import MultiPolygon, Polygon, shape


def fiona_polygon(fiona_item: Dict) -> Polygon:
    """Convert fiona item to Shapely polygon."""
    geometry = shape(fiona_item["geometry"])
    if not geometry.is_valid:
        geometry = geometry.bufer(0.0)
    assert geometry.is_valid
    assert geometry.geom_type == "Polygon"
    return geometry


def fetch_cadastre(path: Path, index: int) -> Polygon:
    """Fetch cadastre from dataset."""
    with fiona.open(path, layer="Teig") as src:
        srid = int(src.crs["init"].split(":")[1])
        assert srid == 25832
        item = src[index + 1]
        return fiona_polygon(item)


def fetch_buildings(path: Path) -> MultiPolygon:
    """Fetch all buildings from dataset."""
    with fiona.open(path, layer="Bygning") as src:
        srid = int(src.crs["init"].split(":")[1])
        assert srid == 25832
        return MultiPolygon([fiona_polygon(item) for item in src])


def construct_observation(path, cadastre, buildings):
    """Construct observation for a given cadastre."""
    intersecting_buildings = cadastre.intersection(buildings)

    with rasterio.open(path) as src:
        assert str(src.crs["proj"]) == "utm" and int(src.crs["zone"]) == 32
        cropped_data, affine_transformation = mask(
            src, shapes=[cadastre], all_touched=True, crop=True
        )

        metadata = src.meta.copy()
        metadata.update(
            {
                "height": cropped_data.shape[1],
                "width": cropped_data.shape[2],
                "transform": affine_transformation,
                "driver": "GTiff",
            }
        )

        cropped_lidar_file = MemoryFile()
        with rasterio.open(cropped_lidar_file, "w", **metadata) as file:
            file.write(cropped_data)
            if intersecting_buildings:
                building_data, _ = mask(
                    file,
                    shapes=[intersecting_buildings],
                    all_touched=True,
                    crop=False,
                )
                building_data[building_data > src.nodata] = 1
                building_data[building_data != 1] = 0
                building_data = building_data.astype("uint8", copy=False)
            else:
                building_data = np.zeros(cropped_data.shape, dtype="uint8")

        metadata = metadata.copy()
        metadata.update({"dtype": "uint8", "nodata": int(2 ** 8 - 1)})

        building_file = MemoryFile()
        with rasterio.open(building_file, "w", **metadata) as file:
            file.write(building_data)

        return cropped_lidar_file, building_file
