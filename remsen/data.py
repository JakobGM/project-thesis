"""Module responsible for fetching, pre-processing, and preparing data."""
from pathlib import Path
from typing import Dict

import fiona

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
