from pathlib import Path
from typing import Dict, Union

import fiona

from shapely.geometry import MultiPolygon, Polygon, shape


def fiona_polygon(fiona_item: Dict) -> Polygon:
    """Convert fiona item to Shapely polygon."""
    geometry = shape(fiona_item["geometry"])
    if not geometry.is_valid:
        geometry = geometry.buffer(0.0)
    assert geometry.is_valid
    assert geometry.geom_type in ("Polygon", "MultiPolygon")
    return geometry


def get_polygon(
    path: Path,
    layer: str,
    index: int,
) -> Union[MultiPolygon, Polygon]:
    """Return indexed polygon from layer in OGR path."""
    with fiona.open(path, "r", layer=layer) as src:
        srid = int(src.crs["init"].split(":")[1])
        assert srid == 25832
        item = src[index + 1]
        return fiona_polygon(item)


def layer_size(path: Path, layer: str) -> int:
    """Return size of given layer in OGR path."""
    with fiona.open(path, "r", layer=layer) as src:
        return len(src)
