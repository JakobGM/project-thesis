"""Tests for remsen.data module."""
from pathlib import Path

import pytest

from remsen.data import fetch_buildings, fetch_cadastre


@pytest.fixture
def data_directory():
    """Directory containing datasets."""
    return Path(__file__).parents[2] / "data"


def test_fetch_cadastre(data_directory):
    """You should be able to retrieve cadastre plot shapely polygons."""
    cadastre_path = data_directory / "cadastre.gpkg"
    assert cadastre_path.exists()

    cadastre = fetch_cadastre(path=cadastre_path, index=0)
    assert int(cadastre.area) == 383


def test_fetch_buildings(data_directory):
    """You should be able to retrieve all buldings as a MultiPolygon."""
    buildings_path = data_directory / "building.gpkg"
    assert buildings_path.exists()

    buildings = fetch_buildings(path=buildings_path)
    assert len(buildings) == 70_082
