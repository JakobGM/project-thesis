"""Tests for remsen.data module."""
from pathlib import Path

import pytest

from remsen.data import construct_observation, fetch_buildings, fetch_cadastre


@pytest.fixture(scope="session")
def data_directory():
    """Directory containing datasets."""
    return Path(__file__).parents[2] / "data"


@pytest.fixture(scope="session")
def cadastre(data_directory):
    cadastre_path = data_directory / "cadastre.gpkg"
    cadastre = fetch_cadastre(path=cadastre_path, index=0)
    return cadastre


@pytest.fixture(scope="session")
def buildings(data_directory):
    buildings_path = data_directory / "building.gpkg"
    buildings = fetch_buildings(path=buildings_path)
    return buildings


def test_fetch_cadastre(cadastre):
    """You should be able to retrieve cadastre plot shapely polygons."""
    assert int(cadastre.area) == 383


def test_fetch_buildings(buildings):
    """You should be able to retrieve all buldings as a MultiPolygon."""
    assert len(buildings) == 70_082


def test_construct_observation(data_directory, cadastre, buildings):
    """You should be able to construct numpy arrays from data."""
    lidar_path = data_directory / "lidar.vrt"
    cropped_lidar_file, building_data_file = construct_observation(
        path=lidar_path, cadastre=cadastre, buildings=buildings
    )
    with cropped_lidar_file.open() as lidar_file:
        assert lidar_file.nodata == pytest.approx(float(-2 ** 128))
        lidar_data = lidar_file.read(1)
        assert lidar_data.dtype == "float32"

    with building_data_file.open() as building_file:
        assert building_file.nodata == 255.0
        building_data = building_file.read(1)
        assert building_data.dtype == "uint8"
