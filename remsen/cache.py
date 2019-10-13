"""Module responsible for all kind of caches used by remsen."""
import hashlib
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import geopandas

from ipypb import track

import numpy as np

from shapely.geometry import MultiPolygon, Polygon

from remsen import raster, vector


DEFAULT_CACHE_DIR = Path(".cache").resolve()


def sha1_hash(file_path: Path) -> str:
    """Return SHA-1 hash of given file."""
    assert file_path.is_file()
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as file_handler:
        while True:
            data = file_handler.read(65536)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def _save_tile(kwargs):
    """
    Generate and save tiles to disk.

    The following keyword arguments should be provided in the kwargs dictionary:
    cadastre - Geometry for which to calculate the bounding box which the
        tiles must overlap.
    raster_path - Path to raster file containing Z and/or RGB data.
    mask - MultiPolygon used to specify which areas to mask.
    max_num_tiles - Skip saving to disk if tiles exceed this number.
    lidar_dir - Directory to save LiDAR tiles.
    rgb_dir - Directory to save RGB tiles.
    mask_dir - Directory to save mask tiles.
    """
    assert len(kwargs) == 7
    cadastre = kwargs["cadastre"]
    raster_path = kwargs["raster_path"]
    mask = kwargs["mask"]
    max_num_tiles = kwargs["max_num_tiles"]
    lidar_dir = kwargs["lidar_dir"]
    rgb_dir = kwargs["rgb_dir"]
    mask_dir = kwargs["mask_dir"]

    bounds = cadastre.bounds
    try:
        result = raster.tiles(
            bounds=bounds,
            raster_path=raster_path,
            mask=mask,
            max_num_tiles=max_num_tiles,
        )
    except RuntimeError as exc:
        if "> max_num_tiles=" in str(exc):
            return
        else:
            raise exc

    if "lidar_tiles" in result:
        lidar_dir.mkdir(parents=True, exist_ok=True)
        lidar_tiles = result["lidar_tiles"]
        for tile_index, lidar_tile in enumerate(lidar_tiles):
            np.save(lidar_dir / f"{tile_index}.npy", lidar_tile)

    if "rgb_tiles" in result:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        rgb_tiles = result["rgb_tiles"]
        for tile_index, rgb_tile in enumerate(rgb_tiles):
            np.save(rgb_dir / f"{tile_index}.npy", rgb_tile)

    if "mask_tiles" in result:
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_tiles = result["mask_tiles"]
        for tile_index, mask_tile in enumerate(mask_tiles):
            np.save(mask_dir / f"{tile_index}.npy", mask_tile)

    return result["tile_dimensions"]


class CadastreCache:
    """Cache for given cadastre set."""

    def __init__(
        self,
        cadastre_path: Path,
        name: str,
        layer_name: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Construct cache for given cadastre set."""
        self.sha1_hash = sha1_hash(cadastre_path)
        self.name = name
        parent_dir = cache_dir or DEFAULT_CACHE_DIR
        self.directory = parent_dir / "cadastre" / self.name
        self.layer_name = layer_name
        self.cadastre_path = cadastre_path
        if self.directory.exists():
            import shutil
            shutil.rmtree(str(self.directory))
        if not self.directory.exists():
            self.first_time_setup()

    @classmethod
    def from_name(cls, name: str) -> "CadastreCache":
        pass

    def first_time_setup(self):
        self.directory.mkdir(parents=True)
        temp_directory = tempfile.TemporaryDirectory()
        command = (
            "ogr2ogr "
            '-f "ESRI Shapefile" '
            f"{temp_directory.name}/out.shp "
            f"{self.cadastre_path}"
        )
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        assert result.returncode == 0
        self.dataframe = geopandas.read_file(
            temp_directory.name + "/out.shp",
            layer="Teig",
        )

    def cadastre(self, index: int) -> Polygon:
        """Fetch cadastre from dataset."""
        return vector.get_polygon(
            path=self.cadastre_path,
            layer=self.layer_name,
            index=index,
        )

    def build_tile_cache(
        self,
        raster_path: Path,
        mask: MultiPolygon,
        mask_name: str,
        lidar_name: Optional[str] = None,
        rgb_name: Optional[str] = None,
        max_num_tiles: Optional[int] = 100,
    ) -> None:
        """
        Build tile cache for given set of cadastre in conjunction with mask and raster.

        Arrays are saved to the following semantic file paths:
        - {cache_path}/{name}/{lidar,rgb,mask}/{cadastre_index}/{tile_index}.npy

        :param raster_path: Path to raster file.
        :param mask: Multipolygon specifying which areas to mask.
        :param mask_name: Canonical name identifying the mask being used.
        :param lidar_name: Canonical name identifying the LiDAR data being used.
        :param rgb_name: Canonical name identifying the RGB data being used.
        :param max_num_tiles: Skip saving tiles if tiles exceed this number.
        """
        bands = raster.bands(raster_path=raster_path)
        if bands == 1:
            assert lidar_name
        elif bands == 3:
            assert rgb_name
        elif bands == 4:
            assert lidar_name and rgb_name
        else:
            raise NotImplementedError

        lidar_super_dir = self.directory / "lidar" / (lidar_name or "")
        rgb_super_dir = self.directory / "rgb" / (rgb_name or "")
        mask_super_dir = self.directory / "mask" / mask_name

        def _kwargs():
            for cadastre_index in range(len(self)):
                cadastre = self.cadastre(index=cadastre_index)
                lidar_dir = lidar_super_dir / str(cadastre_index)
                rgb_dir = rgb_super_dir / str(cadastre_index)
                mask_dir = mask_super_dir / str(cadastre_index)
                yield {
                    "cadastre": cadastre,
                    "raster_path": raster_path,
                    "mask": mask,
                    "max_num_tiles": max_num_tiles,
                    "lidar_dir": lidar_dir,
                    "rgb_dir": rgb_dir,
                    "mask_dir": mask_dir,
                }

        pool = Pool(processes=None)
        pool_tasks = pool.imap(func=_save_tile, iterable=_kwargs(), chunksize=1)
        for _ in track(pool_tasks, total=len(self)):
            pass

    def __len__(self) -> int:
        """Return number of cadastre in source data."""
        return vector.layer_size(
            path=self.cadastre_path,
            layer=self.layer_name,
        )

    def __repr__(self) -> str:
        """Return pertinent information regarding cadastre cache."""
        return (
            "CadastreCache("
            f'name="{self.name}", '
            f"length={len(self)}, "
            f'path="{self.directory}"'
            ")"
        )
