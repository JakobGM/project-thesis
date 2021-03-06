"""Module responsible for all kind of caches used by remsen."""
import hashlib
import json
import pickle
import shelve
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import fiona

import geopandas

from ipypb import track

import numpy as np

import pandas as pd

from shapely.geometry import MultiPolygon, Polygon, mapping

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
    assert len(kwargs) == 8
    cadastre_index = kwargs["cadastre_index"]
    cadastre = kwargs["cadastre"]
    raster_path = kwargs["raster_path"]
    mask = kwargs["mask"]
    max_num_tiles = kwargs["max_num_tiles"]
    lidar_dir = kwargs["lidar_dir"]
    rgb_dir = kwargs["rgb_dir"]
    mask_dir = kwargs["mask_dir"]

    bounds = cadastre.bounds
    if len(bounds) != 4:
        return cadastre_index, None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Cannot provide views")
            result = raster.tiles(
                bounds=bounds,
                raster_path=raster_path,
                mask=mask,
                max_num_tiles=max_num_tiles,
            )
    except RuntimeError as exc:
        if "> max_num_tiles=" in str(exc):
            return cadastre_index, None
        else:
            raise exc

    if "lidar_tiles" in result:
        lidar_dir.mkdir(parents=True, exist_ok=True)
        lidar_tiles = result["lidar_tiles"]
        for tile_index, lidar_tile in enumerate(lidar_tiles):
            save_to = lidar_dir / f"{tile_index:04d}.npy"
            if not save_to.exists():
                np.save(save_to, lidar_tile)

    if "rgb_tiles" in result:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        rgb_tiles = result["rgb_tiles"]
        for tile_index, rgb_tile in enumerate(rgb_tiles):
            save_to = rgb_dir / f"{tile_index:04d}.npy"
            if not save_to.exists():
                np.save(save_to, rgb_tile)

    if "mask_tiles" in result:
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_tiles = result["mask_tiles"]
        for tile_index, mask_tile in enumerate(mask_tiles):
            save_to = mask_dir / f"{tile_index:04d}.npy"
            if not save_to.exists():
                np.save(save_to, mask_tile)

    metadata = {
        "tile_height": result["tile_dimensions"][0],
        "tile_width": result["tile_dimensions"][1],
        "number_of_tiles": result["number_of_tiles"],
        "area": mapping(result["shapely_crop"]),
    }
    return cadastre_index, metadata


class Cache:
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
        self.parent_dir = cache_dir or DEFAULT_CACHE_DIR
        self.directory = self.parent_dir / "cadastre" / self.name
        self.metadata_path = self.directory / "metadata.json"
        self.layer_name = layer_name
        self.cadastre_path = cadastre_path
        if not self.directory.exists():
            self.first_time_setup()

    @classmethod
    def from_name(
        cls,
        name: str,
        cache_dir: Optional[Path] = None,
    ) -> "Cache":
        """Retrieve existing cache from name only."""
        cache_dir = cache_dir or DEFAULT_CACHE_DIR
        directory = cache_dir / "cadastre" / name
        if not directory.exists():
            valid_names = ", ".join(
                [repr(path.name) for path in directory.parent.glob("*/")],
            )
            raise ValueError(
                f"Cache with name {repr(name)} does not exist. "
                f"Name must be one of: {valid_names}."
            )
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        layer_name = metadata["layer_name"]
        cadastre_path = Path(metadata["cadastre_path"])
        return cls(
            cadastre_path=cadastre_path,
            name=name,
            layer_name=layer_name,
            cache_dir=cache_dir,
        )

    @property
    def dataframe(self) -> geopandas.GeoDataFrame:
        if hasattr(self, "_dataframe"):
            return self._dataframe

        self._dataframe = geopandas.read_pickle(
            self.directory / "cadastre.pkl",
        )
        return self._dataframe

    def first_time_setup(self):
        """Initialize the cadastre cache for the first time."""
        self.directory.mkdir(parents=True, exist_ok=True)

        # Save pertinent cadastre cache metadata
        metadata = {
            "cadastre_path": str(self.cadastre_path.resolve().absolute()),
            "layer_name": self.layer_name,
        }
        self.metadata_path.write_text(json.dumps(metadata))
        metadata = json.loads(self.metadata_path.read_text())

        geometries = []
        indeces = []
        with fiona.open(
            metadata["cadastre_path"],
            layer=metadata["layer_name"],
            mode="r",
        ) as src:
            # srid = int(src.crs["init"].split(":")[0])
            crs = src.crs["init"]
            for index, obj in track(enumerate(src), total=len(src)):
                geometries.append(vector.fiona_polygon(obj))
                indeces.append(index + 1)

        self._dataframe = geopandas.GeoDataFrame(
            crs=crs,
            geometry=geopandas.GeoSeries(geometries),
        )
        self._dataframe.set_index(pd.Index(indeces), inplace=True)
        self._dataframe.to_pickle(
            self.directory / "cadastre.pkl",
            protocol=-1,
        )

    def change_dataset(
        self,
        lidar_name: str = "trondheim_lidar_2017",
        rgb_name: str = "trondheim_rgb_2017",
        mask_name: str = "trondheim_buildings_2019",
    ) -> None:
        """Change which datasets that are used for other method calls."""
        self.lidar_name = lidar_name
        self.rgb_name = rgb_name
        self.mask_name = mask_name

        self.mask_dir = self.directory / "mask" / mask_name
        assert self.mask_dir.exists()
        self.rgb_dir = self.directory / "rgb" / rgb_name
        assert self.rgb_dir.exists()
        self.lidar_dir = self.directory / "lidar" / lidar_name
        assert self.lidar_dir.exists()
        self.lidar_metadata = json.loads(
            (self.lidar_dir / "metadata.json").read_text(),
        )

        lidar_indeces = {p.name for p in self.lidar_dir.glob("*/")}
        assert lidar_indeces
        rgb_indeces = {p.name for p in self.rgb_dir.glob("*/")}
        assert rgb_indeces
        mask_indeces = {p.name for p in self.mask_dir.glob("*/")}
        assert mask_indeces
        common_cadastre_indeces = lidar_indeces.intersection(
            rgb_indeces,
            mask_indeces,
        )
        self.cadastre_indeces = sorted(
            (int(index) for index in common_cadastre_indeces)
        )

    def number_of_tiles(
        self,
        cadastre_indeces: Iterable[Union[int, str]],
    ) -> int:
        """Return number of tiles for the given cadastre indeces."""
        num_tiles = 0
        for cadastre_index in cadastre_indeces:
            num_tiles += len(list(
                (self.lidar_dir / str(cadastre_index)).glob("*.npy"),
            ))
        return num_tiles

    def cadastre(self, index: int) -> Polygon:
        """Fetch cadastre from dataset."""
        return vector.get_polygon(
            path=self.cadastre_path,
            layer=self.layer_name,
            index=index,
        )

    def cache_mask(
        self,
        ogr_path: Path,
        layer_name: str,
        mask_name: str,
    ) -> None:
        """Cache given mask to disk."""
        mask_directory = self.directory / "mask" / mask_name
        mask_pickle = mask_directory / "mask.pkl"
        if mask_pickle.exists():
            print("Already cached!")
            return
        else:
            mask_directory.mkdir(parents=True, exist_ok=True)

        with fiona.open(ogr_path, "r", layer=layer_name) as src:
            mask = MultiPolygon(
                [vector.fiona_polygon(feature) for feature in track(src)]
            )
            print("Buffering multipolygon pickle... ", end="")
            mask = mask.buffer(0.0)
            print("Done!")

        print("Writing pickle file... ", end="")
        mask_pickle.write_bytes(
            pickle.dumps(mask, protocol=pickle.HIGHEST_PROTOCOL),
        )
        print("Done!")

    def mask_geometry(self, mask_name: Optional[str] = None) -> MultiPolygon:
        mask_name = mask_name or self.mask_name
        mask_directory = self.directory / "mask" / mask_name
        mask_pickle = mask_directory / "mask.pkl"
        if not mask_pickle.exists():
            raise ValueError(
                f"Mask with name {mask_name} does not exist. "
                "Generate it by using the .cache_mask() method first."
            )
        return pickle.loads(mask_pickle.read_bytes())

    def lidar_tiles(self, cadastre_index: int):
        lidar_tile_directory = self.lidar_dir / str(cadastre_index)
        assert lidar_tile_directory.exists()
        lidar_tile_paths = sorted(lidar_tile_directory.glob("*.npy"))
        lidar_tiles = []
        for lidar_tile_path in lidar_tile_paths:
            lidar_tiles.append(np.load(lidar_tile_path))
        return lidar_tiles

    def mask_tiles(self, cadastre_index: int):
        mask_tile_directory = self.mask_dir / str(cadastre_index)
        assert mask_tile_directory.exists()
        mask_tile_paths = sorted(mask_tile_directory.glob("*.npy"))
        mask_tiles = []
        for mask_tile_path in mask_tile_paths:
            mask_tiles.append(np.load(mask_tile_path))
        return mask_tiles

    def rgb_tiles(self, cadastre_index: int):
        rgb_tile_directory = self.rgb_dir / str(cadastre_index)
        assert rgb_tile_directory.exists()
        rgb_tile_paths = sorted(rgb_tile_directory.glob("*.npy"))
        rgb_tiles = []
        for rgb_tile_path in rgb_tile_paths:
            rgb_tiles.append(np.load(rgb_tile_path))
        return rgb_tiles

    def tile_dimensions(self, cadastre_index: int):
        tile_metadata_file = self.directory / "tile_metadata.json"
        tile_metadata = json.loads(tile_metadata_file.read_text())
        metadata = tile_metadata[str(cadastre_index)]
        return metadata["tile_height"], metadata["tile_width"]

    def build_tile_cache(
        self,
        raster_path: Path,
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
        mask = self.mask_geometry(mask_name=mask_name)
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

        lidar_metadata = {
            "nodata_value": raster.lidar_nodata_value(raster_path=raster_path),
        }
        lidar_metadata_path = lidar_super_dir / "metadata.json"
        lidar_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        lidar_metadata_path.touch()
        lidar_metadata_path.write_text(json.dumps(lidar_metadata))

        def _kwargs():
            for cadastre_index in range(len(self)):
                cadastre = self.cadastre(index=cadastre_index)
                lidar_dir = lidar_super_dir / str(cadastre_index)
                rgb_dir = rgb_super_dir / str(cadastre_index)
                mask_dir = mask_super_dir / str(cadastre_index)
                if lidar_dir.exists() and rgb_dir.exists() and mask_dir.exists():
                    continue

                yield {
                    "cadastre_index": cadastre_index,
                    "cadastre": cadastre,
                    "raster_path": raster_path,
                    "mask": mask,
                    "max_num_tiles": max_num_tiles,
                    "lidar_dir": lidar_dir,
                    "rgb_dir": rgb_dir,
                    "mask_dir": mask_dir,
                }

        metadata_file = self.directory / "tile_metadata.json"
        if metadata_file.exists():
            tile_metadata = json.loads(metadata_file.read_text())
        else:
            tile_metadata = {}

        pool = Pool(processes=None)
        pool_tasks = pool.imap_unordered(
            func=_save_tile,
            iterable=_kwargs(),
            chunksize=1,
        )
        total_cadastre = len(self)
        try:
            for cadastre_index, metadata in pool_tasks:
                print(f"{cadastre_index:06d} / {total_cadastre}", end="\r")
                if metadata:
                    tile_metadata[cadastre_index] = metadata
        finally:
            metadata_file.write_text(json.dumps(tile_metadata))

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


class Store:
    """Simple key-value store cache persisted to disk."""

    def __init__(self, path: Union[str, Path] = "/code/.cache/store.pkl"):
        self.path = Path(path)
        self.shelve = shelve.DbfilenameShelf(
            filename=str(self.path),
            flag="c",
            protocol=pickle.HIGHEST_PROTOCOL,
            writeback=True,
        )

    def insert(self, *keys, value: Any) -> None:
        assert all(isinstance(key, str) for key in keys)
        first_key, *middle_keys, final_key = keys
        container = self.shelve.setdefault(first_key, {})
        for key in middle_keys:
            container = container.setdefault(key, {})
        container[final_key] = value
        self.shelve.sync()

    def __getitem__(self, key: str) -> Dict:
        return self.shelve[key]
