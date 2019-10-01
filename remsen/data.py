"""Module responsible for fetching, pre-processing, and preparing data."""
import pickle
from multiprocessing import Process, SimpleQueue
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import fiona

from ipypb import irange

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask

from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)

from skimage.util import view_as_blocks

import tensorflow as tf


def fiona_polygon(fiona_item: Dict) -> Polygon:
    """Convert fiona item to Shapely polygon."""
    geometry = shape(fiona_item["geometry"])
    if not geometry.is_valid:
        geometry = geometry.buffer(0.0)
    assert geometry.is_valid
    assert geometry.geom_type == "Polygon"
    return geometry


class Dataset:
    """Dataset class for cadastre+building+LiDAR data."""

    def __init__(
        self,
        buildings_path: Path = Path("data/building.gpkg"),
        cadastre_path: Path = Path("data/cadastre.gpkg"),
        lidar_path: Path = Path("data/lidar.vrt"),
        cache_dir: Path = Path(".cache/"),
    ) -> None:
        """Censtruct dataset."""
        assert buildings_path.exists()
        self.buildings_path = buildings_path

        assert cadastre_path.exists()
        self.cadastre_path = cadastre_path

        assert lidar_path.exists()
        self.lidar_path = lidar_path

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = cache_dir / "cache_index.pkl"
        self.tile_cache_path = cache_dir / "tiles"
        self.tile_cache_path.mkdir(parents=True, exist_ok=True)

    def cadastre(self, index: int) -> Polygon:
        """Fetch cadastre from dataset."""
        with fiona.open(self.cadastre_path, layer="Teig") as src:
            srid = int(src.crs["init"].split(":")[1])
            assert srid == 25832
            item = src[index + 1]
            return fiona_polygon(item)

    def buildings(self) -> MultiPolygon:
        """Fetch all buildings from dataset."""
        buildings_cache = self.cache_dir / "fixed_buildings.pkl"
        if hasattr(self, "_buildings"):
            # In-memory cache
            return self._buildings
        elif buildings_cache.exists():
            # On-disk cache
            self._buildings = pickle.loads(buildings_cache.read_bytes())
            return self._buildings
        else:
            # Populate both caches with buffer-fixed buildings
            with fiona.open(self.buildings_path, "r", layer="Bygning") as src:
                srid = int(src.crs["init"].split(":")[1])
                assert srid == 25832
                buildings = MultiPolygon([fiona_polygon(item) for item in src])
                self._buildings = buildings.buffer(0.0)

            buildings_cache.write_bytes(
                pickle.dumps(self._buildings, protocol=pickle.HIGHEST_PROTOCOL),
            )
            return self._buildings

    def building(self, index: int) -> Polygon:
        """Fetch specific building from dataset."""
        return self.buildings()[index]

    def construct_observation(
        self,
        cadastre: Union[int, Polygon],
    ) -> Tuple[MemoryFile, MemoryFile]:
        """Construct observation for a given cadastre."""
        if isinstance(cadastre, int):
            cadastre = self.cadastre(index=cadastre)

        buildings = self.buildings()

        # The following is a dirty hack for preventing intersection time-out
        # See cadastre index 18.466 for why this is necessary
        queue = SimpleQueue()
        process = Process(
            target=lambda c, b, q: q.put(c.intersection(b)),
            args=(cadastre, buildings, queue),
        )
        process.start()
        process.join(10)
        if not queue.empty():
            intersecting_buildings = queue.get()
        else:
            raise RuntimeError(
                "Building intersection took more than 10 seconds. Aborting!",
            )

        if isinstance(intersecting_buildings, GeometryCollection):
            intersecting_buildings = MultiPolygon([
                feature
                for feature
                in cadastre.intersection(buildings)
                if not isinstance(feature, Point)
            ])

        with rasterio.open(self.lidar_path) as src:
            assert str(src.crs["proj"]) == "utm" and int(src.crs["zone"]) == 32
            cropped_data, affine_transformation = mask(
                src, shapes=[cadastre], all_touched=True, crop=True, filled=False
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

    def tiles(
        self,
        cadastre_index,
        with_tile_dimensions: bool = False,
        max_num_tiles: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return LiDAR and building tiles for given cadastre.

        :param cadastre_index: Positive integer identifying the cadastre.
        :param with_tile_dimensions: Return the original dimension of the tiles.
        :max_num_tiles: Skip tile generation if tiles exceed this number.
        """
        # (0.25m, 0.25m) pixels
        # (256px, 256px) tiles -> (64m, 64m) tiles
        cadastre = self.cadastre(index=cadastre_index)
        min_x, min_y, max_x, max_y = cadastre.bounds

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
                f"Cadastre index {cadastre_index} produced {num_tiles} tiles. "
                f"This is greater than max_num_tiles={max_num_tiles}, raising!"
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

        cropped_lidar_file, building_file = self.construct_observation(
            cadastre=bounding_box,
        )

        with cropped_lidar_file.open() as lidar_handle:
            lidar_array = lidar_handle.read(1)

        with building_file.open() as building_handle:
            building_array = building_handle.read(1)

        if lidar_array.shape[0] % 256 != 0:
            lidar_array = lidar_array[:-1, :]
            building_array = building_array[:-1, :]
        if lidar_array.shape[1] % 256 != 0:
            lidar_array = lidar_array[:, :-1]
            building_array = building_array[:, :-1]

        lidar_shape = lidar_array.shape
        building_shape = building_array.shape

        try:
            assert lidar_shape[0] % 256 == 0
            assert lidar_shape[1] % 256 == 0
            assert building_shape[0] % 256 == 0
            assert building_shape[1] % 256 == 0
        except AssertionError:
            raise RuntimeError(
                f"Cadastre index {cadastre_index} could not be reshaped to a"
                "multiple of (256, 256). The resulting shape is: "
                f"lidar_shape={lidar_shape}, building_shape={building_shape}."
            )

        lidar_tiles = view_as_blocks(lidar_array, (256, 256))
        building_tiles = view_as_blocks(building_array, (256, 256))

        tile_dimensions = lidar_tiles.shape[:2]
        number_of_tiles = tile_dimensions[0] * tile_dimensions[1]

        lidar_tiles = lidar_tiles.reshape(number_of_tiles, 256, 256)
        building_tiles = building_tiles.reshape(number_of_tiles, 256, 256)

        if with_tile_dimensions:
            return lidar_tiles, building_tiles, tile_dimensions
        return lidar_tiles, building_tiles

    def plot_tiles(
        self,
        cadastre_index: int,
        show: bool = True,
        with_legend: bool = True,
    ):
        lidar_tiles, building_tiles, tile_dimensions = self.tiles_cache(
            cadastre_index=cadastre_index, with_tile_dimensions=True,
        )
        fig, axes = plt.subplots(
            *tile_dimensions,
            figsize=(15, 15),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        vmin = lidar_tiles.min()
        vmax = lidar_tiles.max()
        for (lidar_tile, building_tile), ax \
                in zip(zip(lidar_tiles, building_tiles), axes.flatten()):
            lidar_image = ax.imshow(lidar_tile, vmin=vmin, vmax=vmax)
            ax.imshow(building_tile, alpha=0.1, cmap="binary")

        if with_legend and len(axes.flatten()) == 1:
            divider = make_axes_locatable(axes[0][0])
            colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(
                lidar_image,
                cax=colorbar_ax,
                format=FormatStrFormatter('%d m'),
            )
        else:
            plt.tight_layout()

        if show:
            fig.show()
        else:
            return fig, axes

    def input_tile_normalizer(self, tiles: np.ndarray) -> np.ndarray:
        if tiles.ndim == 2:
            # A single tile with squeezed channels is given
            tiles = tiles[np.newaxis, :, :, np.newaxis]
        elif tiles.ndim == 3:
            # A single tile with 1 channel is given
            tiles = tiles[np.newaxis, :, :]
        else:
            assert tiles.ndim == 4

        tiles[tiles > 1e5] = 0
        tiles[tiles < -1e5] = 0
        min_vals = np.min(
            tiles.reshape(tiles.shape[0], 256 * 256),
            axis=1,
        ).reshape(tiles.shape[0], 1, 1, 1)
        np.subtract(tiles, min_vals, out=tiles)

        max_vals = np.max(
            tiles.reshape(tiles.shape[0], 256 * 256),
            axis=1,
        ).reshape(tiles.shape[0], 1, 1, 1)
        np.divide(tiles, max_vals, out=tiles)
        return tiles

    def plot_prediction(self, model, cadastre_index):
        lidar_tiles, building_tiles, tile_dimensions = self.tiles_cache(
            cadastre_index=cadastre_index, with_tile_dimensions=True,
        )
        fig, axes = plt.subplots(
            nrows=tile_dimensions[0] * tile_dimensions[1],
            ncols=3,
            figsize=(15, 15),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes[0][0].title.set_text("Original LiDAR data")
        axes[0][1].title.set_text("Prediction probabilities")
        axes[0][2].title.set_text("TP / TN / FP / FN, cut-off = 0.5")

        disable_ticks = {
            "axis": "both",
            "which": "both",
            "bottom": False,
            "labelbottom": False,
            "width": 0.0,
        }
        # print(dir(axes[0][1]))
        axes[0][1].tick_params(**disable_ticks)
        axes[0][2].tick_params(**disable_ticks)

        vmin = lidar_tiles.min()
        vmax = lidar_tiles.max()

        for (lidar_tile, building_tile), (lidar_ax, prediction_ax, metric_ax) \
                in zip(zip(lidar_tiles, building_tiles), axes):
            lidar_ax.imshow(lidar_tile, vmin=vmin, vmax=vmax)

            lidar_tile = np.expand_dims(lidar_tile, 0)
            lidar_tile = np.expand_dims(lidar_tile, -1)
            normalized_lidar_tile = self.input_tile_normalizer(lidar_tile)

            predicted_building_tile = model.predict(normalized_lidar_tile)
            predicted_building_tile = np.squeeze(predicted_building_tile)
            prediction_ax.imshow(
                predicted_building_tile,
                cmap="seismic",
                vmin=0,
                vmax=1,
            )

            predicted_mask = (predicted_building_tile > 0.5).astype("uint8")
            TP = np.logical_and(predicted_mask == 1, building_tile == 1)
            TN = np.logical_and(predicted_mask == 0, building_tile == 0)
            FP = np.logical_and(predicted_mask == 1, building_tile == 0)
            FN = np.logical_and(predicted_mask == 0, building_tile == 1)
            confusion_matrix = TP + 2 * TN + 3 * FP + 4 * FN

            cmap = colors.ListedColormap(
                ['#001F3F', '#DDDDDD', '#2ECC40', '#FF4136']
            )
            bounds = [0, 1.5, 2.5, 3.5, 5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            metric_ax.imshow(confusion_matrix, cmap=cmap, norm=norm)

            # divider = make_axes_locatable(metric_ax)
            # colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)
            legend_elements = [
                Patch(facecolor='#001F3F', edgecolor="white", label='TP'),
                Patch(facecolor='#DDDDDD', edgecolor="white", label='TN'),
                Patch(facecolor='#2ECC40', edgecolor="white", label='FP'),
                Patch(facecolor='#FF4136', edgecolor="white", label='FN'),
            ]
            metric_ax.legend(
                handles=legend_elements,
                loc="lower center",
                ncol=4,
                bbox_to_anchor=(0.5, -0.075),
                frameon=False,
                handlelength=1.3,
                handleheight=1.5,
            )

        plt.tight_layout()
        plt.show()

    def tf_dataset(self, start=0, stop=1_000_000):
        def _generator():
            for index in range(start, stop):
                for lidar_array, building_array in zip(*self.tiles_cache(index)):
                    if building_array.sum() < 64:
                        continue
                    yield (
                        np.squeeze(self.input_tile_normalizer(lidar_array), 0),
                        np.expand_dims(building_array, -1),
                    )

        return tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape([256, 256, 1]), tf.TensorShape([256, 256, 1]),
            ),
            args=None,
        )

    def build_tile_cache(self, number_of_cadastre: int):
        """
        Construct tile cache by preprocessing cadastral data.

        cadastral_offsets[cadastral_index] = (first_tile_index, last_tile_index)
        """
        def save_cache_index(cache_index):
            self.cache_index_path.write_bytes(pickle.dumps(
                cache_index,
                protocol=pickle.HIGHEST_PROTOCOL,
                fix_imports=False,
            ))
            self._cache_index = cache_index

        if self.cache_index_path.exists():
            print("Importing cache index")
            cache_index = pickle.loads(
                self.cache_index_path.read_bytes(),
                fix_imports=False,
            )
        else:
            print("Creating new cache index")
            cache_index = {}
            save_cache_index(cache_index)

        cadastre_index_start = max(cache_index.keys() or [-1]) + 1
        print(cadastre_index_start)

        for cadastre_index in irange(
            cadastre_index_start, cadastre_index_start + number_of_cadastre
        ):
            try:
                image_tiles, mask_tiles, (height, width) = self.tiles(
                    cadastre_index=cadastre_index,
                    with_tile_dimensions=True,
                    max_num_tiles=100,
                )
                cache_path = self.tile_cache_path / f"{cadastre_index:07d}.npz"
                np.savez_compressed(
                    file=cache_path,
                    lidar=image_tiles,
                    buildings=mask_tiles,
                )
                cache_index[cadastre_index] = {
                    "height": height,
                    "width": width,
                    "cache_path": cache_path,
                }
                if cadastre_index % 100 == 0:
                    save_cache_index(cache_index)
            except Exception as exc:
                print(exc)
                print(
                    "Encountered exception for cadastre_index: "
                    + str(cadastre_index)
                )
                continue

        save_cache_index(cache_index)

    def tiles_cache(
        self,
        cadastre_index,
        with_tile_dimensions: bool = False,
    ) -> np.ndarray:
        if not hasattr(self, "_cache_index"):
            self._cache_index = pickle.loads(
                self.cache_index_path.read_bytes(),
                fix_imports=False,
            )
        cache_index = self._cache_index

        with np.load(
            cache_index[cadastre_index]["cache_path"],
            "r",
        ) as cache_file:
            lidar_tiles = cache_file["lidar"]
            building_tiles = cache_file["buildings"]

        if with_tile_dimensions:
            return (
                lidar_tiles,
                building_tiles,
                (
                    cache_index[cadastre_index]["height"],
                    cache_index[cadastre_index]["width"],
                ),
            )
        return lidar_tiles, building_tiles

    def __len__(self) -> int:
        return len(self.buildings())

    def __getitem__(self, index):
        """
        Return LiDAR tiles and masks for the given cadastrals.

        Returns a two tuple of two np.ndarrays, the first being all the LiDAR
        tiles, the second being the building masks.
        """
        images = []
        masks = []
        amount = index.stop - index.start
        range_function = range if amount < 50 else irange
        for cadastre_index in range_function(index.start, index.stop):
            try:
                tiles = self.tiles_cache(cadastre_index=cadastre_index)
            except Exception:
                continue
            for image_tile, mask_tile in zip(*tiles):
                if mask_tile.sum() < 64:
                    continue
                images.append([image_tile])
                masks.append([mask_tile])

        images = np.expand_dims(np.concatenate(images, axis=0), -1)
        images = self.input_tile_normalizer(images)
        masks = np.expand_dims(np.concatenate(masks, axis=0), -1)
        return images, masks
