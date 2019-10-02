"""Module responsible for fetching, pre-processing, and preparing data."""
import pickle
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Collection, Dict, Mapping, Optional, Tuple, Union

import fiona

from ipypb import irange, track

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
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

from remsen import augmentation


def fiona_polygon(fiona_item: Dict) -> Polygon:
    """Convert fiona item to Shapely polygon."""
    geometry = shape(fiona_item["geometry"])
    if not geometry.is_valid:
        geometry = geometry.buffer(0.0)
    assert geometry.is_valid
    assert geometry.geom_type in ("Polygon", "MultiPolygon")
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
        with rasterio.open(lidar_path, "r") as lidar_file:
            self.lidar_nodata_value = lidar_file.nodata
        assert self.lidar_nodata_value < 0

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tile_cache_path = cache_dir / "tiles"
        self.tile_cache_path.mkdir(parents=True, exist_ok=True)

    def cadastre(self, index: int) -> Polygon:
        """Fetch cadastre from dataset."""
        with fiona.open(self.cadastre_path, "r", layer="Teig") as src:
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
        intersecting_buildings = cadastre.intersection(buildings)

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
        cadastre_text = axes[0][-1].annotate(
            f'Cadastre {cadastre_index}',
            xy=(0.98, 0.98),
            xycoords='axes fraction',
            size=20,
            ha='right',
            va='top',
            color="white",
            weight="bold",
            alpha=0.5,
        )
        cadastre_text.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground='black', alpha=0.3)],
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

        # Subtract minimum value of each tile independetly, ignoring nodata
        # Mask nodata values
        masked_tiles = np.ma.asarray(tiles)
        masked_tiles.mask = tiles == self.lidar_nodata_value

        # Calculate minimum of each tile, excluding nodataa
        min_vals = np.ma.min(
            masked_tiles.reshape(tiles.shape[0], 256 * 256),
            axis=1,
        ).reshape(masked_tiles.shape[0], 1, 1, 1)

        # Subtract tile minimum for each tile, leaving nodata values alone
        np.subtract(
            masked_tiles,
            min_vals,
            out=tiles,
            where=np.logical_not(masked_tiles.mask),
        )

        # Set nodata values equal to zero
        tiles[masked_tiles.mask] = 0

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

        cadastre_text = axes[0][0].annotate(
            f'Cadastre\n{cadastre_index}',
            xy=(0.98, 0.98),
            xycoords='axes fraction',
            size=14,
            ha='right',
            va='top',
            color="white",
            weight="bold",
            alpha=0.8,
        )
        cadastre_text.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground='black', alpha=0.3)],
        )

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

            # Add TP/TN/FP/FN legend to plot
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

            # Add metrics to plot
            building_tile = np.expand_dims(building_tile, 0)
            building_tile = np.expand_dims(building_tile, -1)
            evaluation = model.evaluate(
                x=normalized_lidar_tile,
                y=building_tile,
                verbose=0,
            )
            metrics = {
                name: value
                for name, value
                in zip(model.metrics_names, evaluation)
            }
            loss = metrics["loss"]
            mean_iou = metrics["mean_io_u"]
            prediction_ax.set_xlabel(
                f"Loss = {loss:.4f},   Mean IoU = {mean_iou:0.4f}",
                size=13,
            )

        plt.tight_layout()
        plt.show()

    def plot_worst(
        self,
        model: tf.keras.Model,
        cadastre_indeces: Collection[int] = range(0, 10),
        metric: Tuple[Mapping, str] = (min, "mean_io_u"),
        number: int = 5,
    ):
        """Plot the worst predictions according to a given metric."""
        cadastre_metrics = {}
        optimizer, metric = metric

        for cadastre_index in track(cadastre_indeces):
            try:
                x, y = self[cadastre_index:cadastre_index + 1]
            except ValueError:
                continue

            if x.shape[0] > 1:
                continue

            evaluation = model.evaluate(x=x, y=y, verbose=0)
            metrics = {
                name: value
                for name, value
                in zip(model.metrics_names, evaluation)
            }
            cadastre_metrics[cadastre_index] = metrics

        number = min(number, len(cadastre_metrics))
        for _ in range(number):
            worst_cadastre = optimizer(
                cadastre_metrics.keys(),
                key=lambda key: cadastre_metrics[key][metric],
            )
            self.plot_prediction(model=model, cadastre_index=worst_cadastre)
            del cadastre_metrics[worst_cadastre]

    def tf_dataset(
        self,
        batch_size: int = 16,
        prefetch: int = 1000,
        augment: bool = True,
        validation_split: float = 0.05,
        dataset_size: int = 47_853,
        minimum_building_area: float = 4,
    ):
        # TODO: Allow either of the two augmentation methods
        def _generator(start, stop):
            for index in range(start, stop):
                try:
                    for lidar_array, building_array in zip(*self.tiles_cache(index)):
                        if building_array.sum() < (minimum_building_area * 16):
                            continue
                        yield (
                            np.squeeze(self.input_tile_normalizer(lidar_array), 0),
                            np.expand_dims(building_array, -1),
                        )
                except KeyError:
                    continue
                except Exception:
                    # TODO: log exception here
                    continue

        # TODO: Extract 47_853
        validation_size = int(dataset_size * validation_split)
        train_size = int(dataset_size * (1 - validation_split))
        validation_start_index = 47_853 - validation_size
        if validation_start_index < train_size:
            validation_start_index = train_size

        train = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape([256, 256, 1]), tf.TensorShape([256, 256, 1]),
            ),
            args=(0, train_size),
        )
        validation = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape([256, 256, 1]), tf.TensorShape([256, 256, 1]),
            ),
            args=(validation_start_index, 47_853),
        )

        # Data augmentation
        if augment:
            train.map(
                map_func=augmentation.flip_and_rotate,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        # Prefetching
        train = train.prefetch(buffer_size=prefetch)
        validation = validation.prefetch(buffer_size=prefetch)

        # Batching
        train = train.batch(batch_size=batch_size, drop_remainder=False)
        validation = validation.batch(batch_size=batch_size, drop_remainder=False)

        return train, validation

    def _save_tile(self, cadastre_index):
        """Save processed cadastre to cache directory."""
        cache_path = self.tile_cache_path / f"{cadastre_index:07d}.npz"
        if cache_path.exists():
            return

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Cannot provide views")
                image_tiles, mask_tiles, (height, width) = self.tiles(
                    cadastre_index=cadastre_index,
                    with_tile_dimensions=True,
                    max_num_tiles=100,
                )
            np.savez_compressed(
                file=cache_path,
                lidar=image_tiles,
                buildings=mask_tiles,
                dimensions=np.array([height, width]),
            )
        except fiona.errors.DriverError:
            print("Fiona race condition encountered. Sleeping for 1 second...")
            time.sleep(1)
            return self._save_tile(cadastre_index)
        except RuntimeError as exc:
            print(exc)
            return

    def build_tile_cache(self):
        """
        Construct tile cache by preprocessing cadastral data.
        """

        cadastre_indeces = range(0, 47_853)
        pool = Pool(processes=None)
        pool_tasks = pool.imap(
            func=self._save_tile,
            iterable=cadastre_indeces,
            chunksize=1,
        )
        for result in pool_tasks:
            pass

    def tiles_cache(
        self,
        cadastre_index,
        with_tile_dimensions: bool = False,
    ) -> np.ndarray:

        tile_path = self.tile_cache_path / f"{cadastre_index:07d}.npz"
        if not tile_path.exists():
            raise KeyError

        with np.load(tile_path, "r") as cache_file:
            lidar_tiles = cache_file["lidar"]
            building_tiles = cache_file["buildings"]
            dimensions = tuple(cache_file["dimensions"])

        if with_tile_dimensions:
            return lidar_tiles, building_tiles, dimensions
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
