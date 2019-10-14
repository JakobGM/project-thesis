"""Module responsible for fetching, pre-processing, and preparing data."""
import time
import warnings
from typing import Collection, Mapping, Optional, Tuple

import fiona

from ipypb import irange, track

from matplotlib import colors
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from shapely.geometry import (
    MultiPolygon,
    Polygon,
)

from sklearn.model_selection import train_test_split

import tensorflow as tf

from remsen import augmentation, cache


class Dataset:
    """Dataset class for cadastre+building+LiDAR data."""

    def __init__(
        self,
        cache: cache.Cache,
        lidar_dataset: str = "trondheim_lidar_2017",
        rgb_dataset: str = "trondheim_rgb_2017",
        mask_dataset: str = "trondheim_bygning",
    ) -> None:
        """Construct dataset."""
        self.cache = cache
        self.cache.change_dataset(
            lidar_name=lidar_dataset,
            rgb_name=rgb_dataset,
            mask_name=mask_dataset,
        )

        self.lidar_nodata_value = cache.lidar_metadata["nodata_value"]
        assert self.lidar_nodata_value < 0

    def cadastre(self, index: int) -> Polygon:
        """Fetch cadastre from dataset."""
        return self.cache.cadastre(index=index)

    def buildings(self) -> MultiPolygon:
        """Fetch all buildings from dataset."""
        if hasattr(self, "_buildings"):
            # In-memory cache
            return self._buildings
        else:
            # On-disk cache
            self._buildings = self.cache.mask_geometry()
            return self._buildings

    def building(self, index: int) -> Polygon:
        """Fetch specific building from dataset."""
        return self.buildings()[index]

    def plot_tiles(
        self,
        cadastre_index: int,
        show: bool = True,
        with_legend: bool = True,
        rgb: bool = False,
    ):
        building_tiles = self.cache.mask_tiles(cadastre_index=cadastre_index)
        tile_dimensions = self.cache.tile_dimensions(
            cadastre_index=cadastre_index,
        )
        if rgb:
            background_tiles = self.cache.rgb_tiles(
                cadastre_index=cadastre_index,
            )
        else:
            background_tiles = self.cache.lidar_tiles(
                cadastre_index=cadastre_index,
            )

        background_tiles = np.array(background_tiles)

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
            [patheffects.withStroke(linewidth=2, foreground='black', alpha=0.3)],
        )

        if not rgb:
            vmin, vmax = background_tiles.min(), background_tiles.max()
        else:
            vmin, vmax = None, None

        for (lidar_tile, building_tile), ax \
                in zip(zip(background_tiles, building_tiles), axes.flatten()):
            lidar_image = ax.imshow(
                np.squeeze(lidar_tile),
                vmin=vmin,
                vmax=vmax,
            )

            rgba_building = np.zeros(shape=(256, 256, 4))

            # Set building area to be semi-transparent red
            rgba_building[:, :, 0] = building_tile[:, :, 0]
            is_building = (building_tile == 1).reshape(256, 256)
            rgba_building[:, :, 3][is_building] = 0.25 if rgb else 0.2
            ax.imshow(rgba_building, cmap="binary")

        if with_legend and len(axes.flatten()) == 1 and not rgb:
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
            [patheffects.withStroke(linewidth=2, foreground='black', alpha=0.3)],
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
        shuffle: bool = True,
        train_split: float = 0.70,
        validation_split: float = 0.15,
        test_split: float = 0.15,
        minimum_building_area: float = 4,
        rgb: bool = False,
        lidar: bool = True,
    ):
        assert train_split + validation_split + test_split == 1.0
        if rgb and lidar:
            num_channels = 4
        elif rgb:
            num_channels = 3
        elif lidar:
            num_channels = 1
        else:
            raise ValueError("Either rgb or lidar must be True.")

        # TODO: Allow either of the two augmentation methods
        def _generator(cadastre_indeces):
            for index in cadastre_indeces:
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

        # Split all data into train, validation, and test subsets
        cadaster_indeces = self.cache.cadastre_indeces
        train_indeces, remaining = train_test_split(
            cadaster_indeces,
            train_size=train_split,
            shuffle=True,
            random_state=42,
        )
        val_indeces, test_indeces = train_test_split(
            remaining,
            train_size=validation_split / (validation_split + test_split),
            shuffle=False,
            random_state=43,
        )

        # Persist cadastre splits to self
        self.train_cadastre = train_indeces
        self.validation_cadastre = val_indeces
        self.test_cadastre = test_indeces

        train = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape([256, 256, num_channels]),
                tf.TensorShape([256, 256, 1]),
            ),
            args=(train_indeces,),
        )
        validation = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape([256, 256, num_channels]),
                tf.TensorShape([256, 256, 1]),
            ),
            args=(val_indeces,),
        )
        test = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape([256, 256, num_channels]),
                tf.TensorShape([256, 256, 1]),
            ),
            args=(test_indeces,),
        )

        # Train data augmentation
        if augment:
            train.map(
                map_func=augmentation.flip_and_rotate,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        # Prefetching
        train = train.prefetch(buffer_size=prefetch)
        validation = validation.prefetch(buffer_size=prefetch)
        test = test.prefetch(buffer_size=prefetch)

        # Train data shuffling
        if shuffle:
            train = train.shuffle(buffer_size=512)

        # Batching
        train = train.batch(batch_size=batch_size, drop_remainder=False)
        validation = validation.batch(batch_size=batch_size, drop_remainder=False)
        test = test.batch(batch_size=batch_size, drop_remainder=False)

        return train, validation, test

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

    def tiles(
        self,
        cadastre_index,
        *,
        with_tile_dimensions: bool = False,
        max_num_tiles: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return LiDAR and building tiles for given cadastre.

        :param cadastre_index: Positive integer identifying the cadastre.
        :param max_num_tiles: Skip tile generation if tiles exceed this number.
        """
        lidar_tiles = self.cache.lidar_tiles(cadastre_index=cadastre_index)
        if max_num_tiles and len(lidar_tiles) > max_num_tiles:
            return

        building_tiles = self.cache.mask_tiles(cadastre_index=cadastre_index)
        if with_tile_dimensions:
            dimensions = self.cache.tile_dimensions(
                cadastre_index=cadastre_index,
            )
            return lidar_tiles, building_tiles, dimensions
        else:
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
