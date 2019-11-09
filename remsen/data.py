"""Module responsible for fetching, pre-processing, and preparing data."""
from typing import Collection, Dict, Mapping, Optional, Tuple

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

from remsen import augmentation, cache, plot, utils


class Dataset:
    """Dataset class for cadastre+building+LiDAR data."""

    def __init__(
        self,
        cadastre_dataset: str = "trondheim_cadastre_2019",
        lidar_dataset: str = "trondheim_lidar_2017",
        rgb_dataset: str = "trondheim_rgb_2017",
        mask_dataset: str = "trondheim_buildings_2019",
    ) -> None:
        """Construct dataset."""
        self.cache = cache.Cache.from_name(cadastre_dataset)
        self.cache.change_dataset(
            lidar_name=lidar_dataset,
            rgb_name=rgb_dataset,
            mask_name=mask_dataset,
        )

        self.lidar_nodata_value = self.cache.lidar_metadata["nodata_value"]
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
        # TODO: Recast to same dimension
        original_ndim = tiles.ndim
        if original_ndim == 2:
            # A single tile with squeezed channels is given
            tiles = tiles[np.newaxis, :, :, np.newaxis]
        elif original_ndim == 3:
            # A single tile with 1 channel is given
            tiles = tiles[np.newaxis, :, :]
        else:
            assert original_ndim == 4

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

    def plot_prediction(self, model, cadastre_index, tile_index: int = 0):
        with_rgb = model.input.shape[-1] == 4
        tiles = self.tiles(
            cadastre_index=cadastre_index,
            with_tile_dimensions=False,
            with_rgb=with_rgb,
        )
        lidar_tile = tiles["lidar"][tile_index]
        building_tile = tiles["mask"][tile_index]
        input_tile = self.input_tile_normalizer(lidar_tile.copy())[0]

        if with_rgb:
            rgb_tile = tiles["rgb"][tile_index]
            input_tile = np.concatenate(
                [input_tile, rgb_tile / 255],
                axis=2,
            )
        input_tile = np.expand_dims(input_tile, 0)

        # (2, 2) plot if with RGB, otherwise (1, 3)
        fig, axes = plt.subplots(
            nrows=2 if with_rgb else 1,
            ncols=2 if with_rgb else 3,
            figsize=(15, 15),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        if with_rgb:
            lidar_ax, rgb_ax, prediction_ax, metric_ax = axes.flatten()
            rgb_ax.title.set_text("RGB data")
            plot.imshow_with_mask(image=rgb_tile, mask=building_tile, ax=rgb_ax)
        else:
            lidar_ax, prediction_ax, metric_ax = axes.flatten()

        lidar_ax.title.set_text("LiDAR data")
        prediction_ax.title.set_text("Prediction probabilities")
        metric_ax.title.set_text("TP / TN / FP / FN, cut-off = 0.5")

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

        for axis in axes.flatten():
            axis.tick_params(labelbottom=False, labelleft=False, width=0.0)

        lidar_tile = np.squeeze(lidar_tile)
        plot.imshow_with_mask(image=lidar_tile, mask=building_tile, ax=lidar_ax)

        lidar_tile = np.expand_dims(lidar_tile, 0)
        lidar_tile = np.expand_dims(lidar_tile, -1)

        predicted_building_tile = model.predict(input_tile)
        predicted_building_tile = np.squeeze(predicted_building_tile)
        prediction_ax.imshow(
            predicted_building_tile,
            cmap="seismic",
            vmin=0,
            vmax=1,
        )
        plot.imshow_with_mask(
            image=predicted_building_tile,
            mask=building_tile,
            cmap="seismic",
            ax=prediction_ax,
            edge_color=(0, 255, 0, 255),
        )

        predicted_mask = (predicted_building_tile > 0.5).astype("uint8")
        building_tile = np.squeeze(building_tile)
        tp = np.logical_and(predicted_mask == 1, building_tile == 1)
        tn = np.logical_and(predicted_mask == 0, building_tile == 0)
        fp = np.logical_and(predicted_mask == 1, building_tile == 0)
        fn = np.logical_and(predicted_mask == 0, building_tile == 1)
        confusion_matrix = tp + 2 * tn + 3 * fp + 4 * fn

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
            x=input_tile,
            y=building_tile,
            verbose=0,
        )
        metrics = {
            name: value
            for name, value
            in zip(model.metrics_names, evaluation)
        }
        loss = metrics["loss"]
        iou = metrics["iou"]
        prediction_ax.set_xlabel(
            f"Loss = {loss:.4f},   IoU = {iou:0.4f}",
            size=13,
        )

        plt.tight_layout()
        plt.show()

    def plot_worst(
        self,
        model: tf.keras.Model,
        cadastre_indeces: Collection[int] = range(0, 10),
        metric: Tuple[Mapping, str] = (min, "iou"),
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

            if x is None or x.shape[0] > 1:
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
        epochs: int = 1,
        minimum_building_area: float = 4,
        rgb: bool = False,
        lidar: bool = True,
        split_images: Optional[int] = None,
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

        def _generator(cadastre_indeces, epochs: int = 1):
            for _ in range(epochs):
                batch = []
                for index in cadastre_indeces:
                    tiles = self.tiles(index, with_rgb=rgb)
                    building_arrays = tiles["mask"]

                    input_arrays = []
                    if lidar:
                        lidar_arrays = tiles["lidar"]
                        lidar_arrays = self.input_tile_normalizer(lidar_arrays)
                        input_arrays.append(lidar_arrays)

                    if rgb:
                        rgb_arrays = np.array(tiles["rgb"])
                        rgb_arrays = rgb_arrays.astype("float32")
                        rgb_arrays /= 255
                        input_arrays.append(rgb_arrays)

                    input_arrays = np.concatenate(input_arrays, axis=3)
                    if split_images:
                        input_arrays = utils.split_images(
                            batch=input_arrays,
                            divisor=split_images,
                        )
                        building_arrays = utils.split_images(
                            batch=building_arrays,
                            divisor=split_images,
                        )

                    all_tiles = zip(input_arrays, building_arrays)
                    all_tiles = filter(
                        lambda x: x[1].sum() > (minimum_building_area * 16),
                        all_tiles,
                    )
                    for tiles in all_tiles:
                        batch.append(tiles)
                        if len(batch) == batch_size:
                            for x, y in batch:
                                x = tf.convert_to_tensor(x, dtype=tf.float32)
                                y = tf.convert_to_tensor(y, dtype=tf.bool)
                                yield x, y
                            batch = []

        # Split all data into train, validation, and test subsets
        cadastre_indeces = self.cache.cadastre_indeces
        train_indeces, remaining = train_test_split(
            cadastre_indeces,
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

        # TODO: Implement this with Cache.dataframe query
        print("Calculating number of training tiles...")
        num_train_tiles = 0
        for _ in _generator(train_indeces, epochs=1):
            num_train_tiles += 1

        print("Calculating number of validation tiles...")
        num_val_tiles = 0
        for _ in _generator(val_indeces, epochs=1):
            num_val_tiles += 1

        train = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.bool),
            output_shapes=(
                tf.TensorShape([256, 256, num_channels]),
                tf.TensorShape([256, 256, 1]),
            ),
            args=(train_indeces, epochs),
        )
        validation = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.bool),
            output_shapes=(
                tf.TensorShape([256, 256, num_channels]),
                tf.TensorShape([256, 256, 1]),
            ),
            args=(val_indeces, epochs),
        )
        test = tf.data.Dataset.from_generator(
            generator=_generator,
            output_types=(tf.float32, tf.bool),
            output_shapes=(
                tf.TensorShape([256, 256, num_channels]),
                tf.TensorShape([256, 256, 1]),
            ),
            args=(test_indeces,),
        )

        # Prefetching
        train_prefetch = min(prefetch, epochs * num_train_tiles)
        val_prefetch = min(prefetch, epochs * num_val_tiles)
        train = train.prefetch(
            buffer_size=train_prefetch,
        )
        validation = validation.prefetch(buffer_size=val_prefetch)

        # Train data shuffling
        if shuffle:
            train = train.shuffle(buffer_size=min(512, train_prefetch))

        # Batching
        train = train.batch(batch_size=batch_size)
        validation = validation.batch(batch_size=batch_size)
        test = test.batch(batch_size=batch_size)

        # Train data augmentation
        if augment:
            train.map(
                map_func=augmentation.flip_and_rotate,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        assert num_train_tiles % batch_size == 0
        assert num_val_tiles % batch_size == 0
        steps_per_epoch = int(num_train_tiles // batch_size)
        validation_steps = int(num_val_tiles // batch_size)

        train_kwargs = {
            "x": train,
            "validation_data": validation,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
        }

        return train_kwargs, test

    def tiles(
        self,
        cadastre_index,
        *,
        with_tile_dimensions: bool = False,
        max_num_tiles: Optional[int] = None,
        with_rgb: bool = False,
    ) -> Dict:
        """
        Return LiDAR and building tiles for given cadastre.

        :param cadastre_index: Positive integer identifying the cadastre.
        :param max_num_tiles: Skip tile generation if tiles exceed this number.
        """
        lidar_tiles = self.cache.lidar_tiles(cadastre_index=cadastre_index)
        if max_num_tiles and len(lidar_tiles) > max_num_tiles:
            return {}

        building_tiles = self.cache.mask_tiles(cadastre_index=cadastre_index)
        lidar_tiles = np.array(lidar_tiles)
        building_tiles = np.array(building_tiles)

        result = {
            "lidar": lidar_tiles,
            "mask": building_tiles,
        }

        if with_rgb:
            rgb_tiles = self.cache.rgb_tiles(cadastre_index=cadastre_index)
            result["rgb"] = rgb_tiles

        if with_tile_dimensions:
            dimensions = self.cache.tile_dimensions(
                cadastre_index=cadastre_index,
            )
            result["dimensions"] = dimensions

        return result

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
            tiles = self.tiles(
                cadastre_index=cadastre_index,
                with_tile_dimensions=False,
            )
            image_tiles = tiles["lidar"]
            mask_tiles = tiles["mask"]
            if image_tiles is None:
                continue
            for image_tile, mask_tile in zip(image_tiles, mask_tiles):
                if mask_tile.sum() < 64:
                    continue
                images.append(image_tile)
                masks.append(mask_tile)

        if not images:
            return None, None

        images = np.array(images)
        masks = np.array(masks)
        images = self.input_tile_normalizer(images)
        return images, masks
