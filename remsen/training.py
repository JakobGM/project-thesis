import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from ipypb import track

import numpy as np

import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.models import Model, load_model

from remsen.losses import dice_loss, iou_loss
from remsen.metrics import iou
if TYPE_CHECKING:
    from remsen.data import Dataset


def get_callbacks(
    cache_dir: Path,
    early_stopping: bool = False,
    verbose: int = 1,
):
    iou_path = cache_dir / "val_iou" / "{epoch}.h5"
    iou_path.parent.mkdir(parents=True, exist_ok=True)

    latest_path = cache_dir / "latest" / "{epoch}.h5"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    loss_path = cache_dir / "val_loss" / "{epoch}.h5"
    loss_path.parent.mkdir(parents=True, exist_ok=True)

    save_best_val_loss = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filepath=str(loss_path),
        save_weights_only=True,
        save_best_only=True,
        verbose=verbose,
    )
    save_best_val_iou = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        filepath=str(iou_path),
        save_weights_only=True,
        save_best_only=True,
        verbose=verbose,
    )
    save_latest = ModelCheckpoint(
        filepath=str(latest_path),
        save_weights_only=True,
        verbose=0,
    )
    callbacks = [save_latest, save_best_val_loss, save_best_val_iou]

    if early_stopping:
        earlystop_callback = EarlyStopping(
            monitor='val_iou',
            min_delta=0.001,
            patience=20,
            verbose=1,
        )
        callbacks.append(earlystop_callback)

    return callbacks


class Trainer:
    def __init__(self, name: str, model: Model, dataset: "Dataset") -> None:
        self.name = name
        self.model = model
        self.dataset = dataset
        self.cache_path = dataset.cache.parent_dir
        self.checkpoint_path = self.cache_path / "models" / name
        self.model_path = self.checkpoint_path / "model.h5"
        self.tensorboard_dir = self.cache_path / "tensorboard" / name

        if self.checkpoint_path.exists() or self.tensorboard_dir.exists():
            self._existing_model()
        else:
            self.initial_epoch = 0
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.model_path))

        self.callbacks = self._callbacks()

    def _existing_model(self):
        message = [
            f"Model trainer with name {self.name} already exists!",
            "What do you want to do?",
            "1) Delete existing model checkpoints and TensorBoard data",
            "2) Load latest training epoch",
            "3) Load best validation IoU model",
            "4) Load best validation loss model",
        ]

        answer = input("\n".join(message))
        if answer not in "1 2 3 4".split():
            self._existing_model()

        if answer == "1":
            answer = input(
                "You sure? "
                f"You will delete {self.checkpoint_path} "
                f"and {self.tensorboard_dir}... [y/N]",
            )
            if answer.upper() != "Y":
                self._existing_model()
            shutil.rmtree(str(self.checkpoint_path), ignore_errors=True)
            shutil.rmtree(str(self.tensorboard_dir), ignore_errors=True)
            shutil.rmtree(str(self.model_path), ignore_errors=True)

            # Recreate directories
            self.tensorboard_dir.mkdir(exist_ok=True)
            self.checkpoint_path.mkdir(exist_ok=True)

            self.model.save(str(self.model_path))
            self.initial_epoch = 0
            return
        elif answer == "2":
            path = self.checkpoint_path / "latest"
        elif answer == "3":
            path = self.checkpoint_path / "val_iou"
        else:
            path = self.checkpoint_path / "val_loss"

        checkpoint = max(
            path.glob("*.h5"),
            key=lambda p: int(p.name.split(".")[0]),
        )
        self.model = load_model(
            str(self.model_path),
            custom_objects={
                "iou": iou,
                "iou_loss": iou_loss,
                "dice_loss": dice_loss,
            },
        )
        self.model.load_weights(str(checkpoint))
        self.initial_epoch = int(checkpoint.name.split(".")[0])

    def _callbacks(self):
        callbacks = get_callbacks(
            cache_dir=self.checkpoint_path,
            early_stopping=False,
        )
        tensorboard_callback = TensorBoard(
            log_dir=str(self.tensorboard_dir),
            update_freq="epoch",
        )
        callbacks.append(tensorboard_callback)
        return callbacks

    def train(self, verbose: int = 1, **kwargs):
        num_channels = self.model.input.shape[3]
        lidar = num_channels in (1, 4)
        rgb = num_channels in (3, 4)

        self.train_kwargs, self.test_dataset = self.dataset.tf_dataset(
            **kwargs,
            rgb=rgb,
            lidar=lidar,
        )
        self.train_kwargs["epochs"] += self.initial_epoch
        history = self.model.fit(
            **self.train_kwargs,
            callbacks=self.callbacks,
            verbose=verbose,
            initial_epoch=self.initial_epoch,
        )
        self.initial_epoch = history.epoch[-1] + 1

    def calculate_evaluation_statistics(self) -> pd.DataFrame:
        # Add new metrics to model
        model = self.model
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics + [
                "binary_accuracy",
                "FalseNegatives",
                "FalsePositives",
                "Precision",
                "Recall",
            ],
        )

        # Construct dictionary which will become dataframe
        results = {
            "cadastre": [],
            "tile": [],
            "iou": [],
            "binary_accuracy": [],
            "false_positives": [],
            "false_negatives": [],
            "precision": [],
            "recall": [],
            "loss": [],
            "nodata": [],
            "split": [],
            "mask": [],
            "min_elevation": [],
            "max_elevation": [],
            "range": [],
        }

        # Use cadastre indeces and their respective split allocation
        self.dataset.create_splits()
        train = self.dataset.train_cadastre.copy()
        val = self.dataset.validation_cadastre.copy()
        test = self.dataset.test_cadastre.copy()
        indices = sorted([
            *zip(train, ["train"] * len(train)),
            *zip(val, ["val"] * len(val)),
            *zip(test, ["test"] * len(test)),
        ])

        # Find the type of data used by the model
        num_channels = model.input.shape[3]
        with_rgb = num_channels in (3, 4)
        with_lidar = num_channels in (1, 4)

        for index, split in track(indices):
            # Get all tiles for the cadastre
            tiles = self.dataset.tiles(
                cadastre_index=index,
                with_tile_dimensions=False,
                with_rgb=True,
            )
            lidar_tiles = tiles["lidar"]
            mask_tiles = tiles["mask"]
            rgb_tiles = tiles["rgb"]
            iterator = zip(lidar_tiles, mask_tiles, rgb_tiles)

            for tile_number, (lidar_tile, mask_tile, rgb_tile) in enumerate(iterator):
                # Identifying columns
                results["cadastre"].append(index)
                results["tile"].append(tile_number)
                results["split"].append(split)

                # Number of pixels that constitute buildings
                results["mask"].append(mask_tile.sum())

                # Nodata values
                nodata_indices = lidar_tile == self.dataset.lidar_nodata_value
                nodata = nodata_indices.sum()
                results["nodata"].append(nodata)

                # Calculate elevation ranges
                valid_lidar = lidar_tile[~nodata_indices]
                min_elevation = valid_lidar.min()
                max_elevation = valid_lidar.max()
                results["min_elevation"].append(min_elevation)
                results["max_elevation"].append(max_elevation)
                results["range"].append(max_elevation - min_elevation)

                # Create inputs for evaluation
                lidar_input = lidar_tile.reshape(1, 256, 256, 1)
                mask_tile = mask_tile.reshape(1, 256, 256, 1)
                rgb_tile = rgb_tile.reshape(1, 256, 256, 3)
                input_arrays = []
                if with_lidar:
                    lidar_arrays = self.dataset.input_tile_normalizer(lidar_input)
                    input_arrays.append(lidar_arrays)
                if with_rgb:
                    rgb_tile = rgb_tile.astype("float32")
                    rgb_tile /= 255
                    input_arrays.append(rgb_tile)
                input_arrays = np.concatenate(input_arrays, axis=3)

                # Create inputs for evaluation and evaluate
                # lidar_input = self.dataset.input_tile_normalizer(tiles=lidar_tile)
                evaluation = model.test_on_batch(input_arrays, mask_tile)

                metrics = {
                    name: value
                    for name, value
                    in zip(model.metrics_names, evaluation)
                }
                results["iou"].append(metrics["iou"])
                results["loss"].append(metrics["loss"])
                results["binary_accuracy"].append(metrics["binary_accuracy"])
                results["false_negatives"].append(metrics["FalseNegatives"])
                results["false_positives"].append(metrics["FalsePositives"])
                results["precision"].append(metrics["Precision"])
                results["recall"].append(metrics["Recall"])
        df = pd.DataFrame.from_dict(results)
        df["split"] = df.split.astype("category")

        # Save for use by self.evaluation_statistics()
        cache_path = self.checkpoint_path / "evaluation.pkl"
        df.to_pickle(cache_path)
        return df

    @classmethod
    def evaluation_statistics(cls, name: Optional[str] = None):
        if name:
            cache_path = Path(f".cache/models/{name}/evaluation.pkl")
            if cache_path.exists():
                return pd.read_pickle(cache_path)
        else:
            models = list(Path(".cache/models").iterdir())
            return {m.name: cls.evaluation_statistics(m.name) for m in models}


def tensorboard_dataframe(
    name: str,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """Return dataframe representing data in TensorBoard log."""
    if not split:
        validation = tensorboard_dataframe(name=name, split="validation")
        train = tensorboard_dataframe(name=name, split="train")
        validation.pop("datetime")
        validation.pop("elapsed_time")
        return train.join(validation, on="epoch")

    directory = Path(f".cache/tensorboard/{name}/{split}")
    events = EventAccumulator(str(directory))
    events.Reload()

    assert split in ("train", "validation")
    split_prefix = "train_" if split == "train" else "val_"

    dataframe = defaultdict(list)
    for scalar_tag in events.Tags()["scalars"]:
        wall_times, step_numbers, values = zip(*events.Scalars(scalar_tag))
        dataframe["datetime"] = wall_times
        dataframe["epoch"] = step_numbers
        dataframe[split_prefix + scalar_tag[6:]] = values

    dataframe = pd.DataFrame.from_dict(dataframe)
    dataframe["elapsed_time"] = pd.to_timedelta(
        dataframe["datetime"] - dataframe["datetime"].min(),
        unit="s",
    )
    dataframe["datetime"] = pd.to_datetime(dataframe["datetime"], unit="s")
    dataframe.set_index(keys="epoch", inplace=True)
    return dataframe


def model_comparison(models: List[str], titles: Optional[List[str]] = None):
    iou = []
    binary_accuracy = []
    precision = []
    recall = []

    for model in models:
        stats = Trainer.evaluation_statistics(name=model)
        iou.append(stats.iou.mean())
        binary_accuracy.append(stats.binary_accuracy.mean())
        precision.append(stats.precision.mean())
        recall.append(stats.recall.mean())

    dict = {
        "Model": titles or models,
        "IoU": iou,
        "Accuracy": binary_accuracy,
        "Precision": precision,
        "Recall": recall,
    }

    def formatter(value, data, percent=True):
        data = np.array(data)
        if percent:
            value_string = f"{100 * value:2.2f}" + r"\%"
        else:
            value_string = f"{value:0.4f}"

        if value == data.max():
            return r"\textcolor{darkgreen}{" + value_string + r"}"
        elif value == data.min():
            return r"\textcolor{red}{" + value_string + r"}"
        else:
            return value_string

    df = pd.DataFrame.from_dict(dict)
    df.set_index("Model", inplace=True)
    latex = df.to_latex(
        index=True,
        formatters={
            "IoU": partial(formatter, data=iou, percent=False),
            "Accuracy": partial(formatter, data=binary_accuracy),
            "Precision": partial(formatter, data=precision),
            "Recall": partial(formatter, data=recall),
        },
        escape=False,
    )
    return latex
