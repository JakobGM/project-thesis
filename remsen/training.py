import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING

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
            custom_objects={"iou": iou},
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
