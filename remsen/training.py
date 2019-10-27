from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.models import Model

from remsen.data import Dataset


def get_callbacks(
    model_name: str,
    cache_path: Path,
    early_stopping: bool = False,
    verbose: int = 0,
):
    model_cache_path = cache_path / "models" / model_name

    model_cache_path.mkdir(parents=True, exist_ok=True)

    save_best_val_loss = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filepath=str(model_cache_path / "val_loss.ckpt"),
        save_weights_only=True,
        save_best_only=True,
        verbose=verbose,
    )
    save_best_val_iou = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        filepath=str(model_cache_path / "val_iou.ckpt"),
        save_weights_only=False,
        save_best_only=True,
        verbose=verbose,
    )
    save_latest = ModelCheckpoint(
        filepath=str(model_cache_path / "latest.ckpt"),
        save_weights_only=False,
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
    def __init__(self, name: str, model: Model, dataset: Dataset) -> None:
        self.name = name
        self.model = model
        self.dataset = dataset
        self.cache_path = dataset.cache.parent_dir

    def callbacks(self):
        callbacks = get_callbacks(
            model_name=self.name,
            cache_path=self.cache_path,
            early_stopping=True,
        )
        tensorboard_callback = TensorBoard(
            log_dir=str(self.cache_path / "tensorboard" / self.name),
            update_freq="epoch",
        )
        callbacks.append(tensorboard_callback)
        return callbacks

    def train(self, verbose: int = 0, **kwargs):
        self.train_kwargs, self.test_dataset = self.dataset.tf_dataset(
            **kwargs,
            rgb=self.model.input.shape[3] in (3, 4),
        )
        self.model.fit(
            **self.train_kwargs,
            callbacks=self.callbacks(),
            verbose=verbose,
        )
