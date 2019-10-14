from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def get_callbacks(
    model_name: str,
    early_stopping: bool = False,
):
    cache_path = Path(".cache/models")
    model_cache_path = cache_path / model_name

    model_cache_path.mkdir(parents=True, exist_ok=True)

    save_best_val_loss = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filepath=str(model_cache_path / "val_loss.ckpt"),
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    save_best_val_iou = ModelCheckpoint(
        monitor="val_mean_io_u",
        mode="max",
        filepath=str(model_cache_path / "val_mean_io_u.ckpt"),
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    save_latest = ModelCheckpoint(
        filepath=str(model_cache_path / "latest.ckpt"),
        save_weights_only=True,
        verbose=0,
    )
    callbacks = [save_latest, save_best_val_loss, save_best_val_iou]

    if early_stopping:
        earlystop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=5,
            verbose=1,
        )
        callbacks.append(earlystop_callback)

    return callbacks
