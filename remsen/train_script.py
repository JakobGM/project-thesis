from remsen.data import Dataset
from remsen.training import get_callbacks

from notebooks.unet import model

ds = Dataset()
train, val, test = ds.tf_dataset(augment=True)
augmentation_callbacks = get_callbacks("long_train")

model.load_weights(".cache/models/long_train/latest.ckpt")
EPOCHS = 1000
for _ in range(1, EPOCHS + 1):
    try:
        model.fit(
            train,
            validation_data=val,
            callbacks=augmentation_callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        print("Model training interrupted")
        break
