import math
from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    directory = Path("/code/data/nuclei")
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3

    def __init__(self, batch_size: int, directory: Path, with_mask=True):
        self.batch_size = batch_size
        assert directory.exists()
        self.directory = directory
        self.samples = list(self.directory.iterdir())
        self.num_samples = len(self.samples)
        self.with_mask = with_mask
        self._cache = {}

    def __len__(self):
        """Return number of batches."""
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, index):
        """Return batch."""
        if index in self._cache:
            return self._cache[index]

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        inputs = np.zeros(
            shape=(
                self.batch_size,
                self.IMG_HEIGHT,
                self.IMG_WIDTH,
                self.IMG_CHANNELS,
            ),
        )
        targets = np.zeros(
            shape=(
                self.batch_size,
                self.IMG_HEIGHT,
                self.IMG_WIDTH,
                1,
            ),
        )
        if self.with_mask:
            for batch_index, img_index in enumerate(range(start_index, end_index)):
                input, target = self.image(index=img_index)
                inputs[batch_index] = input.numpy()
                targets[batch_index] = target.numpy()

            self._cache[index] = (inputs, targets)
            return inputs, targets
        else:
            for batch_index, img_index in enumerate(range(start_index, end_index)):
                input = self.image(index=img_index)
                inputs[batch_index] = input.numpy()

            self._cache[index] = inputs
            return inputs

    def image(self, index):
        """Return image at location."""
        image_path = next((self.samples[index] / "images").iterdir())
        raw_image = tf.io.read_file(str(image_path))
        image_tensor = tf.image.decode_png(raw_image, channels=self.IMG_CHANNELS)
        image_tensor /= 255
        image_tensor = tf.image.resize(
            image_tensor,
            size=(256, 256),
        )
        assert image_tensor.shape == (
            self.IMG_HEIGHT,
            self.IMG_WIDTH,
            self.IMG_CHANNELS,
        )
        if not self.with_mask:
            return image_tensor

        mask_paths = list((self.samples[index] / "masks").iterdir())
        raw_masks = [tf.io.read_file(str(mask_path)) for mask_path in mask_paths]
        mask_tensors = [tf.image.decode_image(raw_mask) for raw_mask in raw_masks]
        mask_tensor = tf.squeeze(tf.add_n(mask_tensors) / 255)
        mask_tensor = tf.expand_dims(mask_tensor, -1)
        mask_tensor = tf.image.resize(
            mask_tensor,
            size=(256, 256),
        )
        return image_tensor, mask_tensor

    def display(self, index, prediction=None):
        plt.figure(figsize=(15, 15))
        if self.with_mask:
            image_tensor, mask_tensor = self.image(index=index)
        else:
            image_tensor = self.image(index=index)
            mask_tensor = None

        input_ax = plt.subplot(131)
        input_ax.set_title("Input data")
        input_ax.imshow(image_tensor.numpy())

        mask_ax = plt.subplot(132)
        mask_ax.set_title("Train mask")
        if mask_tensor is not None:
            mask_ax.imshow(tf.squeeze(mask_tensor).numpy())
        else:
            mask_ax.imshow(np.ones_like(image_tensor.numpy()))

        prediction_ax = plt.subplot(133)
        prediction_ax.set_title("Prediction")
        if prediction is not None:
            prediction_ax.imshow(
                prediction.reshape(self.IMG_HEIGHT, self.IMG_WIDTH),
            )
        else:
            prediction_ax.imshow(np.ones_like(image_tensor.numpy()))

        print("Showing plot")
        plt.show()
