import math
from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    directory = Path("~/dev/project-thesis/data/nuclei").expanduser()
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 4

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        assert self.directory.exists()
        self.samples = list((self.directory / "train").iterdir())
        self.num_samples = len(self.samples)

    def __len__(self):
        """Return number of batches."""
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, index):
        """Return batch."""
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
        for batch_index, img_index in enumerate(range(start_index, end_index)):
            input, target = self.image(index=img_index)
            inputs[batch_index] = input.numpy()
            targets[batch_index] = target.numpy()

        return inputs, targets

    def image(self, index):
        """Return image at location."""
        image_path = next((self.samples[index] / "images").iterdir())
        raw_image = tf.io.read_file(str(image_path))
        image_tensor = tf.image.decode_image(raw_image)
        image_tensor /= 255
        image_tensor = tf.image.resize(
            image_tensor,
            size=(256, 256),
        )

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

    def display(self, index):
        image_tensor, mask_tensor = self.image(index=index)

        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(image_tensor.numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(tf.squeeze(mask_tensor).numpy())
        plt.show()

    def generator(self):
        for inputs, targets in self:
            yield inputs, targets

    def tf_dataset(self):
        return tf.data.Dataset.from_generator(
            generator=self.generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=((256, 256), (256, 256)),
        )

    def xy(self):
        sample_size = len(self)
        x_train = np.zeros(shape=(sample_size, 256, 256, 4))
        y_train = np.zeros(shape=(sample_size, 256, 256, 1))

        for index, (inputs, target) in enumerate(self):
            x_train[index, :, :, :] = inputs.numpy()
            y_train[index, :, :, 0] = target.numpy().reshape(256, 256)

        return x_train, y_train
