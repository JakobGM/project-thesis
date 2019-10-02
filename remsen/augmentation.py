from typing import Tuple

import tensorflow as tf


@tf.function
def rotate(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # Rotate 0, 90, 180, or 270 degrees
    rotation = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

    x = tf.image.rot90(x, rotation)
    y = tf.image.rot90(y, rotation)

    return x, y


@tf.function
def flip(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)

    return x, y


@tf.function
def flip_and_rotate(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x, y = flip(x=x, y=y)
    x, y = rotate(x=x, y=y)
    return x, y
