"""Implementation of U-Net in Tensorflow v2."""
# Silence verbose logging in Tensorflow
from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow.keras import (
    Model,
    activations,
    callbacks,
    initializers,
    layers,
)

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 4

inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
normalized_inputs = layers.Lambda(lambda x: x / 255)(inputs)


def encoder(previous_layer, filters, dropout_rate):
    """Return encoder layer for U-Net model."""
    convolution_layer = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        activation=activations.relu,
        kernel_initializer=initializers.he_normal(),
        padding="same",
    )(previous_layer)
    convolution_layer = layers.BatchNormalization()(convolution_layer)
    convolution_layer = layers.Dropout(dropout_rate)(convolution_layer)
    convolution_layer = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        activation=activations.relu,
        kernel_initializer=initializers.he_normal(),
        padding="same",
    )(convolution_layer)
    convolution_layer = layers.BatchNormalization()(convolution_layer)
    pool_layer = layers.MaxPool2D(pool_size=(2, 2))(convolution_layer)
    return convolution_layer, pool_layer


encoder_1, pool_1 = encoder(previous_layer=normalized_inputs, filters=32, dropout_rate=0.1)
encoder_2, pool_2 = encoder(previous_layer=pool_1, filters=64, dropout_rate=0.1)
encoder_3, pool_3 = encoder(previous_layer=pool_2, filters=128, dropout_rate=0.2)
encoder_4, pool_4 = encoder(previous_layer=pool_3, filters=256, dropout_rate=0.2)
middle_layer, _ = encoder(previous_layer=pool_4, filters=512, dropout_rate=0.3)


def decoder(*, previous_layer, encoder, dropout_rate):
    """Return decoder layer for U-Net model."""
    unconvolution_layer = layers.Conv2DTranspose(
        filters=encoder.shape[3] // 2,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
    )(previous_layer)
    unconvolution_layer = layers.concatenate([unconvolution_layer, encoder])
    unconvolution_layer = layers.Conv2D(
        filters=encoder.shape[3],
        kernel_size=(3, 3),
        activation=activations.relu,
        kernel_initializer=initializers.he_normal(),
        padding="same"
    )(unconvolution_layer)
    unconvolution_layer = layers.BatchNormalization()(unconvolution_layer)
    unconvolution_layer = layers.Dropout(dropout_rate)(unconvolution_layer)
    unconvolution_layer = layers.Conv2D(
        filters=encoder.shape[3],
        kernel_size=(3, 3),
        activation=activations.relu,
        kernel_initializer=initializers.he_normal(),
        padding="same"
    )(unconvolution_layer)
    return layers.BatchNormalization()(unconvolution_layer)


decoder_4 = decoder(previous_layer=middle_layer, encoder=encoder_4, dropout_rate=0.2)
decoder_3 = decoder(previous_layer=decoder_4, encoder=encoder_3, dropout_rate=0.2)
decoder_2 = decoder(previous_layer=decoder_3, encoder=encoder_2, dropout_rate=0.1)
decoder_1 = decoder(previous_layer=decoder_2, encoder=encoder_1, dropout_rate=0.1)


outputs = layers.Conv2D(
    filters=1,
    kernel_size=(1, 1),
    activation=activations.sigmoid,
)(decoder_1)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.summary()

early_stopper = callbacks.EarlyStopping(patience=15, verbose=1)
checkpointer = callbacks.ModelCheckpoint("model_unet_checkpoint.h5", verbose=1, save_best_only=True)
