import tensorflow as tf
from tensorflow.keras import (
    Model,
    activations,
    initializers,
    layers,
)

from remsen import losses
from remsen.metrics import iou


def encoder(previous_layer, filters, dropout_rate, batch_normalization):
    """Return encoder layer for U-Net model."""
    convolution_layer = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        activation=activations.relu,
        kernel_initializer=initializers.he_normal(),
        padding="same",
    )(previous_layer)
    if batch_normalization:
        convolution_layer = layers.BatchNormalization()(convolution_layer)
    if dropout_rate:
        convolution_layer = layers.Dropout(dropout_rate)(convolution_layer)
    convolution_layer = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        activation=activations.relu,
        kernel_initializer=initializers.he_normal(),
        padding="same",
    )(convolution_layer)
    if batch_normalization:
        convolution_layer = layers.BatchNormalization()(convolution_layer)
    pool_layer = layers.MaxPool2D(pool_size=(2, 2))(convolution_layer)
    return convolution_layer, pool_layer


def decoder(*, previous_layer, encoder, dropout_rate, batch_normalization):
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
    if batch_normalization:
        unconvolution_layer = layers.BatchNormalization()(unconvolution_layer)
    if dropout_rate:
        unconvolution_layer = layers.Dropout(dropout_rate)(unconvolution_layer)
    unconvolution_layer = layers.Conv2D(
        filters=encoder.shape[3],
        kernel_size=(3, 3),
        activation=activations.relu,
        kernel_initializer=initializers.he_normal(),
        padding="same"
    )(unconvolution_layer)
    if batch_normalization:
        return layers.BatchNormalization()(unconvolution_layer)
    else:
        return unconvolution_layer


def unet(
    img_height: int = 256,
    img_width: int = 256,
    img_channels: int = 1,
    loss: str = "binary_cross_entropy",
    dropout: bool = True,
    batch_normalization: bool = True,
) -> Model:
    inputs = layers.Input(
        shape=(img_height, img_width, img_channels),
        name="input",
        dtype=tf.float32,
    )

    # Encoder layers
    encoder_1, pool_1 = encoder(
        previous_layer=inputs,
        filters=32,
        dropout_rate=0.1 if dropout else False,
        batch_normalization=batch_normalization,
    )
    encoder_2, pool_2 = encoder(
        previous_layer=pool_1,
        filters=64,
        dropout_rate=0.1 if dropout else False,
        batch_normalization=batch_normalization,
    )
    encoder_3, pool_3 = encoder(
        previous_layer=pool_2,
        filters=128,
        dropout_rate=0.2 if dropout else False,
        batch_normalization=batch_normalization,
    )
    encoder_4, pool_4 = encoder(
        previous_layer=pool_3,
        filters=256,
        dropout_rate=0.2 if dropout else False,
        batch_normalization=batch_normalization,
    )
    middle_layer, _ = encoder(
        previous_layer=pool_4,
        filters=512,
        dropout_rate=0.3 if dropout else False,
        batch_normalization=batch_normalization,
    )

    # Decoder layers
    decoder_4 = decoder(
        previous_layer=middle_layer,
        encoder=encoder_4,
        dropout_rate=0.2 if dropout else False,
        batch_normalization=batch_normalization,
    )
    decoder_3 = decoder(
        previous_layer=decoder_4,
        encoder=encoder_3,
        dropout_rate=0.2 if dropout else False,
        batch_normalization=batch_normalization,
    )
    decoder_2 = decoder(
        previous_layer=decoder_3,
        encoder=encoder_2,
        dropout_rate=0.1 if dropout else False,
        batch_normalization=batch_normalization,
    )
    decoder_1 = decoder(
        previous_layer=decoder_2,
        encoder=encoder_1,
        dropout_rate=0.1 if dropout else False,
        batch_normalization=batch_normalization,
    )

    outputs = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        activation=activations.sigmoid,
    )(decoder_1)

    available_losses = {
        "binary_cross_entropy": "binary_crossentropy",
        "iou_loss": losses.iou_loss,
        "dice_loss": losses.dice_loss,
    }

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="adam",
        loss=available_losses[loss],
        metrics=[iou],
    )
    return model
