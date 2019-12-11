import tensorflow as tf


@tf.function
def iou_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth=100):
    """Calculate the soft Jaccard loss."""
    y_true = tf.dtypes.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + smooth) / (sum - intersection + smooth)
    return tf.reduce_sum((1 - jac) * smooth)


@tf.function
def dice_loss(y_pred, y_true, smooth=1e-5):
    """
    Calculate the soft dice loss.

    Source: https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html
    """
    y_true = tf.dtypes.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_pred * y_true, axis=(1, 2, 3))
    sum = (
        tf.reduce_sum(y_pred, axis=(1, 2, 3))
        + tf.reduce_sum(y_true, axis=(1, 2, 3))
    )
    dice = (2. * intersection + smooth) / (sum + smooth)
    dice = tf.reduce_mean(dice)
    return dice
