import tensorflow as tf


@tf.function
def iou_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth=100):
    """Calculate the soft Jaccard loss."""
    y_true = tf.dtypes.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + smooth) / (sum - intersection + smooth)
    return tf.reduce_sum((1 - jac) * smooth)
