import tensorflow as tf


def build_model(height, width, channels, classes, b_boxes, grid_size):
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(height, width, channels),
        classes=classes
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = tf.keras.layers.Conv2D(
        filters=b_boxes * (5 + classes),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer="lecun_normal"
    )(base_model.output)

    x = tf.keras.layers.Conv2D(
        filters=b_boxes * (5 + classes),
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer="lecun_normal"
    )(x)

    x = tf.keras.layers.Reshape(
        (grid_size, grid_size, b_boxes, 5 + classes)
    )(x)

    return tf.keras.Model(base_model.input, x)
