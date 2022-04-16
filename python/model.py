import tensorflow as tf


class YOLOModel:

    def __init__(self, height, width, channels, classes, b_boxes, grid, batch):
        self.height = height
        self.width = width
        self.channels = channels
        self.classes = classes
        self.b_boxes = b_boxes
        self.grid_size = grid
        self.batch = batch

        self.model = None

    def build(self):
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(self.height, self.width, self.channels),
            classes=self.classes
        )

        for layer in base_model.layers:
            layer.trainable = False

        x = tf.keras.layers.Conv2D(
            filters=self.b_boxes * (5 + self.classes),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer="lecun_normal"
        )(base_model.output)

        x = tf.keras.layers.Conv2D(
            filters=self.b_boxes * (5 + self.classes),
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer="lecun_normal"
        )(x)

        x = tf.keras.layers.Reshape(
            (self.grid_size, self.grid_size, self.b_boxes, 5 + self.classes)
        )(x)

        self.model = tf.keras.Model(base_model.input, x)

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=2e-4, momentum=0.9),
            loss=yolo_loss
        )

    def train(self, dataset, epochs):
        self.model.fit(
            dataset,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir="logs/train"),
                tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
            ]
        )

    def save(self):
        print("Saving model...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        open("model.tflite", "wb").write(tflite_model)


def yolo_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]

    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(4), [4]), (1, 4, 4, 1, 1)), dtype=tf.float32)

    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [32, 1, 1, 1, 1])
    conf_mask = tf.zeros(mask_shape)

    """
    Adjust prediction
    """
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    # adjust w and h
    pred_box_wh = tf.sigmoid(y_pred[..., 2:4])

    # adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    # adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust ground truth
    """
    # adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

    # adjust w and h
    true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

    # adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    # adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Determine the masks
    """
    # coordinate mask: simply the position of the ground truth boxes (the predictors)

    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1)

    # confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_wh = tf.expand_dims(true_box_wh, 4)

    true_wh_half = true_wh / 2.
    true_mins = true_wh_half
    true_maxes = true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.cast(best_ious < 0.6, dtype=tf.float32) * (1 - y_true[..., 4]) * 0.5

    # penalize the confidence of the boxes, which are responsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * 0.5

    class_wt = tf.ones(100)
    class_mask = y_true[..., 4] * tf.gather(class_wt, true_box_class) * 0.5

    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32))
    nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, dtype=tf.float32))
    nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class * 10

    return loss
