"""Model builders: custom CNN and MobileNetV2 (transfer learning / fine-tuning)."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from src import config


def build_augmentation():
    return keras.Sequential([
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
    ], name="data_augmentation")


def build_cnn(num_classes=4, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(*config.IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),
        build_augmentation(),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mobilenet(num_classes=4, lr=1e-3, dropout=0.2):
    """MobileNetV2 with frozen base + new classification head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=(*config.IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_for_finetuning(model, n_last_layers=30, lr=1e-5):
    """Unfreeze the last n layers of the base (BatchNorm stays frozen) and recompile."""
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_last_layers]:
        layer.trainable = False
    for layer in base.layers[-n_last_layers:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
