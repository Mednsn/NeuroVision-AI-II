
import tensorflow as tf

from tensorflow.keras import Model, layers

from src.config import (
    IMAGE_SHAPE,
    NUM_CLASSES,
    FINE_TUNE_LAYERS,
)


def build_model():
    """Construit MobileNetV2 avec une nouvelle tête de classification."""

    # Charger MobileNetV2 pré-entraîné sur ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Geler la base pendant le transfer learning
    base_model.trainable = False

    # Entrée : image RGB de 224 x 224
    inputs = layers.Input(shape=IMAGE_SHAPE, name="image")

    inputs = layers.Input(shape=IMAGE_SHAPE, name="image")

    # Augmentation légère, active uniquement pendant l'entraînement
    x = layers.RandomZoom(0.10, name="random_zoom")(x)
    x = layers.RandomRotation(0.03, name="random_rotation")(inputs)
    x = layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05,
        name="random_translation",
    )(x)  

# Prétraitement MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Extraire les caractéristiques avec la base pré-entraînée
    x = base_model(x, training=False)

    # Transformer les cartes de caractéristiques en vecteur
    x = layers.GlobalAveragePooling2D(name="global_pooling")(x)

    # Réduire le surapprentissage
    x = layers.Dropout(0.2, name="dropout")(x)

    # Prédire une probabilité pour chacune des 4 classes
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="NeuroVision_MobileNetV2",
    )

    return model, base_model


def compile_model(model, learning_rate=0.001):
    """Compile le modèle pour une classification multiclasse."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_last_layers(base_model, number_of_layers=FINE_TUNE_LAYERS):
    """Dégèle les dernières couches pour le fine-tuning."""

    # Autoriser le fine-tuning de la base
    base_model.trainable = True

    # Garder les premières couches gelées
    cutoff = max(0, len(base_model.layers) - number_of_layers)

    for layer in base_model.layers[:cutoff]:
        layer.trainable = False

    # Éviter de modifier les statistiques BatchNormalization
    for layer in base_model.layers[cutoff:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return base_model
