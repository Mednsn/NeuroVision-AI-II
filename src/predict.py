
from pathlib import Path

import cv2
import numpy as np
import mlflow.keras

from src.config import CLASS_NAMES, IMAGE_SIZE, MLFLOW_MODEL_NAME


def load_champion_model():
    """Charge le modèle actuellement désigné comme champion."""

    model_uri = f"models:/{MLFLOW_MODEL_NAME}@champion"

    try:
        model = mlflow.keras.load_model(model_uri)
        return model

    except Exception as error:
        raise RuntimeError(
            "Impossible de charger le champion depuis MLflow. "
            "Vérifie le serveur MLflow et l'alias champion."
        ) from error


def preprocess_image(image_path):
    """Lit une image, la convertit en RGB et la redimensionne."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    # Lecture de l'image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "Image invalide ou impossible à lire."
        )

    # Conversion BGR vers RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensionnement
    image = cv2.resize(image, IMAGE_SIZE)

    # Le modèle attend un batch, même pour une seule image
    image = np.expand_dims(image, axis=0)

    return image


def predict_image(image_path, model=None):
    """
    Prédit la classe d'une image avec le champion MLflow.

    Retourne la classe prédite et les probabilités.
    """

    # Charger le champion si aucun modèle n'a été fourni
    if model is None:
        model = load_champion_model()

    # Préparer l'image avec le même format que l'entraînement
    image = preprocess_image(image_path)

    # Obtenir les probabilités des 4 classes
    probabilities = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    results = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }

    return results


if __name__ == "__main__":
    # Remplacer par le chemin d'une image de test
    image_path = "data/raw/glioma/example.jpg"

    try:
        result = predict_image(image_path)

        print("Classe prédite :", result["predicted_class"])
        print("Confiance :", f"{result['confidence']:.2%}")

        for class_name, probability in result["probabilities"].items():
            print(f"{class_name} : {probability:.2%}")

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print("Erreur :", error)
