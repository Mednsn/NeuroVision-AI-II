
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.config import CLASS_NAMES, MODELS_DIR
from src.data_loader import load_and_split_dataset
from src.train import make_dataset


def evaluate_model(model, X_test, y_test):

    # Prédire les probabilités de chaque classe
    y_probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1,
    )

    # Choisir la classe ayant la probabilité la plus élevée
    y_pred = np.argmax(y_probabilities, axis=1)

    # Accuracy globale
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy : {accuracy:.4f}")

    # Rapport détaillé par classe
    print("\nRapport de classification :")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # Matrice de confusion
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title("Matrice de confusion - NeuroVision AI")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_probabilities": y_probabilities,
    }


if __name__ == "__main__":
    # Utiliser le meilleur modèle enregistré par le pipeline
    model_path = MODELS_DIR / "mobilenetv2_finetuned.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            "Modèle introuvable. Lance d'abord src/train.py."
        )

    model = tf.keras.models.load_model(model_path)

    # Recréer le même split avec la même seed
    _, _, X_test, _, _, y_test = load_and_split_dataset()

    evaluate_model(model, X_test, y_test)
