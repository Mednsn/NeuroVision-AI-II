
from pathlib import Path

import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)

# Extensions d'images autorisées dans le brief
ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}


def load_images(data_dir=DATA_DIR):
    """
    Charge les images et leurs labels depuis les dossiers de classes.

    Retourne :
        X : images RGB de taille 224 x 224 x 3
        y : labels numériques
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dossier dataset introuvable : {data_dir}"
        )

    images = []
    labels = []

    # Chaque dossier représente une classe
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier de classe introuvable : {class_dir}"
            )

        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in ALLOWED_EXTENSIONS
        )

        print(f"{class_name} : {len(image_paths)} fichiers trouvés")

        for image_path in image_paths:
            try:
                # OpenCV lit les images en BGR
                image = cv2.imread(str(image_path))

                # Ignorer les images illisibles
                if image is None:
                    print(f"Image illisible ignorée : {image_path}")
                    continue

                # Conversion BGR vers RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Redimensionnement pour MobileNetV2
                image = cv2.resize(
                    image,
                    IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA,
                )

                # Conserver les pixels en uint8, entre 0 et 255
                images.append(image)
                labels.append(class_index)

            except (cv2.error, OSError, ValueError) as error:
                print(f"Image ignorée : {image_path} | {error}")

    if not images:
        raise ValueError(
            "Aucune image valide trouvée. Vérifie le dataset."
        )

    # Les images et les labels gardent le même ordre
    X = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels, dtype=np.int32)

    print("\nDimensions des images :", X.shape)
    print("Dimensions des labels :", y.shape)

    return X, y


def split_dataset(X, y):
    """
    Crée les ensembles train, validation et test.

    Répartition finale :
        64 % train
        16 % validation
        20 % test
    """
    # Séparer le test une seule fois
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Adapter la proportion validation au dataset restant
    validation_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print("\nRépartition du dataset :")
    print("Train :", X_train.shape, y_train.shape)
    print("Val   :", X_val.shape, y_val.shape)
    print("Test  :", X_test.shape, y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_split_dataset(data_dir=DATA_DIR):
    """Fonction principale pour charger et diviser le dataset."""
    X, y = load_images(data_dir)

    return split_dataset(X, y)
