
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dossier qui contient les 4 classes d'images IRM
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Dossier pour sauvegarder les modèles et les résultats
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Les classes et leur ordre restent identiques partout
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

NUM_CLASSES = len(CLASS_NAMES)

# MobileNetV2 attend des images RGB de taille 224 x 224
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)

# Paramètres d'entraînement
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

BATCH_SIZE = 32
EPOCHS = 10
EPOCHS_FINE_TUNING = 10

# Paramètres du fine-tuning
FINE_TUNE_LAYERS = 30
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00001

# MLflow
MLFLOW_EXPERIMENT_NAME = "NeuroVision_AI"
MLFLOW_MODEL_NAME = "NeuroVisionMobileNetV2"
