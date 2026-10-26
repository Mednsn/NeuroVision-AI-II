from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FILE_ID = "1PP09vNLPEsiFXFyKu4_8mC-bFJbxfflC"
ZIP_PATH = ROOT / "tumors_dataset.zip"
EXTRACT_DIR = ROOT / "extract"

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}
CLASSES = ["notumor", "pituitary", "meningioma", "glioma"]
IMG_SIZE = (224, 224)
SEED = 42

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "neurovision"
