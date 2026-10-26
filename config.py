from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset sur Vast.ai
DATASET_DIR = Path("/extract/Data")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

# Images
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
MODEL_INPUT_SIZE = (224, 224, 3)

# Dataset split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# Training
BATCH_SIZE = 32
EPOCHS = 15

# MobileNetV2
PRETRAINED_WEIGHTS = "imagenet"
TRANSFER_LEARNING_LR = 0.001
FINE_TUNING_LR = 0.00001
UNFREEZE_LAYERS = 30

# Runtime
RUNTIME_DIR = Path("/workspace/neurovision_runtime")

MODELS_DIR = RUNTIME_DIR / "models"
REPORTS_DIR = RUNTIME_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# MLflow
MLFLOW_EXPERIMENT_NAME = "NeuroVision-AI"
MLFLOW_TRACKING_URI = "file://" + str(RUNTIME_DIR / "mlruns")
MODEL_NAME = "NeuroVision-Classifier"

# Create output directories
for directory in [
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)