"""Inference helpers (no Streamlit dependency)."""
import json
from pathlib import Path

import cv2
import numpy as np

from src import config
from src import data as D


def list_local_models():
    return sorted(config.MODELS_DIR.glob("*.keras"))


def load_class_names(path=None):
    path = Path(path or config.MODELS_DIR / "class_names.json")
    return json.loads(path.read_text()) if path.exists() else sorted(config.CLASSES)


def load_model(path):
    from tensorflow import keras
    return keras.models.load_model(path)


def load_model_from_mlflow(run_id):
    """Download model + class names logged by a training run."""
    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    d = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model"))
    keras_file = next(d.glob("*.keras"))
    return load_model(keras_file), load_class_names(d / "class_names.json")


def preprocess_bytes(file_bytes: bytes) -> np.ndarray:
    """Raw uploaded bytes -> RGB 224x224 uint8 image."""
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image.")
    return D.preprocess_image(img)


def predict_image(model, image_rgb, class_names):
    probs = model.predict(image_rgb[None], verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), dict(zip(class_names, map(float, probs)))
