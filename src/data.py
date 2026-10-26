"""Data functions: download, index, load, split, encode."""
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config


def download_dataset(file_id=config.FILE_ID, zip_path=config.ZIP_PATH,
                     extract_dir=config.EXTRACT_DIR) -> Path:
    """Download (gdown) and extract the dataset; return the data directory."""
    import gdown
    zip_path, extract_dir = Path(zip_path), Path(extract_dir)
    if not zip_path.exists():
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    if (extract_dir / "Data").exists():
        return extract_dir / "Data"
    if (extract_dir / "tumors" / "Data").exists():
        return extract_dir / "tumors" / "Data"
    return extract_dir


def list_images(data_dir, classes=config.CLASSES) -> pd.DataFrame:
    """DataFrame(path, label) of every image found under data_dir/<class>."""
    rows = []
    for name in classes:
        for f in (Path(data_dir) / name).rglob("*"):
            if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS:
                rows.append((f, name))
    return pd.DataFrame(rows, columns=["path", "label"])


def preprocess_image(image_bgr: np.ndarray, size=config.IMG_SIZE) -> np.ndarray:
    """BGR (cv2) -> RGB resized uint8 image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, size)


def load_image(path, size=config.IMG_SIZE):
    img = cv2.imread(str(path))
    return None if img is None else preprocess_image(img, size)


def load_dataset(df: pd.DataFrame, size=config.IMG_SIZE):
    """Load all images listed in df into (X uint8, y labels), skipping unreadable files."""
    images, labels = [], []
    for path, label in zip(df["path"], df["label"]):
        img = load_image(path, size)
        if img is None:
            print("-- unreadable image:", path)
            continue
        images.append(img)
        labels.append(label)
    return np.array(images, dtype=np.uint8), np.array(labels)


def split_data(X, y, seed=config.SEED):
    """Stratified 70/15/15 split."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_labels(y_all, *label_arrays):
    """Fit a LabelEncoder on y_all, return (encoder, encoded arrays...)."""
    enc = LabelEncoder().fit(y_all)
    return (enc, *[enc.transform(a) for a in label_arrays])


def make_tf_dataset(X, y, batch_size=32, shuffle=False):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
