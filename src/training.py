"""Training / evaluation / plotting functions, all tracked with MLflow."""
import json
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src import config
from src import data as D
from src import models as M


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def get_callbacks(ckpt_path, patience=6, lr_patience=2):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss",
                                        save_best_only=True, mode="min", verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=lr_patience, min_lr=1e-5, verbose=1),
    ]


class MlflowEpochLogger(keras.callbacks.Callback):
    """Log every metric at each epoch (with an optional step offset)."""
    def __init__(self, offset=0, prefix=""):
        super().__init__()
        self.offset, self.prefix = offset, prefix

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            mlflow.log_metric(f"{self.prefix}{k}", float(v), step=self.offset + epoch)


def plot_history(history: dict, title="Training") -> plt.Figure:
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, name in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(epochs, history[key], label=f"Train {name}")
        ax.plot(epochs, history[f"val_{key}"], label=f"Validation {name}")
        ax.set(title=f"{name} - {title}", xlabel="Epoch", ylabel=name)
        ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def evaluate_model(model, X_test, y_test_enc, class_names):
    """Return metrics dict, classification report text and confusion-matrix figure."""
    loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    labels = np.arange(len(class_names))
    report = classification_report(y_test_enc, y_pred, labels=labels,
                                   target_names=class_names, zero_division=0)
    rep_dict = classification_report(y_test_enc, y_pred, labels=labels,
                                     target_names=class_names, zero_division=0,
                                     output_dict=True)
    cm = confusion_matrix(y_test_enc, y_pred, labels=labels)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    metrics = {"test_loss": loss, "test_accuracy": acc,
               "test_macro_f1": rep_dict["macro avg"]["f1-score"]}
    return metrics, report, fig


def log_model_artifacts(model, class_names, name):
    """Save the .keras model + class_names.json locally and as MLflow artifacts."""
    path = config.MODELS_DIR / f"{name}.keras"
    model.save(path)
    (config.MODELS_DIR / "class_names.json").write_text(json.dumps(list(class_names)))
    mlflow.log_artifact(str(path), artifact_path="model")
    mlflow.log_artifact(str(config.MODELS_DIR / "class_names.json"), artifact_path="model")
    return path


def _log_eval(model, X_test, y_test_enc, class_names, history, title):
    mlflow.log_figure(plot_history(history, title), "learning_curves.png")
    metrics, report, cm_fig = evaluate_model(model, X_test, y_test_enc, class_names)
    mlflow.log_metrics(metrics)
    mlflow.log_text(report, "classification_report.txt")
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    plt.close("all")
    return metrics


def prepare_data(data_dir=None):
    data_dir = data_dir or D.download_dataset()
    df = D.list_images(data_dir)
    X, y = D.load_dataset(df)
    X_train, X_val, X_test, y_train, y_val, y_test = D.split_data(X, y)
    enc, ytr, yv, yte = D.encode_labels(y, y_train, y_val, y_test)
    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=ytr, y_val=yv, y_test=yte, class_names=list(enc.classes_))


def train_cnn(d, epochs=15, batch_size=32, lr=1e-3):
    setup_mlflow()
    with mlflow.start_run(run_name="cnn") as run:
        mlflow.log_params({"model": "cnn", "epochs": epochs, "batch_size": batch_size,
                           "lr": lr, "img_size": config.IMG_SIZE[0]})
        model = M.build_cnn(len(d["class_names"]), lr)
        t0 = time.time()
        h = model.fit(D.make_tf_dataset(d["X_train"], d["y_train"], batch_size, True),
                      validation_data=D.make_tf_dataset(d["X_val"], d["y_val"], batch_size),
                      epochs=epochs,
                      callbacks=get_callbacks(config.MODELS_DIR / "cnn_best.keras")
                                + [MlflowEpochLogger()])
        mlflow.log_metric("training_minutes", (time.time() - t0) / 60)
        _log_eval(model, d["X_test"], d["y_test"], d["class_names"], h.history, "CNN")
        log_model_artifacts(model, d["class_names"], "cnn_final")
        return run.info.run_id, model


def train_mobilenet(d, epochs=15, ft_epochs=10, batch_size=32, lr=1e-3,
                    ft_lr=1e-5, n_unfreeze=30, finetune=True):
    """Phase 1: transfer learning (frozen base). Phase 2: fine-tuning."""
    setup_mlflow()
    with mlflow.start_run(run_name="mobilenetv2") as run:
        mlflow.log_params({"model": "mobilenetv2", "epochs": epochs, "ft_epochs": ft_epochs,
                           "batch_size": batch_size, "lr": lr, "ft_lr": ft_lr,
                           "n_unfreeze": n_unfreeze, "finetune": finetune})
        tr = D.make_tf_dataset(d["X_train"], d["y_train"], batch_size, True)
        va = D.make_tf_dataset(d["X_val"], d["y_val"], batch_size)
        model = M.build_mobilenet(len(d["class_names"]), lr)

        t0 = time.time()
        h1 = model.fit(tr, validation_data=va, epochs=epochs,
                       callbacks=get_callbacks(config.MODELS_DIR / "mobilenet_best.keras", 4)
                                 + [MlflowEpochLogger(prefix="tl_")])
        mlflow.log_metric("tl_minutes", (time.time() - t0) / 60)
        tl_loss, tl_acc = model.evaluate(d["X_val"], d["y_val"], verbose=0)
        mlflow.log_metrics({"tl_val_loss_final": tl_loss, "tl_val_accuracy_final": tl_acc})
        history, title = h1.history, "Transfer learning"

        if finetune:
            M.unfreeze_for_finetuning(model, n_unfreeze, ft_lr)
            t1 = time.time()
            h2 = model.fit(tr, validation_data=va, epochs=ft_epochs,
                           callbacks=get_callbacks(config.MODELS_DIR / "mobilenet_best.keras", 3)
                                     + [MlflowEpochLogger(prefix="ft_")])
            mlflow.log_metric("ft_minutes", (time.time() - t1) / 60)
            history, title = h2.history, "Fine-tuning"
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            for a, k in zip(ax, ["val_accuracy", "val_loss"]):
                a.plot(h1.history[k], label="Transfer learning")
                a.plot(h2.history[k], label="Fine-tuning")
                a.set_title(k); a.legend(); a.grid(True)
            mlflow.log_figure(fig, "phase_comparison.png")

        _log_eval(model, d["X_test"], d["y_test"], d["class_names"], history, title)
        log_model_artifacts(model, d["class_names"], "mobilenet_final")
        return run.info.run_id, model
