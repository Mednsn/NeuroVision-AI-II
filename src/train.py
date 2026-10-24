
import time
import tensorflow as tf
from src.mlflow_utils import log_training_run, register_and_promote
from src.config import (
    MODELS_DIR,
    BATCH_SIZE,
    EPOCHS,
    EPOCHS_FINE_TUNING,
    LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    FINE_TUNE_LAYERS,
)
from src.data_loader import load_and_split_dataset
from src.model import build_model, compile_model, unfreeze_last_layers


def make_dataset(X, y, training=False):
    """Crée un pipeline tf.data pour accélérer le chargement."""

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(X), 5000),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_callbacks(checkpoint_path, backup_path):
    """Prépare les callbacks de l'entraînement."""

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_path),
        ),
    ]


def train_pipeline():
    """Entraîne MobileNetV2 en transfer learning puis fine-tuning."""

    # 1. Charger et séparer les données
    X_train, X_val, X_test, y_train, y_val, y_test = (
        load_and_split_dataset()
    )

    # 2. Créer les datasets TensorFlow
    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

    # 3. Construire le modèle
    model, base_model = build_model()
    compile_model(model, learning_rate=LEARNING_RATE)

    print("\nArchitecture initiale du modèle :")
    model.summary()

    # 4. Transfer Learning
    checkpoint_tl = MODELS_DIR / "mobilenetv2_transfer.keras"
    backup_tl = MODELS_DIR / "backup_transfer"

    start_time = time.time()

    history_tl = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(checkpoint_tl, backup_tl),
        verbose=1,
    )

    duration_tl = time.time() - start_time

    # Recharger le meilleur modèle de la première phase
    model = tf.keras.models.load_model(checkpoint_tl)

    print(f"\nDurée Transfer Learning : {duration_tl:.2f} secondes")

    # 5. Fine-tuning : récupérer la base du modèle rechargé
    base_model = model.get_layer("mobilenetv2_1.00_224")

    unfreeze_last_layers(
        base_model,
        number_of_layers=FINE_TUNE_LAYERS,
    )

    # Recompiler obligatoirement après avoir changé trainable
    compile_model(
        model,
        learning_rate=FINE_TUNE_LEARNING_RATE,
    )

    print("\nArchitecture pour le fine-tuning :")
    model.summary()

    checkpoint_ft = MODELS_DIR / "mobilenetv2_finetuned.keras"
    backup_ft = MODELS_DIR / "backup_finetuning"

    start_time_ft = time.time()

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNING,
        callbacks=get_callbacks(checkpoint_ft, backup_ft),
        verbose=1,
    )

    duration_ft = time.time() - start_time_ft

    # Charger le meilleur modèle sauvegardé après fine-tuning
    best_model = tf.keras.models.load_model(checkpoint_ft)

    print(f"\nDurée Fine-tuning : {duration_ft:.2f} secondes")

    # 6. Comparer les performances sur validation
    tl_loss, tl_accuracy = tf.keras.models.load_model(
        checkpoint_tl
    ).evaluate(val_ds, verbose=0)

    ft_loss, ft_accuracy = best_model.evaluate(
        val_ds,
        verbose=0,
    )

    print("\n--- Comparaison sur validation ---")
    print(f"Transfer Learning - Loss : {tl_loss:.4f}")
    print(f"Transfer Learning - Accuracy : {tl_accuracy:.4f}")
    print(f"Fine-tuning - Loss : {ft_loss:.4f}")
    print(f"Fine-tuning - Accuracy : {ft_accuracy:.4f}")

    print("\nDurée Transfer Learning :", round(duration_tl, 2), "s")
    print("Durée Fine-tuning :", round(duration_ft, 2), "s")


        # Enregistrer l'expérience Transfer Learning
    tl_model = tf.keras.models.load_model(checkpoint_tl)
    
    log_training_run(
        model=tl_model,
        params={
            "phase": "transfer_learning",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
        metrics={
            "validation_accuracy": float(tl_accuracy),
            "validation_loss": float(tl_loss),
            "training_duration_seconds": float(duration_tl),
        },
        run_name="MobileNetV2_Transfer_Learning",
    )
    
    # Enregistrer l'expérience Fine-tuning
    log_training_run(
        model=best_model,
        params={
            "phase": "fine_tuning",
            "learning_rate": FINE_TUNE_LEARNING_RATE,
            "fine_tune_layers": FINE_TUNE_LAYERS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS_FINE_TUNING,
        },
        metrics={
            "validation_accuracy": float(ft_accuracy),
            "validation_loss": float(ft_loss),
            "training_duration_seconds": float(duration_ft),
        },
        run_name="MobileNetV2_Fine_Tuning",
    )
    
    # Promouvoir le candidat uniquement s'il est meilleur
    register_and_promote(
        model=best_model,
        validation_accuracy=float(ft_accuracy),
    )
    # 7. Retourner le modèle et les données du test
    # Le test n'est pas utilisé pour sélectionner le modèle.
    return {
        "model": best_model,
        "history_tl": history_tl,
        "history_ft": history_ft,
        "duration_tl": duration_tl,
        "duration_ft": duration_ft,
        "X_test": X_test,
        "y_test": y_test,
        "test_ds": test_ds,
        "val_results": {
            "transfer_learning_loss": float(tl_loss),
            "transfer_learning_accuracy": float(tl_accuracy),
            "fine_tuning_loss": float(ft_loss),
            "fine_tuning_accuracy": float(ft_accuracy),
        },
    }


if __name__ == "__main__":
    results = train_pipeline()
