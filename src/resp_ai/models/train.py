from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from resp_ai.config import load_audio_config, load_train_config, load_yaml
from resp_ai.features.dataset import (
    build_class_augmentation_profile,
    infer_input_shape,
    load_split_dataframe,
    make_tf_dataset,
)
from resp_ai.labels import CLASS_NAMES
from resp_ai.models.catalog import build_model
from resp_ai.paths import project_root_from_config, resolve_project_path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[str], out_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    return y_true, y_pred


def build_class_weight_map(train_frame: pd.DataFrame, train_config) -> dict[int, float] | None:
    strategy = str(train_config.class_weight_strategy).strip().lower()
    if strategy in {"none", "off", "disabled"}:
        return None

    label_counts = train_frame["label_index"].value_counts().sort_index()
    if strategy == "balanced":
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_frame["label_index"]),
            y=train_frame["label_index"],
        )
        class_weight_map = {int(index): float(weight) for index, weight in enumerate(class_weights)}
    else:
        max_count = float(label_counts.max())
        raw_weights = {int(label): float(np.sqrt(max_count / float(count))) for label, count in label_counts.items()}

        if strategy == "sqrt_balanced":
            class_weight_map = raw_weights
        else:
            mean_weight = float(np.mean(list(raw_weights.values())))
            class_weight_map = {
                int(label): float(np.clip(weight / mean_weight, train_config.class_weight_min, train_config.class_weight_max))
                for label, weight in raw_weights.items()
            }

    for label_name, multiplier in train_config.class_weight_overrides.items():
        if label_name in CLASS_NAMES:
            label_index = CLASS_NAMES.index(label_name)
            if label_index in class_weight_map:
                class_weight_map[label_index] = float(class_weight_map[label_index] * float(multiplier))

    return class_weight_map


def build_loss(train_config) -> tf.keras.losses.Loss:
    loss_name = str(train_config.loss_name).strip().lower()
    if loss_name in {"crossentropy", "categorical_crossentropy", "ce"}:
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=train_config.label_smoothing)

    if loss_name in {"focal", "categorical_focal", "focal_loss"}:
        gamma = float(train_config.focal_gamma)

        def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            focal_factor = tf.pow(1.0 - y_pred, gamma)
            return tf.reduce_sum(focal_factor * cross_entropy, axis=-1)

        return focal_loss

    raise ValueError(f"Unsupported loss_name: {train_config.loss_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train respiratory sound classifier.")
    parser.add_argument("--config", required=True, help="Path to training config yaml.")
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    audio_config = load_audio_config(raw_config)
    train_config = load_train_config(raw_config)
    project_root = project_root_from_config(args.config)
    data_root = resolve_project_path(project_root, train_config.metadata_root)
    models_root = resolve_project_path(project_root, train_config.output_root)

    splits_root = data_root / "splits"
    train_frame = load_split_dataframe(splits_root / "train.csv")
    val_frame = load_split_dataframe(splits_root / "val.csv")

    set_global_seed(train_config.seed)
    input_shape = infer_input_shape(audio_config)
    augmentation_profile = build_class_augmentation_profile(train_frame)

    train_ds = make_tf_dataset(
        train_frame,
        audio_config,
        train_config.batch_size,
        training=True,
        seed=train_config.seed,
        augmentation_profile=augmentation_profile,
        apply_online_augmentation=train_config.online_augmentation,
    )
    val_ds = make_tf_dataset(val_frame, audio_config, train_config.batch_size, training=False, seed=train_config.seed)

    model = build_model(train_config.architecture, input_shape, len(CLASS_NAMES), train_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_config.learning_rate)
    loss = build_loss(train_config)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    class_weight_map = build_class_weight_map(train_frame, train_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = models_root / f"{train_config.model_name}_{train_config.architecture}_{timestamp}"
    latest_dir = models_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_config.epochs,
        class_weight=class_weight_map,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(run_dir / "last_model.keras")

    y_true, y_pred = evaluate_dataset(model, val_ds)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    metrics = {
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "best_val_loss": float(min(history.history.get("val_loss", [0.0]))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": report,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with (run_dir / "class_names.json").open("w", encoding="utf-8") as handle:
        json.dump(CLASS_NAMES, handle, indent=2)

    with (run_dir / "augmentation_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                CLASS_NAMES[index]: {
                    "unique_samples": int(profile["unique_samples"]),
                    "strength": float(profile["strength"]),
                    "num_ops": int(profile["num_ops"]),
                }
                for index, profile in augmentation_profile.items()
            },
            handle,
            indent=2,
        )

    with (run_dir / "training_options.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "class_weight_strategy": train_config.class_weight_strategy,
                "class_weight_min": train_config.class_weight_min,
                "class_weight_max": train_config.class_weight_max,
                "class_weight_overrides": train_config.class_weight_overrides,
                "online_augmentation": train_config.online_augmentation,
                "loss_name": train_config.loss_name,
                "focal_gamma": train_config.focal_gamma,
                "class_weight_map": class_weight_map,
            },
            handle,
            indent=2,
        )

    pd.DataFrame(history.history).to_csv(run_dir / "history_full.csv", index=False)
    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, run_dir / "val_confusion_matrix.png")

    shutil.copy2(run_dir / "best_model.keras", latest_dir / "best_model.keras")
    shutil.copy2(run_dir / "metrics.json", latest_dir / "metrics.json")
    shutil.copy2(run_dir / "class_names.json", latest_dir / "class_names.json")
    shutil.copy2(run_dir / "augmentation_profile.json", latest_dir / "augmentation_profile.json")

    print("Training complete.")
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
