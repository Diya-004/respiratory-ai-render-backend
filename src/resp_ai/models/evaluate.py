from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from resp_ai.config import load_audio_config, load_train_config, load_yaml
from resp_ai.features.audio import extract_window_batch_from_path
from resp_ai.features.dataset import load_split_dataframe, make_tf_dataset
from resp_ai.labels import CLASS_NAMES
from resp_ai.paths import project_root_from_config, resolve_project_path


def evaluate_clip_level(
    model: tf.keras.Model,
    split_frame,
    dataset: tf.data.Dataset,
) -> tuple[list[int], list[int], list[dict]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict] = []

    row_iter = iter(split_frame.to_dict(orient="records"))
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        true_indices = np.argmax(batch_y.numpy(), axis=1).tolist()
        pred_indices = np.argmax(probs, axis=1).tolist()
        y_true.extend(true_indices)
        y_pred.extend(pred_indices)

        for true_index, pred_index, prob_row in zip(true_indices, pred_indices, probs):
            row = next(row_iter)
            prediction_rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "label": row.get("label"),
                    "true_class": CLASS_NAMES[true_index],
                    "predicted_class": CLASS_NAMES[pred_index],
                    "windows_used": 1,
                    "mode": "clip",
                    **{f"prob_{class_name}": float(prob) for class_name, prob in zip(CLASS_NAMES, prob_row)},
                }
            )

    return y_true, y_pred, prediction_rows


def evaluate_recording_level(
    model: tf.keras.Model,
    split_frame,
    audio_config,
    overlap: float,
    max_windows: int,
) -> tuple[list[int], list[int], list[dict]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict] = []

    for row in split_frame.to_dict(orient="records"):
        batch, window_metadata = extract_window_batch_from_path(
            row["file_path"],
            audio_config,
            overlap=overlap,
            max_windows=max_windows,
        )
        probs = model.predict(batch, verbose=0)
        mean_probs = probs.mean(axis=0)
        pred_index = int(np.argmax(mean_probs))
        true_index = int(row["label_index"])

        y_true.append(true_index)
        y_pred.append(pred_index)
        prediction_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "label": row.get("label"),
                "true_class": CLASS_NAMES[true_index],
                "predicted_class": CLASS_NAMES[pred_index],
                "windows_used": len(window_metadata),
                "mode": "recording",
                **{f"prob_{class_name}": float(prob) for class_name, prob in zip(CLASS_NAMES, mean_probs)},
            }
        )

    return y_true, y_pred, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved respiratory sound model.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--model-path", required=True, help="Path to trained Keras model.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument("--mode", default="recording", choices=["clip", "recording"], help="Whether to evaluate one exported clip per sample or aggregate multiple windows from the full recording.")
    parser.add_argument("--window-overlap", type=float, default=0.5, help="Window overlap ratio for recording-level evaluation.")
    parser.add_argument("--max-windows", type=int, default=5, help="Maximum number of windows to evaluate per recording.")
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    audio_config = load_audio_config(raw_config)
    train_config = load_train_config(raw_config)
    project_root = project_root_from_config(args.config)
    data_root = resolve_project_path(project_root, train_config.metadata_root)
    split_frame = load_split_dataframe(data_root / "splits" / f"{args.split}.csv")

    model = tf.keras.models.load_model(args.model_path, compile=False)

    if args.mode == "clip":
        dataset = make_tf_dataset(split_frame, audio_config, train_config.batch_size, training=False, seed=train_config.seed)
        y_true, y_pred, prediction_rows = evaluate_clip_level(model, split_frame, dataset)
    else:
        y_true, y_pred, prediction_rows = evaluate_recording_level(
            model,
            split_frame,
            audio_config,
            overlap=args.window_overlap,
            max_windows=args.max_windows,
        )

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mode": args.mode,
        "window_overlap": float(args.window_overlap),
        "max_windows": int(args.max_windows),
        "classification_report": report,
    }

    output_dir = Path(args.model_path).resolve().parent / f"{args.split}_evaluation_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(prediction_rows, handle, indent=2)

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{args.split.title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    print(json.dumps(metrics, indent=2))
    print("Saved evaluation to:", output_dir)


if __name__ == "__main__":
    main()
