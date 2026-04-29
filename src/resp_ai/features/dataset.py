from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from resp_ai.config import AudioConfig
from resp_ai.labels import CLASS_TO_INDEX, CLASS_NAMES
from resp_ai.features.audio import preprocess_path


def _remap_dataset_path(path_value: str, csv_path: str | Path) -> str:
    if pd.isna(path_value):
        return path_value
    path = Path(str(path_value))
    if path.exists():
        return str(path)

    dataset_root = Path(csv_path).expanduser().resolve().parents[1]
    if dataset_root.name in path.parts:
        dataset_index = path.parts.index(dataset_root.name)
        remapped = dataset_root.joinpath(*path.parts[dataset_index + 1 :])
        return str(remapped)

    return str(path)


def load_split_dataframe(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    for column in ["file_path", "processed_path", "pool_path"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: _remap_dataset_path(value, csv_path))
    if "processed_path" in frame.columns:
        frame["active_path"] = frame["processed_path"].fillna(frame["file_path"])
    else:
        frame["active_path"] = frame["file_path"]
    frame["label_index"] = frame["label"].map(CLASS_TO_INDEX)
    return frame


def infer_input_shape(config: AudioConfig) -> Tuple[int, int, int]:
    dummy = np.zeros(config.target_samples, dtype=np.float32)
    feature = preprocess_path_from_signal(dummy, config)
    return feature.shape


def preprocess_path_from_signal(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    from resp_ai.features.audio import compute_logmel_image

    return compute_logmel_image(signal, config)


def build_class_augmentation_profile(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if "sample_id" in frame.columns:
        unique_counts = frame.groupby("label")["sample_id"].nunique().to_dict()
    else:
        unique_counts = frame["label"].value_counts().to_dict()

    max_count = max(unique_counts.values())
    min_count = min(unique_counts.values())
    denom = max(max_count - min_count, 1)

    profile: dict[int, dict[str, Any]] = {}
    for label_name in CLASS_NAMES:
        count = int(unique_counts.get(label_name, 0))
        rarity = (max_count - count) / denom if count else 1.0
        strength = 0.8 + (0.55 * rarity)
        num_ops = 1 + int(rarity >= 0.33) + int(rarity >= 0.66)
        profile[CLASS_TO_INDEX[label_name]] = {
            "label": label_name,
            "unique_samples": count,
            "strength": round(float(strength), 4),
            "num_ops": int(num_ops),
        }
    return profile


def make_tf_dataset(
    frame: pd.DataFrame,
    audio_config: AudioConfig,
    batch_size: int,
    training: bool,
    seed: int,
    augmentation_profile: dict[int, dict[str, Any]] | None = None,
    apply_online_augmentation: bool = True,
) -> tf.data.Dataset:
    paths = frame["active_path"].astype(str).to_numpy()
    labels = frame["label_index"].astype(np.int32).to_numpy()
    output_shape = infer_input_shape(audio_config)
    if augmentation_profile is None:
        augmentation_profile = {
            index: {"strength": 1.0, "num_ops": 1}
            for index in range(len(CLASS_NAMES))
        }
    strength_by_class = np.array(
        [float(augmentation_profile[index]["strength"]) for index in range(len(CLASS_NAMES))],
        dtype=np.float32,
    )
    ops_by_class = np.array(
        [int(augmentation_profile[index]["num_ops"]) for index in range(len(CLASS_NAMES))],
        dtype=np.int32,
    )

    def _load(path: tf.Tensor, label: tf.Tensor):
        def _py_load(path_bytes: bytes, label_value: np.int32):
            label_index = int(label_value)
            image = preprocess_path(
                path_bytes.decode("utf-8"),
                audio_config,
                training=training and apply_online_augmentation,
                augmentation_strength=float(strength_by_class[label_index]),
                augmentation_ops=int(ops_by_class[label_index]),
            )
            return image.astype(np.float32)

        image = tf.numpy_function(_py_load, [path, label], tf.float32)
        image.set_shape(output_shape)
        one_hot = tf.one_hot(label, depth=len(CLASS_NAMES))
        return image, one_hot

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(frame), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
