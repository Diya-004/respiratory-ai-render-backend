from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AudioConfig:
    sample_rate: int
    duration_seconds: float
    trim_top_db: int
    n_mels: int
    n_fft: int
    hop_length: int
    fmin: int
    fmax: int
    use_deltas: bool

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)


@dataclass
class TrainConfig:
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    label_smoothing: float
    architecture: str
    dropout: float
    dense_units: int
    freeze_backbone: bool
    output_root: str
    metadata_root: str
    model_name: str
    class_weight_strategy: str
    class_weight_min: float
    class_weight_max: float
    class_weight_overrides: dict[str, float]
    online_augmentation: bool
    loss_name: str
    focal_gamma: float


@dataclass
class AppConfig:
    host: str
    port: int
    allowed_extensions: list[str]


@dataclass
class InferenceConfig:
    window_overlap: float
    max_windows: int
    aggregation: str


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_audio_config(config: Dict[str, Any]) -> AudioConfig:
    audio = config["audio"]
    return AudioConfig(
        sample_rate=audio["sample_rate"],
        duration_seconds=audio["duration_seconds"],
        trim_top_db=audio["trim_top_db"],
        n_mels=audio["n_mels"],
        n_fft=audio["n_fft"],
        hop_length=audio["hop_length"],
        fmin=audio["fmin"],
        fmax=audio["fmax"],
        use_deltas=audio["use_deltas"],
    )


def load_train_config(config: Dict[str, Any]) -> TrainConfig:
    train = config["train"]
    paths = config["paths"]
    return TrainConfig(
        seed=train["seed"],
        batch_size=train["batch_size"],
        epochs=train["epochs"],
        learning_rate=train["learning_rate"],
        label_smoothing=train["label_smoothing"],
        architecture=train["architecture"],
        dropout=train["dropout"],
        dense_units=train["dense_units"],
        freeze_backbone=train["freeze_backbone"],
        output_root=paths["models_root"],
        metadata_root=paths["data_root"],
        model_name=train["model_name"],
        class_weight_strategy=str(train.get("class_weight_strategy", "sqrt_balanced_clipped")),
        class_weight_min=float(train.get("class_weight_min", 0.5)),
        class_weight_max=float(train.get("class_weight_max", 2.0)),
        class_weight_overrides={
            str(label): float(weight)
            for label, weight in dict(train.get("class_weight_overrides", {})).items()
        },
        online_augmentation=bool(train.get("online_augmentation", True)),
        loss_name=str(train.get("loss_name", "crossentropy")),
        focal_gamma=float(train.get("focal_gamma", 2.0)),
    )


def load_app_config(config: Dict[str, Any]) -> AppConfig:
    app = config["app"]
    return AppConfig(
        host=app["host"],
        port=app["port"],
        allowed_extensions=app["allowed_extensions"],
    )


def load_inference_config(config: Dict[str, Any]) -> InferenceConfig:
    inference = config.get("inference", {})
    return InferenceConfig(
        window_overlap=float(inference.get("window_overlap", 0.5)),
        max_windows=int(inference.get("max_windows", 5)),
        aggregation=str(inference.get("aggregation", "mean_probability")),
    )
