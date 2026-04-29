from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local smoke test for dataset loading and model forward pass.")
    parser.add_argument("--config", required=True, help="Path to training config yaml.")
    parser.add_argument("--rows", type=int, default=8, help="Number of rows to use from the training split.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the smoke test dataset.")
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    audio_config = load_audio_config(raw_config)
    train_config = load_train_config(raw_config)
    project_root = project_root_from_config(args.config)
    data_root = resolve_project_path(project_root, train_config.metadata_root)

    train_csv = data_root / "splits" / "train.csv"
    frame = load_split_dataframe(train_csv).head(args.rows)
    augmentation_profile = build_class_augmentation_profile(frame)
    dataset = make_tf_dataset(
        frame,
        audio_config,
        batch_size=args.batch_size,
        training=False,
        seed=train_config.seed,
        augmentation_profile=augmentation_profile,
    )

    batch_x, batch_y = next(iter(dataset))
    model = build_model(train_config.architecture, infer_input_shape(audio_config), len(CLASS_NAMES), train_config)
    output = model(batch_x, training=False)

    print("Smoke test OK")
    print("Config:", Path(args.config).resolve())
    print("Rows tested:", len(frame))
    print("Input batch shape:", tuple(batch_x.shape))
    print("Label batch shape:", tuple(batch_y.shape))
    print("Output batch shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
