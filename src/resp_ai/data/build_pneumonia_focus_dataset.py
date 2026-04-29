from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from resp_ai.config import load_audio_config, load_yaml
from resp_ai.features.audio import augment_audio, fit_audio_length


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def replace_dataset_root(value: str, base_root: Path, output_root: Path) -> str:
    if pd.isna(value):
        return value
    return str(value).replace(str(base_root), str(output_root))


def rewrite_paths(frame: pd.DataFrame, *, base_root: Path, output_root: Path) -> pd.DataFrame:
    updated = frame.copy()
    for column in ["file_path", "processed_path", "pool_path"]:
        if column in updated.columns:
            updated[column] = updated[column].map(lambda value: replace_dataset_root(value, base_root, output_root))
    return updated


def copy_base_dataset(base_root: Path, output_root: Path) -> None:
    for dirname in ["audio", "metadata", "recording_splits", "train", "val", "test"]:
        src = base_root / dirname
        dst = output_root / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _round_robin_rows(frame: pd.DataFrame) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame.sort_values(["sample_id", "clip_id"]).to_dict(orient="records"):
        grouped[str(row["sample_id"])].append(row)

    sample_ids = sorted(grouped)
    ordered: list[dict] = []
    while sample_ids:
        next_ids: list[str] = []
        for sample_id in sample_ids:
            remaining = grouped[sample_id]
            if remaining:
                ordered.append(remaining.pop(0))
            if remaining:
                next_ids.append(sample_id)
        sample_ids = next_ids
    return ordered


def _make_augmented_clip(
    src_path: Path,
    dst_path: Path,
    *,
    sample_rate: int,
    target_samples: int,
    strength: float,
    num_ops: int,
    seed: int,
) -> None:
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        signal, _ = librosa.load(src_path, sr=sample_rate, mono=True)
        augmented = augment_audio(signal, sample_rate, strength=float(strength), num_ops=int(num_ops))
        prepared = fit_audio_length(augmented, target_samples).astype(np.float32)
        ensure_dir(dst_path.parent)
        sf.write(dst_path, prepared, sample_rate)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _pneumonia_aug_params(count: int, target: int) -> tuple[float, int]:
    shortage_ratio = max(target - count, 0) / max(target, 1)
    strength = 1.12 + (0.16 * shortage_ratio)
    num_ops = 3 if shortage_ratio < 0.35 else 4
    return (round(float(strength), 4), int(num_ops))


def augment_pneumonia_group(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    sample_rate: int,
    target_samples: int,
    target_count: int,
    seed: int,
    stage_name: str,
) -> tuple[pd.DataFrame, list[dict]]:
    if frame.empty or len(frame) >= target_count:
        return pd.DataFrame(), []

    train_root = output_root / "train"
    ordered_rows = _round_robin_rows(frame)
    current_count = len(frame)
    needed = target_count - current_count
    strength, num_ops = _pneumonia_aug_params(current_count, target_count)

    augmented_rows: list[dict] = []
    events: list[dict] = []

    for index in range(needed):
        base_row = ordered_rows[index % len(ordered_rows)]
        base_clip_path = Path(base_row["processed_path"])
        base_clip_id = str(base_row["clip_id"])
        source = str(base_row["source"])
        aug_clip_id = f"{base_clip_id}_{stage_name}_aug{index:03d}"
        dst_path = train_root / "Pneumonia" / f"{aug_clip_id}.wav"
        local_seed = seed + (hash((source, stage_name, base_clip_id, index)) & 0xFFFF_FFFF)

        _make_augmented_clip(
            base_clip_path,
            dst_path,
            sample_rate=sample_rate,
            target_samples=target_samples,
            strength=strength,
            num_ops=num_ops,
            seed=local_seed,
        )

        new_row = dict(base_row)
        new_row.update(
            {
                "clip_id": aug_clip_id,
                "processed_path": str(dst_path),
                "is_augmented": True,
                "augmentation_type": f"{stage_name}:pneumonia_focus:strength_{strength:.3f}:ops_{num_ops}",
                "augmentation_source_clip_id": base_clip_id,
            }
        )
        augmented_rows.append(new_row)
        events.append(
            {
                "stage": stage_name,
                "source": source,
                "base_clip_id": base_clip_id,
                "augmented_clip_id": aug_clip_id,
                "strength": strength,
                "num_ops": num_ops,
            }
        )

    return pd.DataFrame(augmented_rows), events


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pneumonia-focused train dataset on top of the gap-augmented base.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--base-data-root", required=True, help="Existing dataset root to extend")
    parser.add_argument("--output-root", required=True, help="Destination dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pneumonia-source-clips", type=int, default=350)
    parser.add_argument("--min-pneumonia-clips", type=int, default=900)
    args = parser.parse_args()

    config = load_yaml(args.config)
    audio_config = load_audio_config(config)
    base_root = Path(args.base_data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    copy_base_dataset(base_root, output_root)

    splits_root = output_root / "splits"
    ensure_dir(splits_root)

    train_frame = pd.read_csv(base_root / "splits" / "train.csv")
    val_frame = pd.read_csv(base_root / "splits" / "val.csv")
    test_frame = pd.read_csv(base_root / "splits" / "test.csv")

    train_frame = rewrite_paths(train_frame, base_root=base_root, output_root=output_root)
    val_frame = rewrite_paths(val_frame, base_root=base_root, output_root=output_root)
    test_frame = rewrite_paths(test_frame, base_root=base_root, output_root=output_root)

    if "is_augmented" not in train_frame.columns:
        train_frame["is_augmented"] = False
    if "augmentation_type" not in train_frame.columns:
        train_frame["augmentation_type"] = ""
    if "augmentation_source_clip_id" not in train_frame.columns:
        train_frame["augmentation_source_clip_id"] = ""

    augmented_frames = [train_frame]
    events: list[dict] = []
    current_train = train_frame.copy()

    pneumonia = current_train[current_train["label"] == "Pneumonia"].copy()
    for source, group in pneumonia.groupby("source", sort=True):
        if len(group) >= args.min_pneumonia_source_clips:
            continue
        augmented, stage_events = augment_pneumonia_group(
            group,
            output_root=output_root,
            sample_rate=audio_config.sample_rate,
            target_samples=audio_config.target_samples,
            target_count=args.min_pneumonia_source_clips,
            seed=args.seed,
            stage_name=f"sourcefocus_{source}",
        )
        if not augmented.empty:
            augmented_frames.append(augmented)
            events.extend(stage_events)
            current_train = pd.concat([current_train, augmented], ignore_index=True)

    pneumonia = current_train[current_train["label"] == "Pneumonia"].copy()
    augmented, stage_events = augment_pneumonia_group(
        pneumonia,
        output_root=output_root,
        sample_rate=audio_config.sample_rate,
        target_samples=audio_config.target_samples,
        target_count=args.min_pneumonia_clips,
        seed=args.seed + 2000,
        stage_name="classfocus",
    )
    if not augmented.empty:
        augmented_frames.append(augmented)
        events.extend(stage_events)
        current_train = pd.concat([current_train, augmented], ignore_index=True)

    final_train = pd.concat(augmented_frames, ignore_index=True).sort_values(
        ["label", "source", "sample_id", "clip_id"]
    ).reset_index(drop=True)
    final_train["split"] = "train"
    final_train.to_csv(splits_root / "train.csv", index=False)
    val_frame.to_csv(splits_root / "val.csv", index=False)
    test_frame.to_csv(splits_root / "test.csv", index=False)
    pd.concat([final_train, val_frame, test_frame], ignore_index=True).to_csv(splits_root / "all_splits.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "min_pneumonia_source_clips": int(args.min_pneumonia_source_clips),
        "min_pneumonia_clips": int(args.min_pneumonia_clips),
        "original_train_label_counts": train_frame["label"].value_counts().sort_index().to_dict(),
        "final_train_label_counts": final_train["label"].value_counts().sort_index().to_dict(),
        "original_pneumonia_source_counts": train_frame[train_frame["label"] == "Pneumonia"].groupby("source").size().to_dict(),
        "final_pneumonia_source_counts": final_train[final_train["label"] == "Pneumonia"].groupby("source").size().to_dict(),
        "augmented_rows": int(final_train["is_augmented"].fillna(False).sum()),
        "events": events,
    }

    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata" / "pneumonia_focus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pneumonia-focus dataset created at:", output_root)
    print("Final train label counts:")
    print(final_train["label"].value_counts().sort_index())
    print("\nFinal pneumonia source counts:")
    print(final_train[final_train["label"] == "Pneumonia"].groupby("source").size())


if __name__ == "__main__":
    main()
