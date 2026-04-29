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
    text = str(value)
    return text.replace(str(base_root), str(output_root))


def load_full_train_pool(base_root: Path) -> pd.DataFrame:
    recording_train = pd.read_csv(base_root / "recording_splits" / "train.csv")
    pool_root = base_root / "_train_pool"

    rows: list[dict] = []
    for row in recording_train.to_dict(orient="records"):
        label = row["label"]
        sample_id = row["sample_id"]
        matches = sorted((pool_root / label).glob(f"{sample_id}_clip*.wav"))
        for match in matches:
            clip_index = int(match.stem.rsplit("clip", 1)[1])
            clip_row = dict(row)
            clip_row.update(
                {
                    "clip_id": match.stem,
                    "clip_index": clip_index,
                    "clip_start_sec": np.nan,
                    "clip_strategy": "overlap_50",
                    "pool_path": str(match),
                    "processed_path": "",
                    "is_augmented": False,
                    "augmentation_type": "",
                    "augmentation_source_clip_id": "",
                }
            )
            rows.append(clip_row)
    return pd.DataFrame(rows).sort_values(["source", "label", "sample_id", "clip_index"]).reset_index(drop=True)


def copy_static_dirs(base_root: Path, output_root: Path) -> None:
    for dirname in ["audio", "metadata", "recording_splits", "val", "test"]:
        src = base_root / dirname
        dst = output_root / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def copy_original_train_rows(frame: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    train_root = output_root / "train"
    if train_root.exists():
        shutil.rmtree(train_root)
    ensure_dir(train_root)

    exported_rows: list[dict] = []
    for row in tqdm(frame.to_dict(orient="records"), desc="copy_original_train"):
        src_path = Path(row["pool_path"])
        dst_path = train_root / row["label"] / src_path.name
        ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)

        updated = dict(row)
        updated["processed_path"] = str(dst_path)
        exported_rows.append(updated)

    return pd.DataFrame(exported_rows)


def _round_robin_rows(frame: pd.DataFrame) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame.sort_values(["sample_id", "clip_index"]).to_dict(orient="records"):
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


def _augment_strength(count: int, target: int) -> tuple[float, int]:
    shortage_ratio = max(target - count, 0) / max(target, 1)
    if count < 20:
        return (1.15 + 0.10 * shortage_ratio, 3)
    if count < 80:
        return (1.0 + 0.10 * shortage_ratio, 2)
    return (0.9 + 0.08 * shortage_ratio, 2)


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


def augment_group_to_target(
    frame: pd.DataFrame,
    *,
    output_root: Path,
    sample_rate: int,
    target_samples: int,
    target_count: int,
    seed: int,
    stage_name: str,
) -> tuple[pd.DataFrame, list[dict]]:
    train_root = output_root / "train"
    augmented_rows: list[dict] = []
    augmentation_events: list[dict] = []

    current_count = len(frame)
    if current_count == 0 or current_count >= target_count:
        return pd.DataFrame(), augmentation_events

    label = str(frame.iloc[0]["label"])
    strength, num_ops = _augment_strength(current_count, target_count)
    ordered_rows = _round_robin_rows(frame)
    needed = target_count - current_count

    for index in range(needed):
        base_row = ordered_rows[index % len(ordered_rows)]
        base_clip_path = Path(base_row["processed_path"])
        base_clip_id = str(base_row["clip_id"])
        base_source = str(base_row["source"])
        aug_clip_id = f"{base_clip_id}_{stage_name}_aug{index:03d}"
        dst_path = train_root / label / f"{aug_clip_id}.wav"
        local_seed = seed + (hash((base_source, label, stage_name, index, base_clip_id)) & 0xFFFF_FFFF)
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
                "augmentation_type": f"{stage_name}:strength_{strength:.3f}:ops_{num_ops}",
                "augmentation_source_clip_id": base_clip_id,
            }
        )
        augmented_rows.append(new_row)
        augmentation_events.append(
            {
                "stage": stage_name,
                "source": base_source,
                "label": label,
                "base_clip_id": base_clip_id,
                "augmented_clip_id": aug_clip_id,
                "strength": round(float(strength), 4),
                "num_ops": int(num_ops),
            }
        )

    return pd.DataFrame(augmented_rows), augmentation_events


def rewrite_split_paths(frame: pd.DataFrame, *, base_root: Path, output_root: Path) -> pd.DataFrame:
    updated = frame.copy()
    for column in ["file_path", "processed_path", "pool_path"]:
        if column in updated.columns:
            updated[column] = updated[column].map(lambda value: replace_dataset_root(value, base_root, output_root))
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-augmented dataset that fills source/class and class-level gaps.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--base-data-root", required=True, help="Existing dataset_final root")
    parser.add_argument("--output-root", required=True, help="Destination dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-source-class-clips", type=int, default=120)
    parser.add_argument("--min-class-clips", type=int, default=500)
    args = parser.parse_args()

    config = load_yaml(args.config)
    audio_config = load_audio_config(config)
    base_root = Path(args.base_data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    copy_static_dirs(base_root, output_root)

    original_train = load_full_train_pool(base_root)
    exported_train = copy_original_train_rows(original_train, output_root)
    augmented_frames = [exported_train]
    augmentation_events: list[dict] = []

    # Stage 1: lift weak source/class pairs to a minimum clip count.
    current_train = exported_train.copy()
    for (source, label), group in current_train.groupby(["source", "label"], sort=True):
        if len(group) >= args.min_source_class_clips:
            continue
        augmented, events = augment_group_to_target(
            group,
            output_root=output_root,
            sample_rate=audio_config.sample_rate,
            target_samples=audio_config.target_samples,
            target_count=args.min_source_class_clips,
            seed=args.seed,
            stage_name="sourcegap",
        )
        if not augmented.empty:
            augmented_frames.append(augmented)
            augmentation_events.extend(events)
            current_train = pd.concat([current_train, augmented], ignore_index=True)

    # Stage 2: lift weak classes to a minimum clip count after source-gap filling.
    current_train = pd.concat(augmented_frames, ignore_index=True)
    for label, group in current_train.groupby("label", sort=True):
        if len(group) >= args.min_class_clips:
            continue
        ordered = []
        source_counts = group.groupby("source").size().sort_values()
        for source in source_counts.index:
            ordered.append(group[group["source"] == source].copy())
        prioritized_group = pd.concat(ordered, ignore_index=True) if ordered else group
        augmented, events = augment_group_to_target(
            prioritized_group,
            output_root=output_root,
            sample_rate=audio_config.sample_rate,
            target_samples=audio_config.target_samples,
            target_count=args.min_class_clips,
            seed=args.seed + 1000,
            stage_name="classgap",
        )
        if not augmented.empty:
            augmented_frames.append(augmented)
            augmentation_events.extend(events)
            current_train = pd.concat([current_train, augmented], ignore_index=True)

    final_train = pd.concat(augmented_frames, ignore_index=True).sort_values(
        ["label", "source", "sample_id", "clip_id"]
    ).reset_index(drop=True)

    splits_root = output_root / "splits"
    ensure_dir(splits_root)

    final_train["split"] = "train"
    final_train.to_csv(splits_root / "train.csv", index=False)

    for split in ["val", "test"]:
        split_frame = pd.read_csv(base_root / "splits" / f"{split}.csv")
        split_frame = rewrite_split_paths(split_frame, base_root=base_root, output_root=output_root)
        split_frame.to_csv(splits_root / f"{split}.csv", index=False)

    all_splits = pd.concat(
        [
            final_train,
            pd.read_csv(splits_root / "val.csv"),
            pd.read_csv(splits_root / "test.csv"),
        ],
        ignore_index=True,
    )
    all_splits.to_csv(splits_root / "all_splits.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "min_source_class_clips": int(args.min_source_class_clips),
        "min_class_clips": int(args.min_class_clips),
        "original_train_rows": int(len(exported_train)),
        "final_train_rows": int(len(final_train)),
        "original_source_label_counts": [
            {"source": source, "label": label, "count": int(count)}
            for (source, label), count in exported_train.groupby(["source", "label"]).size().items()
        ],
        "final_source_label_counts": [
            {"source": source, "label": label, "count": int(count)}
            for (source, label), count in final_train.groupby(["source", "label"]).size().items()
        ],
        "original_label_counts": exported_train["label"].value_counts().sort_index().to_dict(),
        "final_label_counts": final_train["label"].value_counts().sort_index().to_dict(),
        "augmented_rows": int(final_train["is_augmented"].fillna(False).sum()),
        "augmentation_events": augmentation_events,
    }

    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata" / "augmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Gap-augmented dataset created at:", output_root)
    print("Original train label counts:")
    print(exported_train["label"].value_counts().sort_index())
    print("\nFinal train label counts:")
    print(final_train["label"].value_counts().sort_index())
    print("\nFinal source/label counts:")
    print(final_train.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
