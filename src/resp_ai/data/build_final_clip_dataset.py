from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from resp_ai.config import load_audio_config, load_yaml
from resp_ai.features.audio import fit_audio_length, prepare_signal


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_trimmed_signal(path: str, sample_rate: int, trim_top_db: int) -> tuple[np.ndarray, float, float]:
    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    trimmed, _ = librosa.effects.trim(signal, top_db=trim_top_db)
    original_duration = len(signal) / max(sample_rate, 1)
    trimmed_duration = len(trimmed) / max(sample_rate, 1)
    if len(trimmed) == 0:
        trimmed = np.zeros(1, dtype=np.float32)
    trimmed = librosa.util.normalize(trimmed)
    return trimmed.astype(np.float32), round(original_duration, 4), round(trimmed_duration, 4)


def generate_overlapping_windows(signal: np.ndarray, target_samples: int, overlap: float) -> list[tuple[int, np.ndarray]]:
    if len(signal) <= target_samples:
        return [(0, fit_audio_length(signal, target_samples))]

    stride = max(int(target_samples * (1.0 - overlap)), 1)
    starts = list(range(0, len(signal) - target_samples + 1, stride))
    last_start = len(signal) - target_samples
    if starts[-1] != last_start:
        starts.append(last_start)

    return [(start, signal[start:start + target_samples]) for start in starts]


def save_audio(signal: np.ndarray, path: Path, sample_rate: int) -> None:
    ensure_dir(path.parent)
    sf.write(path, signal, sample_rate)


def build_train_pool(
    frame: pd.DataFrame,
    output_root: Path,
    sample_rate: int,
    trim_top_db: int,
    target_samples: int,
    overlap: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    pool_root = output_root / "_train_pool"
    if pool_root.exists():
        shutil.rmtree(pool_root)
    ensure_dir(pool_root)

    for row in tqdm(frame.to_dict(orient="records"), desc="train_pool"):
        signal, original_duration, trimmed_duration = load_trimmed_signal(
            row["file_path"], sample_rate, trim_top_db
        )
        windows = generate_overlapping_windows(signal, target_samples, overlap)
        for clip_index, (start, clip_signal) in enumerate(windows):
            clip_id = f"{row['sample_id']}_clip{clip_index:03d}"
            clip_path = pool_root / row["label"] / f"{clip_id}.wav"
            save_audio(clip_signal, clip_path, sample_rate)
            clip_row = dict(row)
            clip_row.update(
                {
                    "clip_id": clip_id,
                    "clip_index": clip_index,
                    "clip_start_sec": round(start / max(sample_rate, 1), 4),
                    "clip_strategy": f"overlap_{int(overlap * 100):02d}",
                    "original_duration_sec": original_duration,
                    "trimmed_duration_sec": trimmed_duration,
                    "processed_duration_sec": round(len(clip_signal) / max(sample_rate, 1), 4),
                    "pool_path": str(clip_path),
                }
            )
            rows.append(clip_row)

    return pd.DataFrame(rows)


def select_balanced_train_rows(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    selected_indices: list[int] = []
    rng = random.Random(seed)

    per_class_target = int(frame["label"].value_counts().min())
    for label in sorted(frame["label"].unique()):
        label_frame = frame[frame["label"] == label].copy()
        by_sample: dict[str, list[int]] = {}
        for record in label_frame.sort_values(["sample_id", "clip_index"]).itertuples():
            by_sample.setdefault(str(record.sample_id), []).append(int(record.Index))

        sample_ids = list(by_sample.keys())
        rng.shuffle(sample_ids)

        chosen_for_label: list[int] = []
        while sample_ids and len(chosen_for_label) < per_class_target:
            next_round: list[str] = []
            for sample_id in sample_ids:
                remaining = by_sample[sample_id]
                if remaining and len(chosen_for_label) < per_class_target:
                    chosen_for_label.append(remaining.pop(0))
                if remaining:
                    next_round.append(sample_id)
            rng.shuffle(next_round)
            sample_ids = next_round

        selected_indices.extend(chosen_for_label)

    return frame.loc[selected_indices].copy().reset_index(drop=True)


def export_train_subset(frame: pd.DataFrame, output_root: Path, sample_rate: int) -> pd.DataFrame:
    train_root = output_root / "train"
    if train_root.exists():
        shutil.rmtree(train_root)
    ensure_dir(train_root)

    exported_rows: list[dict] = []
    for row in tqdm(frame.to_dict(orient="records"), desc="export_train"):
        src_path = Path(row["pool_path"])
        dst_path = train_root / row["label"] / src_path.name
        ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)

        updated = dict(row)
        updated["processed_path"] = str(dst_path)
        exported_rows.append(updated)

    return pd.DataFrame(exported_rows)


def export_eval_split(
    frame: pd.DataFrame,
    split: str,
    output_root: Path,
    sample_rate: int,
    trim_top_db: int,
    target_samples: int,
) -> pd.DataFrame:
    split_root = output_root / split
    if split_root.exists():
        shutil.rmtree(split_root)
    ensure_dir(split_root)

    rows: list[dict] = []
    for row in tqdm(frame.to_dict(orient="records"), desc=f"export_{split}"):
        signal, original_duration, trimmed_duration = load_trimmed_signal(
            row["file_path"], sample_rate, trim_top_db
        )
        prepared = fit_audio_length(signal, target_samples)
        clip_id = row["sample_id"]
        dst_path = split_root / row["label"] / f"{clip_id}.wav"
        save_audio(prepared, dst_path, sample_rate)

        updated = dict(row)
        updated.update(
            {
                "clip_id": clip_id,
                "clip_index": 0,
                "clip_start_sec": 0.0,
                "clip_strategy": "best_window",
                "original_duration_sec": original_duration,
                "trimmed_duration_sec": trimmed_duration,
                "processed_duration_sec": round(len(prepared) / max(sample_rate, 1), 4),
                "processed_path": str(dst_path),
            }
        )
        rows.append(updated)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset_final with balanced train clips and patient-wise val/test clips."
    )
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--data-root", required=True, help="Root containing metadata and recording_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-overlap", type=float, default=0.5)
    args = parser.parse_args()

    config = load_yaml(args.config)
    audio_config = load_audio_config(config)
    data_root = Path(args.data_root).expanduser().resolve()
    recording_splits_root = data_root / "recording_splits"
    splits_root = data_root / "splits"
    ensure_dir(splits_root)

    train_frame = pd.read_csv(recording_splits_root / "train.csv")
    val_frame = pd.read_csv(recording_splits_root / "val.csv")
    test_frame = pd.read_csv(recording_splits_root / "test.csv")

    train_pool = build_train_pool(
        train_frame,
        data_root,
        sample_rate=audio_config.sample_rate,
        trim_top_db=audio_config.trim_top_db,
        target_samples=audio_config.target_samples,
        overlap=args.train_overlap,
    )
    balanced_train = select_balanced_train_rows(train_pool, seed=args.seed)
    exported_train = export_train_subset(
        balanced_train,
        data_root,
        sample_rate=audio_config.sample_rate,
    )
    exported_val = export_eval_split(
        val_frame,
        "val",
        data_root,
        sample_rate=audio_config.sample_rate,
        trim_top_db=audio_config.trim_top_db,
        target_samples=audio_config.target_samples,
    )
    exported_test = export_eval_split(
        test_frame,
        "test",
        data_root,
        sample_rate=audio_config.sample_rate,
        trim_top_db=audio_config.trim_top_db,
        target_samples=audio_config.target_samples,
    )

    split_frames = {
        "train": exported_train,
        "val": exported_val,
        "test": exported_test,
    }

    all_splits = []
    for split, frame in split_frames.items():
        frame = frame.copy()
        frame["split"] = split
        frame.to_csv(splits_root / f"{split}.csv", index=False)
        all_splits.append(frame)

    pd.concat(all_splits, ignore_index=True).to_csv(splits_root / "all_splits.csv", index=False)

    print("Final clip dataset created at:", data_root)
    for split, frame in split_frames.items():
        print(f"{split}: {len(frame)} clips")
        print(frame["label"].value_counts().sort_index())
        print()


if __name__ == "__main__":
    main()
