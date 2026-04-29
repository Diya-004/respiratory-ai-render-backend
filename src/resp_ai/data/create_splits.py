from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ["train", "val", "test"]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_groups_within_subset(subset: pd.DataFrame, seed: int) -> dict[str, str]:
    grouped = (
        subset.groupby("patient_id")
        .size()
        .reset_index(name="group_size")
        .sample(frac=1.0, random_state=seed)
        .sort_values("group_size", ascending=False)
        .reset_index(drop=True)
    )

    total_subset_samples = int(grouped["group_size"].sum())
    target_sizes = {split: total_subset_samples * ratio for split, ratio in RATIOS.items()}
    current_sizes = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}

    for _, row in grouped.iterrows():
        group_size = int(row["group_size"])

        def score(split: str) -> tuple[float, int]:
            filled_ratio = current_sizes[split] / max(target_sizes[split], 1.0)
            projected_gap = abs((current_sizes[split] + group_size) - target_sizes[split])
            return (filled_ratio, projected_gap)

        chosen_split = min(SPLITS, key=score)
        assignments[str(row["patient_id"])] = chosen_split
        current_sizes[chosen_split] += group_size

    return assignments


def assign_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["split"] = ""

    offset = 0
    for label in sorted(frame["label"].unique()):
        label_frame = frame[frame["label"] == label].copy()
        for source in sorted(label_frame["source"].unique()):
            subset = label_frame[label_frame["source"] == source].copy()
            assignments = assign_groups_within_subset(subset, seed + offset)
            frame.loc[subset.index, "split"] = subset["patient_id"].astype(str).map(assignments)
            offset += 1

    rng = np.random.default_rng(seed)
    return frame.sample(frac=1.0, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label-balanced, group-aware train/val/test splits.")
    parser.add_argument("--metadata", required=True, help="Path to master_metadata.csv")
    parser.add_argument("--output-root", required=True, help="Data root for writing split CSVs")
    parser.add_argument("--splits-dirname", default="splits", help="Output folder name for split CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    splits_root = output_root / args.splits_dirname
    splits_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(metadata_path)
    split_frame = assign_splits(frame, seed=args.seed)
    split_frame.to_csv(splits_root / "all_splits.csv", index=False)

    for split in SPLITS:
        subset = split_frame[split_frame["split"] == split].copy()
        subset.to_csv(splits_root / f"{split}.csv", index=False)
        print(f"{split}: {len(subset)} samples")
        print(subset["label"].value_counts().sort_index())
        print()


if __name__ == "__main__":
    main()
