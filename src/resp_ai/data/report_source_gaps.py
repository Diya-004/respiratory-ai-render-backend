from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_gap_report(frame: pd.DataFrame, *, target_per_source_class: int) -> dict:
    source_label_counts = pd.crosstab(frame["source"], frame["label"]).sort_index().sort_index(axis=1)
    source_label_patients = (
        frame.groupby(["source", "label"])["patient_id"].nunique().unstack(fill_value=0).sort_index().sort_index(axis=1)
    )

    gap_rows: list[dict] = []
    for source in source_label_counts.index:
        for label in source_label_counts.columns:
            current_count = int(source_label_counts.loc[source, label])
            current_patients = int(source_label_patients.loc[source, label])
            gap_rows.append(
                {
                    "source": source,
                    "label": label,
                    "current_recordings": current_count,
                    "current_patients": current_patients,
                    "target_recordings": int(target_per_source_class),
                    "recording_gap": max(int(target_per_source_class) - current_count, 0),
                }
            )

    hardest_pairs = sorted(gap_rows, key=lambda row: (-row["recording_gap"], row["source"], row["label"]))
    return {
        "target_per_source_class": int(target_per_source_class),
        "class_counts": frame["label"].value_counts().sort_index().to_dict(),
        "patients_per_label": frame.groupby("label")["patient_id"].nunique().sort_index().to_dict(),
        "source_by_label": source_label_counts.to_dict(orient="index"),
        "patients_by_source_label": source_label_patients.to_dict(orient="index"),
        "source_class_gaps": gap_rows,
        "largest_gaps_first": hardest_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report source/class gaps for collecting matched respiratory recordings.")
    parser.add_argument("--metadata", required=True, help="Path to the curated metadata CSV")
    parser.add_argument("--output", required=True, help="Path to write the JSON gap report")
    parser.add_argument("--target-per-source-class", type=int, default=40, help="Desired minimum recordings per source/class")
    parser.add_argument(
        "--include-sources",
        help="Optional comma-separated list of sources to include, e.g. icbhi,chest_wall",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(metadata_path)
    if args.include_sources:
        include_sources = [value.strip() for value in str(args.include_sources).split(",") if value.strip()]
        frame = frame[frame["source"].isin(include_sources)].copy()
        if frame.empty:
            raise ValueError("No rows remain after applying --include-sources.")
    report = build_gap_report(frame, target_per_source_class=args.target_per_source_class)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Saved gap report to:", output_path)
    print(json.dumps(report["largest_gaps_first"][:8], indent=2))


if __name__ == "__main__":
    main()
