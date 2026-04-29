from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf


def build_audio_stats(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in metadata.itertuples(index=False):
        info = sf.info(row.file_path)
        rows.append(
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "source": row.source,
                "raw_label": row.raw_label,
                "patient_id": row.patient_id,
                "samplerate": int(info.samplerate),
                "channels": int(info.channels),
                "duration_sec": float(info.duration),
                "frames": int(info.frames),
                "format": info.format,
                "subtype": info.subtype,
            }
        )
    return pd.DataFrame(rows)


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    return frame.reset_index().to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit merged dataset quality and split balance.")
    parser.add_argument("--data-root", required=True, help="Path to rebuild data root.")
    parser.add_argument("--output", required=True, help="Path to JSON audit report.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data_root / "metadata" / "master_metadata.csv")
    raw_records = pd.read_csv(data_root / "metadata" / "raw_source_records.csv")
    audio_stats = build_audio_stats(metadata)

    report = {
        "raw_records": int(len(raw_records)),
        "deduped_records": int(len(metadata)),
        "dedup_removed": int(len(raw_records) - len(metadata)),
        "class_counts": metadata["label"].value_counts().sort_index().to_dict(),
        "patients_per_label": metadata.groupby("label")["patient_id"].nunique().sort_index().to_dict(),
        "source_by_label": frame_to_records(pd.crosstab(metadata["label"], metadata["source"])),
        "sample_rate_distribution": audio_stats["samplerate"].value_counts().sort_index().to_dict(),
        "channel_distribution": audio_stats["channels"].value_counts().sort_index().to_dict(),
        "duration_summary": audio_stats["duration_sec"].describe().round(4).to_dict(),
        "duration_by_label": frame_to_records(
            audio_stats.groupby("label")["duration_sec"]
            .describe()[["count", "mean", "std", "min", "50%", "max"]]
            .round(4)
        ),
        "raw_label_counts_by_normalized_label": frame_to_records(
            metadata.groupby(["label", "raw_label"]).size().reset_index(name="count")
        ),
        "source_samplerate_by_label": frame_to_records(
            pd.crosstab([audio_stats["label"], audio_stats["source"]], audio_stats["samplerate"])
        ),
        "very_short_lt_1s": int((audio_stats["duration_sec"] < 1.0).sum()),
        "very_long_gt_20s": int((audio_stats["duration_sec"] > 20.0).sum()),
        "top_longest_examples": (
            metadata.assign(duration_sec=audio_stats["duration_sec"])
            .sort_values("duration_sec", ascending=False)
            [["label", "source", "source_filename", "duration_sec"]]
            .head(10)
            .to_dict(orient="records")
        ),
    }

    split_path = data_root / "splits" / "all_splits.csv"
    if split_path.exists():
        split_frame = pd.read_csv(split_path)
        report["split_by_label"] = frame_to_records(pd.crosstab(split_frame["label"], split_frame["split"]))
        report["patients_by_label_and_split"] = frame_to_records(
            split_frame.groupby(["label", "split"])["patient_id"].nunique().unstack(fill_value=0)
        )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Saved dataset audit to:", output_path)


if __name__ == "__main__":
    main()
