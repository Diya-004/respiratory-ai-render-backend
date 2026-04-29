from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STRICT_RAW_LABELS = {
    "Asthma": {"asthma"},
    "COPD": {"copd"},
    "Normal": {"healthy", "n"},
    "Pneumonia": {"pneumonia"},
}


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def curate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_rows: list[dict] = []
    drop_rows: list[dict] = []

    for row in frame.to_dict(orient="records"):
        raw_label = normalize_text(row["raw_label"])
        allowed = STRICT_RAW_LABELS.get(row["label"], set())
        if raw_label in allowed:
            keep_rows.append(row)
        else:
            drop = dict(row)
            drop["drop_reason"] = f"raw_label_not_allowed:{raw_label}"
            drop_rows.append(drop)

    curated = pd.DataFrame(keep_rows).sort_values(["label", "source", "sample_id"]).reset_index(drop=True)
    dropped = pd.DataFrame(drop_rows).sort_values(["label", "source", "sample_id"]).reset_index(drop=True)
    return curated, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter merged metadata down to strict diagnosis labels.")
    parser.add_argument("--metadata", required=True, help="Path to master_metadata.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for curated metadata outputs")
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(metadata_path)
    curated, dropped = curate_frame(frame)
    curated.to_csv(output_dir / "curated_metadata.csv", index=False)
    dropped.to_csv(output_dir / "dropped_records.csv", index=False)

    print("Curated dataset saved to:", output_dir / "curated_metadata.csv")
    print(curated["label"].value_counts().sort_index())
    print("\nDropped records:", len(dropped))


if __name__ == "__main__":
    main()
