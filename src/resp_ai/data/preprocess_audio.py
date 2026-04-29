from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from resp_ai.config import load_audio_config, load_yaml
from resp_ai.features.audio import save_preprocessed_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Create normalized fixed-length audio clips for all splits.")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--data-root", required=True, help="Path to rebuild data root")
    args = parser.parse_args()

    config = load_yaml(args.config)
    audio_config = load_audio_config(config)
    data_root = Path(args.data_root).expanduser().resolve()
    processed_root = data_root / "processed_audio"
    processed_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_csv = data_root / "splits" / f"{split}.csv"
        frame = pd.read_csv(split_csv)
        processed_rows: list[dict] = []

        for row in tqdm(frame.to_dict(orient="records"), desc=f"preprocess:{split}"):
            src_path = Path(row["file_path"])
            dst_dir = processed_root / split / row["label"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / f"{row['sample_id']}.wav"
            clip_info = save_preprocessed_clip(str(src_path), str(dst_path), audio_config)

            updated = dict(row)
            updated["processed_path"] = str(dst_path)
            updated.update(clip_info)
            processed_rows.append(updated)

        processed_frame = pd.DataFrame(processed_rows)
        processed_frame.to_csv(split_csv, index=False)
        print(f"Updated split file with processed clips: {split_csv}")


if __name__ == "__main__":
    main()
