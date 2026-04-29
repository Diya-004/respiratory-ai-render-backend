from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from resp_ai.labels import guess_patient_id, normalize_label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_record(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def prepare_kaggle_records(major_root: Path) -> list[dict]:
    dataset_root = major_root / "respiratory-sound-database" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database"
    audio_dir = dataset_root / "audio_and_txt_files"
    diagnosis_csv = dataset_root / "patient_diagnosis.csv"

    diagnosis_df = pd.read_csv(diagnosis_csv, header=None, names=["patient_id", "diagnosis"])
    diag_map = dict(zip(diagnosis_df["patient_id"].astype(str), diagnosis_df["diagnosis"].astype(str)))

    records: list[dict] = []
    for filename in sorted(os.listdir(audio_dir)):
        if not filename.lower().endswith(".wav"):
            continue
        patient_id = filename.split("_", 1)[0]
        raw_label = diag_map.get(patient_id)
        if raw_label is None:
            continue
        label = normalize_label(raw_label)
        if label is None:
            continue
        records.append(
            {
                "source": "kaggle",
                "source_filename": filename,
                "patient_id": patient_id,
                "raw_label": raw_label,
                "label": label,
                "src_path": str(audio_dir / filename),
            }
        )
    return records


def prepare_mendeley_records(major_root: Path) -> list[dict]:
    audio_dir = major_root / "mendley" / "Audio Files"
    records: list[dict] = []
    for filename in sorted(os.listdir(audio_dir)):
        if not filename.lower().endswith(".wav"):
            continue
        stem = filename.rsplit(".", 1)[0]
        try:
            raw_label = stem.split("_", 1)[1].split(",")[0].strip()
        except IndexError:
            raw_label = stem
        label = normalize_label(raw_label)
        if label is None:
            continue
        records.append(
            {
                "source": "mendeley",
                "source_filename": filename,
                "patient_id": guess_patient_id("mendeley", filename),
                "raw_label": raw_label,
                "label": label,
                "src_path": str(audio_dir / filename),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and normalize the respiratory audio dataset.")
    parser.add_argument("--major-project-root", required=True, help="Path to the original project folder.")
    parser.add_argument("--output-root", required=True, help="Path to the new rebuild data folder.")
    args = parser.parse_args()

    major_root = Path(args.major_project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    audio_root = output_root / "audio"
    metadata_root = output_root / "metadata"
    ensure_dir(audio_root)
    ensure_dir(metadata_root)

    source_records = prepare_kaggle_records(major_root) + prepare_mendeley_records(major_root)
    source_frame = pd.DataFrame(source_records)

    copied_records: list[dict] = []
    seen_hashes: set[str] = set()

    for record in tqdm(source_records, desc="Copying audio"):
        src_path = Path(record["src_path"])
        file_hash = sha256_file(src_path)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)

        sample_id = f"{record['source']}_{src_path.stem}"
        dst_path = audio_root / record["label"] / f"{sample_id}.wav"
        copy_record(src_path, dst_path)

        copied_records.append(
            {
                "sample_id": sample_id,
                "source": record["source"],
                "source_filename": record["source_filename"],
                "patient_id": record["patient_id"],
                "raw_label": record["raw_label"],
                "label": record["label"],
                "sha256": file_hash,
                "file_path": str(dst_path),
            }
        )

    frame = pd.DataFrame(copied_records).sort_values(["label", "source", "sample_id"]).reset_index(drop=True)
    frame.to_csv(metadata_root / "master_metadata.csv", index=False)
    source_frame.to_csv(metadata_root / "raw_source_records.csv", index=False)

    print("Prepared dataset written to:", output_root)
    print("Saved metadata:", metadata_root / "master_metadata.csv")
    print(frame["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
