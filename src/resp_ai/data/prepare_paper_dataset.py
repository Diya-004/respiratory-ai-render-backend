from __future__ import annotations

import argparse
import hashlib
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


def read_icbhi_diagnosis(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["patient_id", "diagnosis"], engine="python")
    return dict(zip(frame["patient_id"].astype(str), frame["diagnosis"].astype(str)))


def read_icbhi_official_split(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["recording_id", "official_split"], engine="python")
    return dict(zip(frame["recording_id"].astype(str), frame["official_split"].astype(str)))


def resolve_audio_dir(root: Path, *, direct_name: str | None = None) -> Path:
    root = root.expanduser().resolve()
    if direct_name is not None:
        candidate = root / direct_name
        if candidate.exists():
            return candidate
    if any(root.glob("*.wav")):
        return root
    for pattern in ("audio_and_txt_files", "RespiratoryDatabase@TR"):
        matches = list(root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find audio directory under {root}")


def prepare_icbhi_records(audio_dir: Path, diagnosis_path: Path, split_path: Path | None) -> list[dict]:
    diagnosis_map = read_icbhi_diagnosis(diagnosis_path)
    official_split_map = read_icbhi_official_split(split_path)

    records: list[dict] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        patient_id = wav_path.stem.split("_", 1)[0]
        raw_label = diagnosis_map.get(patient_id)
        if raw_label is None:
            continue
        label = normalize_label(raw_label)
        if label is None:
            continue
        records.append(
            {
                "source": "icbhi",
                "source_filename": wav_path.name,
                "patient_id": patient_id,
                "raw_label": raw_label,
                "label": label,
                "official_source_split": official_split_map.get(wav_path.stem, ""),
                "src_path": str(wav_path),
            }
        )
    return records


def prepare_chest_wall_records(audio_dir: Path) -> list[dict]:
    records: list[dict] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        stem = wav_path.stem
        try:
            raw_label = stem.split("_", 1)[1].split(",", 1)[0].strip()
        except IndexError:
            raw_label = stem
        label = normalize_label(raw_label)
        if label is None:
            continue
        records.append(
            {
                "source": "chest_wall",
                "source_filename": wav_path.name,
                "patient_id": guess_patient_id("mendeley", wav_path.name),
                "raw_label": raw_label,
                "label": label,
                "official_source_split": "",
                "src_path": str(wav_path),
            }
        )
    return records


def prepare_respdb_tr_records(audio_dir: Path) -> list[dict]:
    records: list[dict] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        records.append(
            {
                "source": "respdb_tr",
                "source_filename": wav_path.name,
                "patient_id": guess_patient_id("respdb_tr", wav_path.name),
                "raw_label": "COPD",
                "label": "COPD",
                "official_source_split": "",
                "src_path": str(wav_path),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the paper-native respiratory dataset sources.")
    parser.add_argument("--output-root", required=True, help="Destination data root for the prepared dataset")
    parser.add_argument("--icbhi-root", required=True, help="Directory containing the ICBHI wav files")
    parser.add_argument("--icbhi-diagnosis", required=True, help="Path to ICBHI_Challenge_diagnosis.txt")
    parser.add_argument("--icbhi-train-test", help="Optional path to ICBHI_challenge_train_test.txt")
    parser.add_argument("--chest-wall-root", required=True, help="Directory containing the chest-wall wav files")
    parser.add_argument("--respdb-tr-root", help="Directory containing RespiratoryDatabase@TR wav files")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    audio_root = output_root / "audio"
    metadata_root = output_root / "metadata"
    ensure_dir(audio_root)
    ensure_dir(metadata_root)

    icbhi_audio_dir = resolve_audio_dir(Path(args.icbhi_root), direct_name="audio_and_txt_files")
    chest_wall_audio_dir = resolve_audio_dir(Path(args.chest_wall_root))
    source_records = prepare_icbhi_records(
        icbhi_audio_dir,
        Path(args.icbhi_diagnosis).expanduser().resolve(),
        Path(args.icbhi_train_test).expanduser().resolve() if args.icbhi_train_test else None,
    ) + prepare_chest_wall_records(chest_wall_audio_dir)

    if args.respdb_tr_root:
        respdb_tr_audio_dir = resolve_audio_dir(Path(args.respdb_tr_root), direct_name="RespiratoryDatabase@TR")
        source_records += prepare_respdb_tr_records(respdb_tr_audio_dir)
    source_frame = pd.DataFrame(source_records)

    copied_records: list[dict] = []
    seen_hashes: set[str] = set()

    for record in tqdm(source_records, desc="Copying paper audio"):
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
                "official_source_split": record["official_source_split"],
                "sha256": file_hash,
                "file_path": str(dst_path),
            }
        )

    frame = pd.DataFrame(copied_records).sort_values(["label", "source", "sample_id"]).reset_index(drop=True)
    frame.to_csv(metadata_root / "master_metadata.csv", index=False)
    source_frame.to_csv(metadata_root / "raw_source_records.csv", index=False)

    print("Prepared paper dataset written to:", output_root)
    print("Saved metadata:", metadata_root / "master_metadata.csv")
    print(frame["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
