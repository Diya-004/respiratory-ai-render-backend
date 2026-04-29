from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from resp_ai.labels import normalize_label
from resp_ai.data.prepare_paper_dataset import (
    copy_record,
    ensure_dir,
    prepare_chest_wall_records,
    prepare_icbhi_records,
    prepare_respdb_tr_records,
    resolve_audio_dir,
    sha256_file,
)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_optional_path(value: str | None, *, base_dir: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate.resolve()


def _resolve_required_path(value: str, *, base_dir: Path) -> Path:
    resolved = _resolve_optional_path(value, base_dir=base_dir)
    if resolved is None:
        raise ValueError("Expected a required path value.")
    return resolved


def prepare_manifest_csv_records(source_spec: dict[str, Any], *, base_dir: Path) -> tuple[list[dict], list[dict]]:
    source_name = str(source_spec["source_name"])
    metadata_csv = _resolve_required_path(str(source_spec["metadata_csv"]), base_dir=base_dir)
    audio_root = _resolve_optional_path(source_spec.get("audio_root"), base_dir=base_dir)
    file_path_column = str(source_spec.get("file_path_column", "file_path"))
    patient_id_column = str(source_spec.get("patient_id_column", "patient_id"))
    raw_label_column = str(source_spec.get("raw_label_column", "raw_label"))
    label_column = source_spec.get("label_column")
    split_column = source_spec.get("split_column")
    source_filename_column = source_spec.get("source_filename_column")

    frame = pd.read_csv(metadata_csv)
    keep_records: list[dict] = []
    skipped_records: list[dict] = []

    for row in frame.to_dict(orient="records"):
        path_value = row.get(file_path_column)
        if pd.isna(path_value):
            skipped_records.append({"source": source_name, "row": row, "skip_reason": "missing_file_path"})
            continue

        file_path = Path(str(path_value)).expanduser()
        if not file_path.is_absolute():
            if audio_root is not None:
                file_path = (audio_root / file_path).resolve()
            else:
                file_path = (metadata_csv.parent / file_path).resolve()
        else:
            file_path = file_path.resolve()

        if not file_path.exists():
            skipped_records.append({"source": source_name, "row": row, "skip_reason": f"missing_audio:{file_path}"})
            continue

        patient_id_value = row.get(patient_id_column)
        if pd.isna(patient_id_value):
            skipped_records.append({"source": source_name, "row": row, "skip_reason": "missing_patient_id"})
            continue

        if label_column:
            normalized_value = row.get(str(label_column))
            label = None if pd.isna(normalized_value) else str(normalized_value).strip()
            if label not in {"Asthma", "COPD", "Normal", "Pneumonia"}:
                raw_label_value = row.get(raw_label_column, normalized_value)
                label = None if pd.isna(raw_label_value) else normalize_label(str(raw_label_value))
        else:
            raw_label_value = row.get(raw_label_column)
            label = None if pd.isna(raw_label_value) else normalize_label(str(raw_label_value))

        if label is None:
            skipped_records.append({"source": source_name, "row": row, "skip_reason": "unmapped_label"})
            continue

        raw_label_value = row.get(raw_label_column)
        raw_label = "" if pd.isna(raw_label_value) else str(raw_label_value)
        source_filename_value = row.get(str(source_filename_column)) if source_filename_column else None
        source_filename = file_path.name if pd.isna(source_filename_value) or source_filename_value is None else str(source_filename_value)
        official_split_value = row.get(str(split_column)) if split_column else ""
        official_split = "" if pd.isna(official_split_value) else str(official_split_value)

        keep_records.append(
            {
                "source": source_name,
                "source_filename": source_filename,
                "patient_id": str(patient_id_value),
                "raw_label": raw_label,
                "label": label,
                "official_source_split": official_split,
                "src_path": str(file_path),
            }
        )

    return keep_records, skipped_records


def build_records_from_source(source_spec: dict[str, Any], *, base_dir: Path) -> tuple[list[dict], list[dict]]:
    parser_name = str(source_spec["parser"]).strip().lower()

    if parser_name == "icbhi":
        audio_dir = resolve_audio_dir(_resolve_required_path(str(source_spec["root"]), base_dir=base_dir), direct_name="audio_and_txt_files")
        records = prepare_icbhi_records(
            audio_dir,
            _resolve_required_path(str(source_spec["diagnosis_path"]), base_dir=base_dir),
            _resolve_optional_path(source_spec.get("split_path"), base_dir=base_dir),
        )
        return records, []

    if parser_name == "chest_wall":
        audio_dir = resolve_audio_dir(_resolve_required_path(str(source_spec["root"]), base_dir=base_dir))
        return prepare_chest_wall_records(audio_dir), []

    if parser_name == "respdb_tr":
        audio_dir = resolve_audio_dir(_resolve_required_path(str(source_spec["root"]), base_dir=base_dir), direct_name="RespiratoryDatabase@TR")
        return prepare_respdb_tr_records(audio_dir), []

    if parser_name == "manifest_csv":
        return prepare_manifest_csv_records(source_spec, base_dir=base_dir)

    raise ValueError(f"Unsupported parser in source manifest: {parser_name}")


def copy_and_index_records(records: list[dict], *, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    audio_root = output_root / "audio"
    metadata_root = output_root / "metadata"
    ensure_dir(audio_root)
    ensure_dir(metadata_root)

    copied_records: list[dict] = []
    dedup_removed: list[dict] = []
    seen_hashes: dict[str, str] = {}

    for record in tqdm(records, desc="Copying extended audio"):
        src_path = Path(record["src_path"])
        file_hash = sha256_file(src_path)
        if file_hash in seen_hashes:
            duplicate = dict(record)
            duplicate["sha256"] = file_hash
            duplicate["dedup_keep_sample_id"] = seen_hashes[file_hash]
            dedup_removed.append(duplicate)
            continue

        sample_id = f"{record['source']}_{src_path.stem}"
        dst_path = audio_root / record["label"] / f"{sample_id}.wav"
        copy_record(src_path, dst_path)
        seen_hashes[file_hash] = sample_id

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
    dedup_frame = pd.DataFrame(dedup_removed).sort_values(["source", "source_filename"]).reset_index(drop=True)
    return frame, dedup_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an extended respiratory dataset from a YAML source manifest.")
    parser.add_argument("--manifest", required=True, help="YAML manifest describing input dataset sources")
    parser.add_argument("--output-root", help="Optional override for the manifest output_root")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    base_dir = manifest_path.parent

    output_root_value = args.output_root or manifest.get("output_root")
    if not output_root_value:
        raise ValueError("The manifest must define output_root, or you must pass --output-root.")
    output_root = _resolve_required_path(str(output_root_value), base_dir=base_dir)

    source_specs = manifest.get("sources", [])
    if not source_specs:
        raise ValueError("The manifest does not contain any sources.")

    all_records: list[dict] = []
    skipped_records: list[dict] = []
    for source_spec in source_specs:
        records, skipped = build_records_from_source(source_spec, base_dir=base_dir)
        all_records.extend(records)
        skipped_records.extend(skipped)

    frame, dedup_frame = copy_and_index_records(all_records, output_root=output_root)
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    frame.to_csv(metadata_root / "master_metadata.csv", index=False)
    pd.DataFrame(all_records).sort_values(["source", "label", "source_filename"]).reset_index(drop=True).to_csv(
        metadata_root / "raw_source_records.csv",
        index=False,
    )
    pd.DataFrame(skipped_records).to_json(metadata_root / "skipped_manifest_rows.json", orient="records", indent=2)
    dedup_frame.to_csv(metadata_root / "dedup_removed.csv", index=False)

    print("Prepared extended dataset written to:", output_root)
    print("Saved metadata:", metadata_root / "master_metadata.csv")
    print(frame["label"].value_counts().sort_index())
    if skipped_records:
        print(f"Skipped manifest rows: {len(skipped_records)}")
    if len(dedup_frame) > 0:
        print(f"Deduplicated files removed: {len(dedup_frame)}")


if __name__ == "__main__":
    main()
