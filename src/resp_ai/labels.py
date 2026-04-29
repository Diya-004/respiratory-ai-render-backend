from __future__ import annotations

import re
from typing import Optional

CLASS_NAMES = ["Asthma", "COPD", "Normal", "Pneumonia"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_label(raw_label: str) -> Optional[str]:
    label = _compact(raw_label)

    if label in {"healthy", "normal", "n"}:
        return "Normal"
    if label == "urti":
        return "Normal"
    if "asthma" in label:
        return "Asthma"
    if "copd" in label:
        return "COPD"

    pneumonia_aliases = {
        "pneumonia",
        "bronchiectasis",
        "bronchiolitis",
        "lrti",
        "heart failure",
        "lung fibrosis",
        "pleural effusion",
        "plueral effusion",
        "bron",
        "crep",
    }
    if any(alias in label for alias in pneumonia_aliases):
        return "Pneumonia"

    return None


def guess_patient_id(source: str, filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]

    if source == "kaggle":
        return stem.split("_", 1)[0]

    if source == "mendeley":
        prefix = stem.split("_", 1)[0]
        match = re.match(r"^[A-Za-z]P(\d+)$", prefix)
        if match:
            return match.group(1)
        return prefix if prefix else stem

    if source == "respdb_tr":
        prefix = stem.split("_", 1)[0]
        match = re.match(r"^(H\d+)$", prefix, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return prefix if prefix else stem

    return stem
