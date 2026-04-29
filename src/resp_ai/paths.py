from __future__ import annotations

from pathlib import Path


def project_root_from_config(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().resolve().parents[1]


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path
