"""Shared bookkeeping for raw transaction conversion."""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = ".raw_sources.json"


def source_manifest(files: list[Path]) -> dict[str, object]:
    """Describe the raw files whose rows were included in a conversion."""
    return {
        "version": 1,
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        ],
    }


def write_source_manifest(files: list[Path], out_dir: Path) -> None:
    """Record the exact raw inputs after a successful conversion."""
    manifest_path = out_dir / MANIFEST_NAME
    temp_path = out_dir / f"{MANIFEST_NAME}.tmp"
    temp_path.write_text(
        json.dumps(source_manifest(files), indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(manifest_path)


def conversion_is_current(files: list[Path], out_dir: Path) -> bool:
    """Return whether yearly Parquet files represent the current raw inputs."""
    if not list(out_dir.glob("transactions_year_*.parquet")):
        return False

    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return False
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == source_manifest(files)
