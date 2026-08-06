"""Validated, origin-partitioned feature cache shared by LightGBM models."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Iterator

import duckdb
import pandas as pd

from src.models.benchmark.config import ROOT, BenchmarkDesign
from src.models.lightgbm.features.builder import (
    REMOVED_FEATURE_COLUMNS,
    _required_materialized_columns,
    create_feature_tables,
    materialize_features_for_origins,
)


FEATURE_SET_NAME = "lightgbm_daily"
FEATURE_SET_VERSION = "1"
FEATURE_BUILDER_VERSION = "2026-08-06.18"
DEFAULT_FEATURE_STORE_DIR = ROOT / "data" / "processed" / "model_features"


@dataclass(frozen=True)
class FeatureDataset:
    """Resolved immutable cache generation and its requested partitions."""

    directory: Path
    manifest_path: Path
    parquet_paths: tuple[Path, ...]
    fingerprint: str


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


class LightGBMFeatureStore:
    """Build missing origins once and safely reuse them across model processes."""

    def __init__(self, root: Path | str = DEFAULT_FEATURE_STORE_DIR) -> None:
        self.root = Path(root)

    def _source_signature(
        self,
        con: duckdb.DuckDBPyConnection,
        design: BenchmarkDesign,
    ) -> tuple[str, dict[str, object]]:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                MIN(period) AS first_period,
                MAX(period) AS last_period,
                BIT_XOR(HASH(
                    ARTIKEL_ID,
                    MARKT_ID,
                    period,
                    demand,
                    is_active,
                    COALESCE(reason_closed, ''),
                    action_flag,
                    sourcing_group,
                    category_id
                )) AS content_hash
            FROM benchmark_daily_rows
            """
        ).fetchone()
        if row is None or row[0] == 0:
            raise RuntimeError("benchmark_daily_rows is empty")
        design_values = {
            key: _json_value(value) for key, value in asdict(design).items()
        }
        signature = {
            "feature_set": FEATURE_SET_NAME,
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_builder_version": FEATURE_BUILDER_VERSION,
            "source": {
                "row_count": int(row[0]),
                "first_period": str(row[1]),
                "last_period": str(row[2]),
                "content_hash": str(row[3]),
            },
            "forecast_design": design_values,
        }
        encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20], signature

    def _generation_dir(self, fingerprint: str) -> Path:
        return (
            self.root
            / FEATURE_SET_NAME
            / f"v{FEATURE_SET_VERSION}"
            / fingerprint
        )

    @staticmethod
    def _partition_path(directory: Path, origin: pd.Timestamp) -> Path:
        return directory / f"origin={origin.date().isoformat()}" / "features.parquet"

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, object] | None:
        try:
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _partition_valid(
        con: duckdb.DuckDBPyConnection,
        path: Path,
        origin: pd.Timestamp,
    ) -> bool:
        if not path.exists():
            return False
        try:
            columns = {
                row[0]
                for row in con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=false)",
                    [str(path)],
                ).fetchall()
            }
            bounds = con.execute(
                """SELECT MIN(origin), MAX(origin), COUNT(*)
                FROM read_parquet(?, hive_partitioning=false)""",
                [str(path)],
            ).fetchone()
        except Exception:
            return False
        required = _required_materialized_columns()
        expected = origin.normalize()
        return bool(
            required.issubset(columns)
            and REMOVED_FEATURE_COLUMNS.isdisjoint(columns)
            and bounds is not None
            and bounds[2] > 0
            and pd.Timestamp(bounds[0]).normalize() == expected
            and pd.Timestamp(bounds[1]).normalize() == expected
        )

    @contextmanager
    def _build_lock(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".build.lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(
                f"Feature cache is already being built: {directory}"
            ) from error
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_manifest(path: Path, values: dict[str, object]) -> None:
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(values, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary.replace(path)

    @staticmethod
    def _materialize_missing(
        con: duckdb.DuckDBPyConnection,
        missing: list[tuple[pd.Timestamp, Path]],
        design: BenchmarkDesign,
        *,
        max_workers: int | None = None,
    ) -> None:
        """Materialize origin partitions concurrently on separate cursors.

        ``create_feature_tables`` leaves everything the feature query reads in
        the shared (non-temporary) schema, so each worker can run the query on
        its own cursor of the same database. Every worker writes only its own
        partition file, making the workers independent.
        """
        workers = max_workers if max_workers is not None else min(4, os.cpu_count() or 1)
        workers = max(1, min(int(workers), len(missing)))
        completed = 0
        progress_lock = threading.Lock()

        def build(item: tuple[pd.Timestamp, Path]) -> None:
            nonlocal completed
            origin, path = item
            cursor = con.cursor()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                materialize_features_for_origins(
                    cursor,
                    origins=[origin],
                    design=design,
                    feature_path=path,
                    return_frame=False,
                )
            finally:
                cursor.close()
            with progress_lock:
                completed += 1
                print(
                    f"[features {completed}/{len(missing)}] "
                    f"Materialized origin {origin.date()}",
                    flush=True,
                )

        if workers == 1:
            for item in missing:
                build(item)
            return
        print(
            f"[features] Materializing {len(missing)} origins with "
            f"{workers} parallel workers...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(build, item) for item in missing]
            for future in futures:
                future.result()

    def ensure(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        origins: pd.DatetimeIndex,
        design: BenchmarkDesign,
        rebuild: bool = False,
        max_workers: int | None = None,
    ) -> FeatureDataset:
        """Return valid partitions, materializing only absent or invalid origins."""
        normalized = pd.DatetimeIndex(origins).normalize().unique().sort_values()
        if len(normalized) == 0:
            raise ValueError("At least one feature origin is required")
        fingerprint, signature = self._source_signature(con, design)
        directory = self._generation_dir(fingerprint)
        manifest_path = directory / "manifest.json"
        paths = tuple(self._partition_path(directory, origin) for origin in normalized)
        manifest = self._read_manifest(manifest_path)
        signature_matches = bool(
            manifest
            and manifest.get("fingerprint") == fingerprint
            and manifest.get("signature") == signature
        )
        missing = [
            (origin, path)
            for origin, path in zip(normalized, paths, strict=True)
            if rebuild
            or not signature_matches
            or not self._partition_valid(con, path, origin)
        ]
        if missing:
            print(
                f"[features] Building {len(missing):,} of {len(paths):,} "
                "required origin partitions...",
                flush=True,
            )
            with self._build_lock(directory):
                # Another process may have completed partitions before the lock was won.
                if not rebuild:
                    missing = [
                        (origin, path)
                        for origin, path in missing
                        if not self._partition_valid(con, path, origin)
                    ]
                if missing:
                    create_feature_tables(con, origins=normalized, design=design)
                    self._materialize_missing(
                        con, missing, design, max_workers=max_workers
                    )
                available = sorted(
                    part.parent.name.removeprefix("origin=")
                    for part in directory.glob("origin=*/features.parquet")
                )
                self._write_manifest(
                    manifest_path,
                    {
                        "fingerprint": fingerprint,
                        "signature": signature,
                        "required_columns": sorted(_required_materialized_columns()),
                        "available_origins": available,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            print(f"[features] Feature cache ready: {directory}", flush=True)
        else:
            print(
                f"[features] Reusing {len(paths):,} cached origin partitions "
                f"from {directory}",
                flush=True,
            )
        return FeatureDataset(directory, manifest_path, paths, fingerprint)
