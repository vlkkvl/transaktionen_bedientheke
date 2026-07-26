"""Single- and multi-model runner for registered LightGBM specifications."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import duckdb
import pandas as pd

from src.models.benchmark import load_benchmark_design
from src.models.benchmark.config import BenchmarkDesign, DEFAULT_DESIGN_PATH
from src.models.benchmark.evaluation import (
    _create_assessed_origins,
    prepare_daily_rows,
)
from src.models.benchmark.models import create_history_features
from src.models.core.contracts import ModelSpec
from src.models.lightgbm.config import LightGBMModelConfig
from src.models.lightgbm.features import (
    DEFAULT_FEATURE_STORE_DIR,
    LightGBMFeatureStore,
)
from src.models.lightgbm.features.builder import (
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
    _feature_cache_covers,
    create_feature_tables,
    historical_training_origins,
    iter_lightgbm_origin_windows,
    materialize_features_for_origins,
)
from src.models.results import (
    DEFAULT_RESULTS_DIR,
    result_path,
    write_result,
)


def _validate_design(
    design: BenchmarkDesign,
    config: GlobalLightGBMConfig,
    evaluation_origins: pd.DatetimeIndex,
) -> None:
    if design.origin_spacing_days != config.origin_spacing_days:
        raise ValueError(
            "LightGBM features and forecasts must use the shared origin spacing"
        )
    if design.forecast_horizon_days != config.origin_spacing_days:
        raise ValueError("The LightGBM forecast horizon must be one full week")
    if design.max_origins != config.test_origins:
        raise ValueError(
            "Forecast design and model design configure different test-origin counts"
        )
    if len(evaluation_origins) != config.test_origins:
        raise RuntimeError(
            f"Required {config.test_origins} complete origins, "
            f"found {len(evaluation_origins)}"
        )


def _load_origins(
    con: duckdb.DuckDBPyConnection,
    source: Path | Sequence[Path],
    origins: pd.DatetimeIndex,
    *,
    active_only: bool,
) -> pd.DataFrame:
    con.register(
        "runner_requested_feature_origins",
        pd.DataFrame({"origin": pd.DatetimeIndex(origins)}),
    )
    parquet_source: str | list[str] = (
        str(source)
        if isinstance(source, Path)
        else [str(path) for path in source]
    )
    active_filter = "WHERE f.is_active" if active_only else ""
    return con.execute(
        f"""
        SELECT f.*
        FROM read_parquet(?, hive_partitioning=false) AS f
        INNER JOIN runner_requested_feature_origins AS requested USING (origin)
        {active_filter}
        ORDER BY f.origin, f.ARTIKEL_ID, f.MARKT_ID, f.period
        """,
        [parquet_source],
    ).fetchdf()


def iter_run_frames(
    con: duckdb.DuckDBPyConnection,
    design: BenchmarkDesign,
    config: GlobalLightGBMConfig,
    *,
    data_dir: Path | None = None,
    feature_dataset_path: Path | None = None,
    rebuild_features: bool = False,
) -> Iterator[GlobalLightGBMFrames]:
    """Yield shared expanding-window frames used by every LightGBM model."""
    resolved_data_dir = design.data_dir if data_dir is None else data_dir
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "benchmark_daily_rows" not in tables:
        prepare_daily_rows(con, data_dir=resolved_data_dir)
    last_observed = con.execute(
        "SELECT MAX(period) FROM benchmark_daily_rows"
    ).fetchone()[0]
    evaluation_origins = design.origins_through(last_observed)
    _validate_design(design, config, evaluation_origins)

    create_history_features(con)
    _create_assessed_origins(con, evaluation_origins, design)
    con.execute("DROP TABLE IF EXISTS benchmark_row_features")
    con.execute("DROP TABLE IF EXISTS benchmark_weekday_features")
    con.execute("DROP TABLE IF EXISTS benchmark_positive_features")

    initial_origins = historical_training_origins(
        con, design, evaluation_origins.min(), config.history_origins
    )
    history_origins = initial_origins.append(evaluation_origins[:-1])
    history_origins = history_origins.unique().sort_values()
    required_origins = history_origins.append(evaluation_origins)
    required_origins = required_origins.unique().sort_values()

    requested_path = (
        None if feature_dataset_path is None else Path(feature_dataset_path)
    )
    legacy_file = bool(
        requested_path is not None
        and requested_path.suffix.lower() in {".parquet", ".pq"}
    )
    if legacy_file:
        assert requested_path is not None
        feature_source: Path | Sequence[Path] = requested_path
        if rebuild_features or not _feature_cache_covers(
            con, requested_path, required_origins
        ):
            create_feature_tables(con)
            materialize_features_for_origins(
                con,
                origins=required_origins,
                design=design,
                feature_path=requested_path,
                return_frame=False,
            )
    else:
        store = LightGBMFeatureStore(
            DEFAULT_FEATURE_STORE_DIR if requested_path is None else requested_path
        )
        feature_source = store.ensure(
            con,
            origins=required_origins,
            design=design,
            rebuild=rebuild_features,
        ).parquet_paths

    for window in iter_lightgbm_origin_windows(
        initial_origins, evaluation_origins, config
    ):
        training = _load_origins(
            con, feature_source, window.training, active_only=True
        )
        validation = _load_origins(
            con, feature_source, window.validation, active_only=True
        )
        evaluation = _load_origins(
            con, feature_source, window.evaluation, active_only=False
        )
        if training.empty or validation.empty or evaluation.empty:
            raise RuntimeError("Training, validation, and evaluation must be nonempty")
        yield GlobalLightGBMFrames(
            training=training,
            validation=validation,
            evaluation=evaluation,
            training_origins=window.training.append(window.validation),
            evaluation_origin=pd.Timestamp(window.evaluation.min()),
        )


def prepare_run_frames(
    con: duckdb.DuckDBPyConnection,
    design: BenchmarkDesign,
    config: GlobalLightGBMConfig,
    *,
    data_dir: Path | None = None,
    feature_dataset_path: Path | None = None,
    rebuild_features: bool = False,
) -> GlobalLightGBMFrames:
    """Collect shared streaming frames for workloads such as feature ablation."""
    parts = tuple(
        iter_run_frames(
            con,
            design,
            config,
            data_dir=data_dir,
            feature_dataset_path=feature_dataset_path,
            rebuild_features=rebuild_features,
        )
    )
    if not parts:
        raise RuntimeError("No LightGBM origin frames were prepared")
    return GlobalLightGBMFrames(
        training=parts[0].training,
        validation=parts[0].validation,
        evaluation=pd.concat([part.evaluation for part in parts], ignore_index=True),
        training_origins=parts[0].training_origins,
        evaluation_origin=parts[0].evaluation_origin,
        origin_frames=parts,
    )


class _StreamingResultWriter:
    """Append per-origin model artifacts and publish them atomically."""

    def __init__(
        self,
        model_name: str,
        design: BenchmarkDesign,
        results_dir: Path | str,
    ) -> None:
        self.model_name = model_name
        self.design = design
        self.results_dir = results_dir
        self.paths = {
            artifact: result_path(
                model_name,
                design,
                artifact=artifact,
                results_dir=results_dir,
            )
            for artifact in (
                "forecasts",
                "training_summaries",
                "feature_importance",
            )
        }
        self.started: set[str] = set()

    def _append(self, artifact: str, frame: pd.DataFrame) -> None:
        path = self.paths[artifact]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        first = artifact not in self.started
        frame.to_csv(
            temporary,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        self.started.add(artifact)

    def append(self, result: Any) -> None:
        self._append("forecasts", result.forecasts)
        self._append("training_summaries", result.training_summary)
        importance = result.feature_importance.copy()
        if "evaluation_origin" not in importance:
            importance.insert(
                0,
                "evaluation_origin",
                pd.Timestamp(result.training_summary.iloc[0]["evaluation_origin"]),
            )
        self._append("feature_importance", importance)
        if result.allocation_audit is not None:
            artifact = "allocation_audits"
            if artifact not in self.paths:
                self.paths[artifact] = result_path(
                    self.model_name,
                    self.design,
                    artifact=artifact,
                    results_dir=self.results_dir,
                )
            self._append(artifact, result.allocation_audit)

    def publish(self) -> list[Path]:
        paths = []
        for artifact in self.started:
            path = self.paths[artifact]
            path.with_suffix(f"{path.suffix}.tmp").replace(path)
            paths.append(path)
        return paths


def _config_record(model: str, config: object) -> pd.DataFrame:
    values = asdict(config) if is_dataclass(config) else {"value": repr(config)}
    encoded = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return pd.DataFrame(
        [
            {
                "model": model,
                "config_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "config_json": encoded,
            }
        ]
    )


def run_specs(
    specs: Sequence[ModelSpec[Any]],
    *,
    design_path: Path = DEFAULT_DESIGN_PATH,
    data_dir: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    feature_dataset_path: Path | None = None,
    rebuild_features: bool = False,
    configs: Mapping[str, LightGBMModelConfig] | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> list[Path]:
    """Run one or more models over the same prepared origin frames."""
    if not specs:
        raise ValueError("At least one LightGBM model specification is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("LightGBM model specifications must have unique names")

    resolved_configs: dict[str, LightGBMModelConfig] = {}
    supplied = {} if configs is None else dict(configs)
    unknown = sorted(set(supplied) - set(names))
    if unknown:
        raise KeyError(f"Configurations supplied for unknown models: {', '.join(unknown)}")
    for spec in specs:
        config = supplied.get(spec.name, spec.make_config())
        if not isinstance(config, LightGBMModelConfig):
            raise TypeError(f"{spec.name} requires a LightGBMModelConfig")
        resolved_configs[spec.name] = config

    windows = {config.window for config in resolved_configs.values()}
    if len(windows) != 1:
        raise ValueError(
            "Models in one family run must share forecast-window settings; "
            "runtime and statistical hyperparameters remain independent."
        )
    execution_config = next(iter(resolved_configs.values())).execution_config()
    design = load_benchmark_design(design_path)
    owns_connection = connection is None
    con = duckdb.connect() if connection is None else connection
    con.execute("SET temp_directory='/tmp/ba_lightgbm_family_duckdb'")
    con.execute(
        f"SET threads={max(config.runtime.num_threads for config in resolved_configs.values())}"
    )
    con.execute("SET preserve_insertion_order=false")
    writers = {
        spec.name: _StreamingResultWriter(spec.name, design, results_dir)
        for spec in specs
    }
    try:
        refit_count = math.ceil(
            execution_config.test_origins
            / execution_config.refit_interval_origins
        )
        for position, frames in enumerate(
            iter_run_frames(
                con,
                design,
                execution_config,
                data_dir=data_dir,
                feature_dataset_path=feature_dataset_path,
                rebuild_features=rebuild_features,
            ),
            start=1,
        ):
            origin = pd.to_datetime(frames.evaluation["origin"]).min().date()
            for spec in specs:
                print(
                    f"[{position}/{refit_count}] Fitting {spec.name} at {origin} "
                    f"({len(frames.training):,} training rows)...",
                    flush=True,
                )
                result = spec.fit(frames, resolved_configs[spec.name])
                writers[spec.name].append(result)
        paths: list[Path] = []
        for spec in specs:
            paths.extend(writers[spec.name].publish())
            paths.append(
                write_result(
                    _config_record(spec.name, resolved_configs[spec.name]),
                    spec.name,
                    design,
                    artifact="run_configs",
                    results_dir=results_dir,
                )
            )
        return paths
    finally:
        if owns_connection:
            con.close()
