"""Stable CSV result paths shared by model runners and analysis notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.models.benchmark.config import BenchmarkDesign


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "reports" / "results"
RESULT_GRAIN = "artikel_markt"

MODEL_FILE_NAMES = {
    "global_lightgbm": "lightgbm_l2",
    "global_lightgbm_tweedie_daily": "lightgbm_tweedie",
    "global_lightgbm_two_stage": "lightgbm_two_stage",
    "global_lightgbm_two_stage_quantile": "lightgbm_two_stage_quantile",
    "global_lightgbm_weekly_total": "lightgbm_weekly_total",
}

DATE_COLUMNS = {
    "evaluation_end",
    "evaluation_origin",
    "first_date",
    "last_date",
    "origin",
    "period",
    "training_end",
    "training_start",
    "validation_end",
    "validation_start",
}


@dataclass(frozen=True)
class PersistedModelResult:
    """Model outputs loaded from the report CSVs."""

    forecasts: pd.DataFrame
    training_summary: pd.DataFrame
    feature_importance: pd.DataFrame
    allocation_audit: pd.DataFrame | None = None
    validation_predictions: pd.DataFrame | None = None


def model_file_name(model: str) -> str:
    """Return the public, filesystem-safe model name."""
    if model in MODEL_FILE_NAMES:
        return MODEL_FILE_NAMES[model]
    if model.startswith("tweedie_ablation_"):
        return f"lightgbm_{model}"
    return model


def result_path(
    model: str,
    design: BenchmarkDesign,
    *,
    artifact: str = "forecasts",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> Path:
    """Build ``grain_scope_model.csv`` under the requested artifact folder."""
    scope = f"multi{design.forecast_horizon_days}days"
    filename = f"{RESULT_GRAIN}_{scope}_{model_file_name(model)}.csv"
    return Path(results_dir) / artifact / filename


def write_result(
    frame: pd.DataFrame,
    model: str,
    design: BenchmarkDesign,
    *,
    artifact: str = "forecasts",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> Path:
    """Write one model artifact and return its path."""
    path = result_path(
        model, design, artifact=artifact, results_dir=results_dir
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)
    return path


def write_forecasts(
    forecasts: pd.DataFrame,
    design: BenchmarkDesign,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict[str, Path]:
    """Write a separate forecast CSV for each model in a long result frame."""
    if "model" not in forecasts:
        raise KeyError("Forecast results must contain a model column")
    paths = {}
    for model, frame in forecasts.groupby("model", observed=True, sort=False):
        model_name = str(model)
        paths[model_name] = write_result(
            frame.reset_index(drop=True),
            model_name,
            design,
            results_dir=results_dir,
        )
    return paths


def read_result(
    model: str,
    design: BenchmarkDesign,
    *,
    artifact: str = "forecasts",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Read one result CSV and restore its date columns."""
    path = result_path(
        model, design, artifact=artifact, results_dir=results_dir
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Model result not found: {path}. Run the corresponding "
            "model package's main module first."
        )
    frame = pd.read_csv(path)
    for column in DATE_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def read_forecasts(
    models: Iterable[str],
    design: BenchmarkDesign,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Read and combine forecast CSVs for the requested model names."""
    frames = [
        read_result(model, design, results_dir=results_dir) for model in models
    ]
    return pd.concat(frames, ignore_index=True)


def load_benchmark_result(
    design: BenchmarkDesign,
    *,
    include_extended_baselines: bool = False,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
):
    """Load the persisted benchmark result in the runtime result container."""
    from src.models.baseline.evaluation import BASELINE_MODEL_LABELS
    from src.models.benchmark.evaluation import BenchmarkResult
    from src.models.benchmark.models import MODEL_COLUMNS

    models = list(MODEL_COLUMNS)
    if include_extended_baselines:
        models.extend(BASELINE_MODEL_LABELS)
    return BenchmarkResult(
        design=design,
        forecasts=read_forecasts(models, design, results_dir=results_dir),
        origin_summary=read_result(
            "benchmark_origin_summary",
            design,
            artifact="diagnostics",
            results_dir=results_dir,
        ),
        data_audit=read_result(
            "benchmark_data_audit",
            design,
            artifact="diagnostics",
            results_dir=results_dir,
        ),
    )


def load_lightgbm_results(
    design: BenchmarkDesign,
    model_names: Iterable[str],
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> Mapping[str, PersistedModelResult]:
    """Load forecasts and diagnostics for fitted LightGBM architectures."""
    loaded = {}
    for model in model_names:
        allocation_path = result_path(
            model,
            design,
            artifact="allocation_audits",
            results_dir=results_dir,
        )
        validation_predictions_path = result_path(
            model,
            design,
            artifact="validation_predictions",
            results_dir=results_dir,
        )
        loaded[model] = PersistedModelResult(
            forecasts=read_result(model, design, results_dir=results_dir),
            training_summary=read_result(
                model,
                design,
                artifact="training_summaries",
                results_dir=results_dir,
            ),
            feature_importance=read_result(
                model,
                design,
                artifact="feature_importance",
                results_dir=results_dir,
            ),
            allocation_audit=(
                read_result(
                    model,
                    design,
                    artifact="allocation_audits",
                    results_dir=results_dir,
                )
                if allocation_path.exists()
                else None
            ),
            validation_predictions=(
                read_result(
                    model,
                    design,
                    artifact="validation_predictions",
                    results_dir=results_dir,
                )
                if validation_predictions_path.exists()
                else None
            ),
        )
    return loaded


def load_ablation_results(
    design: BenchmarkDesign,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, Mapping[str, PersistedModelResult]]:
    """Load the ablation design and every model named by that design."""
    feature_design = read_result(
        "lightgbm_tweedie_ablation_design",
        design,
        artifact="diagnostics",
        results_dir=results_dir,
    )
    models = feature_design["model"].astype(str).tolist()
    return feature_design, load_lightgbm_results(
        design, models, results_dir=results_dir
    )
