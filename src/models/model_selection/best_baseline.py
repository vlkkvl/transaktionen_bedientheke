"""Select the best baseline model per ADI/CV2 demand cluster.

Workflow:

1. Compute ADI/CV2 metrics from period-level product-store demand.
2. Pick series long enough for rolling-origin evaluation.
3. Slide the configured horizon across every eligible series and model.
4. Reduce window forecasts to per-series and per-cluster metrics.
5. Select the best model per demand class using the configured metric.

Each step lives in its own module — this file only orchestrates and writes
CSV artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.baseline.registry import DEMAND_CLASSES
from src.models.model_selection.config import (
    DEFAULT_DEMAND_COL,
    DEFAULT_FORECAST_PERIODS,
    DEFAULT_GROUP_COLS,
    DEFAULT_HORIZON,
    DEFAULT_MIN_DEMAND_PERIODS,
    DEFAULT_MIN_TRAIN_SIZE,
    DEFAULT_N_JOBS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SELECTION_METRIC,
    DEFAULT_STEP,
    DEFAULT_MAX_SERIES_PER_CLUSTER,
    SelectionConfig,
    SUPPORTED_HORIZONS,
)
from src.models.model_selection.data_loading import (
    compute_series_metrics,
    load_period_series,
    select_evaluation_keys,
)
from src.models.model_selection.metrics import (
    per_cluster_metrics,
    per_series_metrics,
)
from src.models.model_selection.selection import choose_best_by_cluster
from src.models.model_selection.sliding_window import evaluate_sliding_windows


@dataclass(frozen=True)
class BaselineSelectionResult:
    """Dataframes produced by the baseline selection workflow."""

    series_metrics: pd.DataFrame
    evaluation_keys: pd.DataFrame
    window_scores: pd.DataFrame
    series_scores: pd.DataFrame
    cluster_metrics: pd.DataFrame
    best_by_cluster: pd.DataFrame


def _print_progress(message: str) -> None:
    print(f"[best_baseline] {message}", flush=True)


def _count_forecast_windows(
    window_scores: pd.DataFrame,
    group_cols: tuple[str, ...],
) -> int:
    if window_scores.empty:
        return 0
    return len(window_scores[[*group_cols, "model", "window_index"]].drop_duplicates())


def _print_series_counts_by_cluster(evaluation_keys: pd.DataFrame) -> None:
    if evaluation_keys.empty:
        _print_progress("Selected series by demand class: none")
        return

    counts = evaluation_keys["demand_class"].value_counts().reindex(
        DEMAND_CLASSES,
        fill_value=0,
    )
    _print_progress("Selected series by demand class:")
    for demand_class, count in counts.items():
        _print_progress(f"  {demand_class}: {count:,}")


def run_baseline_selection(config: SelectionConfig) -> BaselineSelectionResult:
    """Run the complete cluster-wise baseline selection workflow."""
    period_label = config.period_label
    horizon_text = f"{config.forecast_periods} {period_label}"
    if config.forecast_periods != 1:
        horizon_text += "s"

    _print_progress("Computing ADI/CV2 series metrics ...")
    series_metrics = compute_series_metrics(
        data_dir=config.data_dir,
        demand_col=config.demand_col,
        group_cols=config.group_cols,
        min_demand_periods=config.min_demand_periods,
    )
    _print_progress(f"Classified {len(series_metrics):,} eligible series.")

    _print_progress(
        "Selecting evaluation series "
        f"(min_train_size={config.min_train_size}, horizon={horizon_text}) ..."
    )
    evaluation_keys = select_evaluation_keys(
        series_metrics=series_metrics,
        group_cols=config.group_cols,
        min_train_size=config.min_train_size,
        forecast_periods=config.forecast_periods,
        max_series_per_class=config.max_series_per_class,
    )
    _print_progress(f"Selected {len(evaluation_keys):,} series for evaluation.")
    _print_series_counts_by_cluster(evaluation_keys)

    _print_progress(f"Loading {config.horizon} demand rows ...")
    period_series = load_period_series(
        data_dir=config.data_dir,
        selected_keys=evaluation_keys,
        group_cols=config.group_cols,
        demand_col=config.demand_col,
    )
    _print_progress(f"Loaded {len(period_series):,} {config.horizon} rows.")

    _print_progress(
        "Evaluating baselines "
        f"(horizon={horizon_text}, step={config.step} {period_label}"
        f"{'s' if config.step != 1 else ''}, "
        f"n_jobs={config.n_jobs}) ..."
    )
    window_scores = evaluate_sliding_windows(
        period_series=period_series,
        group_cols=config.group_cols,
        min_train_size=config.min_train_size,
        step=config.step,
        forecast_periods=config.forecast_periods,
        model_names=config.model_names,
        n_jobs=config.n_jobs,
        show_progress=True,
        progress_label="[best_baseline] Evaluating baselines",
    )
    _print_progress(
        f"Generated {len(window_scores):,} forecast rows from "
        f"{_count_forecast_windows(window_scores, config.group_cols):,} "
        "model windows."
    )

    _print_progress("Aggregating per-series metrics ...")
    series_scores = per_series_metrics(window_scores, group_cols=config.group_cols)
    _print_progress(f"Built {len(series_scores):,} series/model score rows.")

    _print_progress("Aggregating per-cluster metrics ...")
    cluster_metrics = per_cluster_metrics(series_scores)
    _print_progress(f"Built {len(cluster_metrics):,} cluster/model score rows.")

    _print_progress("Selecting best baseline per demand class ...")
    best_by_cluster = choose_best_by_cluster(
        cluster_metrics,
        metric=config.metric,
    )
    _print_progress(f"Selected {len(best_by_cluster):,} demand-class winners.")
    return BaselineSelectionResult(
        series_metrics=series_metrics,
        evaluation_keys=evaluation_keys,
        window_scores=window_scores,
        series_scores=series_scores,
        cluster_metrics=cluster_metrics,
        best_by_cluster=best_by_cluster,
    )


def write_results(
    result: BaselineSelectionResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Persist every artifact as a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.evaluation_keys.to_csv(
        output_dir / "baseline_evaluation_keys.csv", index=False
    )
    result.window_scores.to_csv(
        output_dir / "baseline_window_scores.csv", index=False
    )
    result.series_scores.to_csv(
        output_dir / "baseline_series_scores.csv", index=False
    )
    result.cluster_metrics.to_csv(
        output_dir / "baseline_model_summary.csv", index=False
    )
    result.best_by_cluster.to_csv(
        output_dir / "baseline_best_by_cluster.csv", index=False
    )


def _parse_group_cols(value: str) -> tuple[str, ...]:
    group_cols = tuple(col.strip() for col in value.split(",") if col.strip())
    if not group_cols:
        raise argparse.ArgumentTypeError("at least one grouping column is required")
    return group_cols


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        choices=SUPPORTED_HORIZONS,
        default=DEFAULT_HORIZON,
        help="Aggregation horizon to evaluate.",
    )
    parser.add_argument(
        "--forecast-periods",
        type=int,
        default=DEFAULT_FORECAST_PERIODS,
        help="Number of future periods to forecast for each sliding window.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Processed parquet directory. Defaults to the directory implied by "
            "--horizon."
        ),
    )
    parser.add_argument("--demand-col", default=DEFAULT_DEMAND_COL)
    parser.add_argument(
        "--group-cols", type=_parse_group_cols, default=DEFAULT_GROUP_COLS
    )
    parser.add_argument(
        "--min-demand-periods", type=int, default=DEFAULT_MIN_DEMAND_PERIODS
    )
    parser.add_argument(
        "--min-demand-weeks",
        dest="min_demand_periods",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=DEFAULT_MIN_TRAIN_SIZE,
        help="Minimum training periods before the first sliding-window forecast.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=(
            "Stride between successive sliding-window origins. Defaults to "
            f"{DEFAULT_STEP} from config.DEFAULT_STEP."
        ),
    )
    parser.add_argument(
        "--max-series-per-class",
        type=int,
        default=DEFAULT_MAX_SERIES_PER_CLUSTER,
        help=(
            "Cap for faster experiments. Keeps the most recent N eligible "
            f"series per demand class. Defaults to {DEFAULT_MAX_SERIES_PER_CLUSTER}."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Number of worker processes for per-series baseline evaluation. "
            "Use 1 to run sequentially."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("wape_pooled", "wape_median", "rmse_mean"),
        default=DEFAULT_SELECTION_METRIC,
    )
    parser.add_argument(
        "--model",
        dest="model_names",
        action="append",
        help="Optional baseline model name. Repeat to restrict the registry pool.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SelectionConfig(
        horizon=args.horizon,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        demand_col=args.demand_col,
        group_cols=args.group_cols,
        min_demand_periods=args.min_demand_periods,
        min_train_size=args.min_train_size,
        forecast_periods=args.forecast_periods,
        step=args.step,
        n_jobs=args.n_jobs,
        max_series_per_class=args.max_series_per_class,
        metric=args.metric,
        model_names=tuple(args.model_names) if args.model_names else None,
    )
    result = run_baseline_selection(config)
    write_results(result, config.output_dir)

    print(f"Classified series: {len(result.series_metrics):,}")
    print(f"Evaluated series:  {len(result.evaluation_keys):,}")
    print(f"Forecast rows:     {len(result.window_scores):,}")
    print(f"Horizon:          {config.horizon}")
    print(f"Data directory:   {config.data_dir}")
    print(
        "Model windows:     "
        f"{_count_forecast_windows(result.window_scores, config.group_cols):,}"
    )
    print(f"Wrote results to {config.output_dir}")
    if not result.best_by_cluster.empty:
        columns = ["demand_class", "model", "wape_median", "wape_pooled"]
        print(result.best_by_cluster[columns].to_string(index=False))


if __name__ == "__main__":
    main()
