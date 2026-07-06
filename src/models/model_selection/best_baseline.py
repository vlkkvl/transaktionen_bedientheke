"""Select the best baseline model per ADI/CV2 demand cluster.

Workflow:

1. Compute ADI/CV2 metrics from weekly product-store demand.
2. Pick series long enough for rolling-origin evaluation.
3. Slide a one-week horizon across every eligible series and model.
4. Reduce window forecasts to per-series and per-cluster metrics.
5. Drop models that grossly under-forecast, then select on ``wape_median``.

Each step lives in its own module — this file only orchestrates and writes
CSV artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.model_selection.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_DEMAND_COL,
    DEFAULT_GROUP_COLS,
    DEFAULT_MIN_DEMAND_WEEKS,
    DEFAULT_MIN_TRAIN_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SELECTION_METRIC,
    DEFAULT_STEP,
    DEFAULT_UNDERFORECAST_RATIO,
    SelectionConfig,
)
from src.models.model_selection.data_loading import (
    compute_series_metrics,
    load_weekly_series,
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


def run_baseline_selection(config: SelectionConfig) -> BaselineSelectionResult:
    """Run the complete cluster-wise baseline selection workflow."""
    series_metrics = compute_series_metrics(
        data_dir=config.data_dir,
        demand_col=config.demand_col,
        group_cols=config.group_cols,
        min_demand_weeks=config.min_demand_weeks,
    )
    evaluation_keys = select_evaluation_keys(
        series_metrics=series_metrics,
        group_cols=config.group_cols,
        min_train_size=config.min_train_size,
        max_series_per_class=config.max_series_per_class,
        random_state=config.random_state,
    )
    weekly_series = load_weekly_series(
        data_dir=config.data_dir,
        selected_keys=evaluation_keys,
        group_cols=config.group_cols,
        demand_col=config.demand_col,
    )
    window_scores = evaluate_sliding_windows(
        weekly_series=weekly_series,
        group_cols=config.group_cols,
        min_train_size=config.min_train_size,
        step=config.step,
        model_names=config.model_names,
    )
    series_scores = per_series_metrics(window_scores, group_cols=config.group_cols)
    cluster_metrics = per_cluster_metrics(series_scores)
    best_by_cluster = choose_best_by_cluster(
        cluster_metrics,
        metric=config.metric,
        underforecast_ratio=config.underforecast_ratio,
    )
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--demand-col", default=DEFAULT_DEMAND_COL)
    parser.add_argument(
        "--group-cols", type=_parse_group_cols, default=DEFAULT_GROUP_COLS
    )
    parser.add_argument(
        "--min-demand-weeks", type=int, default=DEFAULT_MIN_DEMAND_WEEKS
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=DEFAULT_MIN_TRAIN_SIZE,
        help="Minimum training weeks before the first sliding-window forecast.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help="Stride (in weeks) between successive sliding-window origins.",
    )
    parser.add_argument(
        "--max-series-per-class",
        type=int,
        default=None,
        help="Optional cap for faster experiments; omit to evaluate all eligible series.",
    )
    parser.add_argument(
        "--metric",
        choices=("wape_median", "wape_pooled", "mae_mean", "mae_median", "rmse_mean"),
        default=DEFAULT_SELECTION_METRIC,
    )
    parser.add_argument(
        "--underforecast-ratio",
        type=float,
        default=DEFAULT_UNDERFORECAST_RATIO,
        help=(
            "Drop models whose pooled forecast/actual ratio falls below this "
            "value before selecting on the chosen metric."
        ),
    )
    parser.add_argument(
        "--model",
        dest="model_names",
        action="append",
        help="Optional baseline model name. Repeat to restrict the registry pool.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SelectionConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        demand_col=args.demand_col,
        group_cols=args.group_cols,
        min_demand_weeks=args.min_demand_weeks,
        min_train_size=args.min_train_size,
        step=args.step,
        max_series_per_class=args.max_series_per_class,
        metric=args.metric,
        underforecast_ratio=args.underforecast_ratio,
        model_names=tuple(args.model_names) if args.model_names else None,
        random_state=args.random_state,
    )
    result = run_baseline_selection(config)
    write_results(result, config.output_dir)

    print(f"Classified series: {len(result.series_metrics):,}")
    print(f"Evaluated series:  {len(result.evaluation_keys):,}")
    print(f"Forecast windows:  {len(result.window_scores):,}")
    print(f"Wrote results to {config.output_dir}")
    if not result.best_by_cluster.empty:
        columns = ["demand_class", "model", "wape_median", "wape_pooled"]
        print(result.best_by_cluster[columns].to_string(index=False))


if __name__ == "__main__":
    main()
