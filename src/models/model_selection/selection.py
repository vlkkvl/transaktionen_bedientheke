"""Pick one baseline model per demand cluster from precomputed metrics.

The selection rule:

1. Pick the model with the lowest configured metric in each demand cluster.
2. Report the selected metric alongside the chosen model.
"""
from __future__ import annotations

import pandas as pd

from src.models.model_selection.config import DEFAULT_SELECTION_METRIC

SELECTION_COLUMNS_EXTRA = ("selection_metric",)


def _pick_best_in_cluster(
    candidates: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """Lowest-metric row within an already cluster-scoped frame."""
    tie_breakers = [
        col for col in ("wape_pooled", "mae_mean", "model") if col != metric
    ]
    ordered = candidates.sort_values([metric, *tie_breakers])
    return ordered.iloc[0]


def choose_best_by_cluster(
    cluster_metrics: pd.DataFrame,
    metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    """Return the chosen baseline for every demand cluster."""
    if metric not in cluster_metrics.columns:
        raise KeyError(f"Metric {metric!r} not found in cluster metrics")

    if cluster_metrics.empty:
        return pd.DataFrame(
            columns=[*cluster_metrics.columns, *SELECTION_COLUMNS_EXTRA]
        )

    ranked: list[pd.Series] = []

    for cluster, cluster_df in cluster_metrics.groupby(
        "demand_class", observed=True, sort=True
    ):
        ranked.append(_select_for_cluster(cluster_df, metric))

    selected = pd.DataFrame(ranked).reset_index(drop=True)
    selected["selection_metric"] = metric
    return selected


def _select_for_cluster(
    cluster_df: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """Pick one row from a single demand-class frame."""
    sortable = cluster_df.dropna(subset=[metric])
    if sortable.empty:
        raise ValueError(
            f"No non-null {metric!r} values for demand class "
            f"{cluster_df['demand_class'].iloc[0]!r}"
        )
    return _pick_best_in_cluster(sortable, metric)
