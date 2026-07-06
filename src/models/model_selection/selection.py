"""Pick one baseline model per demand cluster from precomputed metrics.

The selection rule:

1. Drop models that grossly under-forecast — their pooled forecast volume is
   below ``underforecast_ratio`` of pooled actuals. A trivial near-zero model
   has a low ``wape_median`` but is useless for restocking, so we filter it
   out before ranking.
2. From the survivors, pick the model with the lowest ``wape_median`` in each
   cluster. ``wape_pooled`` is reported alongside for context.
3. If a cluster has no survivors (every model under-forecasts), fall back to
   the full pool so we still emit a row and flag it.
"""
from __future__ import annotations

import pandas as pd

from src.models.model_selection.config import (
    DEFAULT_SELECTION_METRIC,
    DEFAULT_UNDERFORECAST_RATIO,
)

SELECTION_COLUMNS_EXTRA = (
    "selection_metric",
    "underforecast_ratio",
    "selected_after_underforecast_filter",
)


def flag_underforecasting(
    cluster_metrics: pd.DataFrame,
    underforecast_ratio: float = DEFAULT_UNDERFORECAST_RATIO,
) -> pd.DataFrame:
    """Add an ``is_underforecasting`` column to the cluster metrics frame."""
    if cluster_metrics.empty:
        return cluster_metrics.assign(is_underforecasting=pd.Series(dtype=bool))
    flagged = cluster_metrics.copy()
    flagged["is_underforecasting"] = (
        flagged["forecast_to_actual_ratio"] < underforecast_ratio
    )
    return flagged


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
    underforecast_ratio: float = DEFAULT_UNDERFORECAST_RATIO,
) -> pd.DataFrame:
    """Return the chosen baseline for every demand cluster."""
    if metric not in cluster_metrics.columns:
        raise KeyError(f"Metric {metric!r} not found in cluster metrics")

    if cluster_metrics.empty:
        return pd.DataFrame(
            columns=[*cluster_metrics.columns, *SELECTION_COLUMNS_EXTRA]
        )

    flagged = flag_underforecasting(cluster_metrics, underforecast_ratio)
    ranked: list[pd.Series] = []

    for cluster, cluster_df in flagged.groupby(
        "demand_class", observed=True, sort=True
    ):
        ranked.append(_select_for_cluster(cluster_df, metric, underforecast_ratio))

    selected = pd.DataFrame(ranked).reset_index(drop=True)
    selected["selection_metric"] = metric
    selected["underforecast_ratio"] = underforecast_ratio
    return selected


def _select_for_cluster(
    cluster_df: pd.DataFrame,
    metric: str,
    underforecast_ratio: float,
) -> pd.Series:
    """Pick one row from a single demand-class frame."""
    eligible = cluster_df[~cluster_df["is_underforecasting"]]
    sortable = eligible.dropna(subset=[metric])
    if sortable.empty:
        # Every survivor lacks the metric — fall back to the full pool so the
        # cluster still produces a row, but flag the fallback for the reader.
        sortable = cluster_df.dropna(subset=[metric])
        chosen = _pick_best_in_cluster(sortable, metric)
        chosen["selected_after_underforecast_filter"] = False
    else:
        chosen = _pick_best_in_cluster(sortable, metric)
        chosen["selected_after_underforecast_filter"] = True
    return chosen
