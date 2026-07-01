"""Registry for baseline forecast models and demand-class routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.models.baseline.base import ForecastModel
from src.models.baseline.croston import CrostonForecast
from src.models.baseline.damped_trend_ets import DampedTrendETSForecast
from src.models.baseline.mean_forecast import MeanForecast
from src.models.baseline.moving_average import MovingAverageForecast
from src.models.baseline.naive import NaiveForecast
from src.models.baseline.sba import SBAForecast
from src.models.baseline.simple_exponential_smoothing import (
    SimpleExponentialSmoothingForecast,
)
from src.models.baseline.tsb import TSBForecast

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49

DEMAND_CLASSES = ("smooth", "erratic", "intermittent", "lumpy")
REGULAR_DEMAND_CLASSES = ("smooth", "erratic")
SPARSE_DEMAND_CLASSES = ("intermittent", "lumpy")


@dataclass(frozen=True)
class ModelSpec:
    """Factory and routing metadata for one baseline model."""

    name: str
    factory: Callable[[], ForecastModel]
    demand_classes: tuple[str, ...]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "naive": ModelSpec("naive", NaiveForecast, DEMAND_CLASSES),
    "mean": ModelSpec("mean", MeanForecast, DEMAND_CLASSES),
    "moving_average_4": ModelSpec(
        "moving_average_4", lambda: MovingAverageForecast(window=4), DEMAND_CLASSES
    ),
    "simple_exponential_smoothing": ModelSpec(
        "simple_exponential_smoothing",
        SimpleExponentialSmoothingForecast,
        REGULAR_DEMAND_CLASSES,
    ),
    "damped_trend_ets": ModelSpec(
        "damped_trend_ets", DampedTrendETSForecast, REGULAR_DEMAND_CLASSES
    ),
    "croston": ModelSpec("croston", CrostonForecast, SPARSE_DEMAND_CLASSES),
    "sba": ModelSpec("sba", SBAForecast, SPARSE_DEMAND_CLASSES),
    "tsb": ModelSpec("tsb", TSBForecast, SPARSE_DEMAND_CLASSES),
}

# Edit this tuple to enable or disable baselines globally.
DEFAULT_ACTIVE_MODEL_NAMES = tuple(MODEL_REGISTRY)

# Edit this mapping to change which active baselines are evaluated per cluster.
CLASS_MODEL_NAMES: dict[str, tuple[str, ...]] = {
    "smooth": (
        "naive",
        "mean",
        "moving_average_4",
        "simple_exponential_smoothing",
        "damped_trend_ets",
    ),
    "erratic": (
        "naive",
        "mean",
        "moving_average_4",
        "simple_exponential_smoothing",
        "damped_trend_ets",
    ),
    "intermittent": (
        "naive",
        "mean",
        "moving_average_4",
        "croston",
        "sba",
        "tsb",
    ),
    "lumpy": (
        "naive",
        "mean",
        "moving_average_4",
        "croston",
        "sba",
        "tsb",
    ),
}


def classify_demand(adi: float, cv2: float) -> str:
    """Classify a demand series by ADI and squared coefficient of variation."""
    if not np.isfinite(adi) or not np.isfinite(cv2):
        return "unclassified"
    if adi < ADI_CUTOFF and cv2 < CV2_CUTOFF:
        return "smooth"
    if adi < ADI_CUTOFF and cv2 >= CV2_CUTOFF:
        return "erratic"
    if adi >= ADI_CUTOFF and cv2 < CV2_CUTOFF:
        return "intermittent"
    return "lumpy"


def add_demand_class(
    df: pd.DataFrame,
    adi_col: str = "ADI",
    cv2_col: str = "CV2",
    output_col: str = "demand_class",
) -> pd.DataFrame:
    """Return a copy of ``df`` with an ADI/CV2 demand-class column."""
    classified = df.copy()
    classified[output_col] = [
        classify_demand(adi, cv2)
        for adi, cv2 in zip(classified[adi_col], classified[cv2_col])
    ]
    return classified


def get_model_specs(
    demand_class: str | None = None,
    model_names: Iterable[str] | None = None,
) -> list[ModelSpec]:
    """Return model specs enabled for an optional demand class."""
    requested_names = (
        tuple(model_names) if model_names is not None else DEFAULT_ACTIVE_MODEL_NAMES
    )
    unknown = sorted(set(requested_names) - set(MODEL_REGISTRY))
    if unknown:
        raise KeyError(f"Unknown baseline model names: {unknown}")

    if demand_class is None:
        eligible_names = requested_names
    else:
        if demand_class not in CLASS_MODEL_NAMES:
            raise KeyError(f"Unknown demand class: {demand_class!r}")
        class_names = set(CLASS_MODEL_NAMES[demand_class])
        eligible_names = tuple(name for name in requested_names if name in class_names)

    return [MODEL_REGISTRY[name] for name in eligible_names]


def build_models(
    demand_class: str | None = None,
    model_names: Iterable[str] | None = None,
) -> list[ForecastModel]:
    """Instantiate enabled baseline models for an optional demand class."""
    return [spec.factory() for spec in get_model_specs(demand_class, model_names)]
