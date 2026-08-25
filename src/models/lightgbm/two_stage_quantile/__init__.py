"""Occurrence-times-quantile-quantity LightGBM model."""

from src.models.lightgbm.two_stage_quantile.model import TwoStageQuantileConfig
from src.models.lightgbm.two_stage_quantile.spec import SPEC

__all__ = ["SPEC", "TwoStageQuantileConfig"]
