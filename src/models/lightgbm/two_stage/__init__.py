"""Occurrence-times-quantity LightGBM model."""

from src.models.lightgbm.two_stage.model import TwoStageConfig
from src.models.lightgbm.two_stage.spec import SPEC

__all__ = ["SPEC", "TwoStageConfig"]
