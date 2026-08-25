"""Runner specification for the two-stage quantile model."""
from __future__ import annotations

from src.models.core.contracts import ModelSpec
from src.models.lightgbm.two_stage_quantile.model import (
    TWO_STAGE_QUANTILE_MODEL_NAME,
    TwoStageQuantileConfig,
    fit,
)


SPEC = ModelSpec(
    name=TWO_STAGE_QUANTILE_MODEL_NAME,
    family="lightgbm",
    config_factory=TwoStageQuantileConfig,
    fit=fit,
)
