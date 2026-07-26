"""Runner specification for the two-stage model."""
from __future__ import annotations

from src.models.core.contracts import ModelSpec
from src.models.lightgbm.two_stage.model import (
    TWO_STAGE_MODEL_NAME,
    TwoStageConfig,
    fit,
)


SPEC = ModelSpec(
    name=TWO_STAGE_MODEL_NAME,
    family="lightgbm",
    config_factory=TwoStageConfig,
    fit=fit,
)
