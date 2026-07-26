"""Runner specification for the weekly-total model."""
from __future__ import annotations

from src.models.core.contracts import ModelSpec
from src.models.lightgbm.weekly_total.model import (
    WEEKLY_MODEL_NAME,
    WeeklyTotalConfig,
    fit,
)


SPEC = ModelSpec(
    name=WEEKLY_MODEL_NAME,
    family="lightgbm",
    config_factory=WeeklyTotalConfig,
    fit=fit,
)
