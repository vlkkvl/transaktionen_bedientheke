"""Runner specification for the daily Tweedie model."""
from __future__ import annotations

from src.models.core.contracts import ModelSpec
from src.models.lightgbm.tweedie.model import (
    TWEEDIE_MODEL_NAME,
    TweedieConfig,
    fit,
)


SPEC = ModelSpec(
    name=TWEEDIE_MODEL_NAME,
    family="lightgbm",
    config_factory=TweedieConfig,
    fit=fit,
)
