"""Runner specification for the daily L2 model."""
from __future__ import annotations

from src.models.core.contracts import ModelSpec
from src.models.lightgbm.l2.model import L2Config, MODEL_NAME, fit


SPEC = ModelSpec(name=MODEL_NAME, family="lightgbm", config_factory=L2Config, fit=fit)
