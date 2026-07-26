"""Direct daily Tweedie LightGBM model."""

from src.models.lightgbm.tweedie.model import TweedieConfig
from src.models.lightgbm.tweedie.spec import SPEC

__all__ = ["SPEC", "TweedieConfig"]
