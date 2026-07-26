"""Weekly-total LightGBM model with daily allocation."""

from src.models.lightgbm.weekly_total.model import WeeklyTotalConfig
from src.models.lightgbm.weekly_total.spec import SPEC

__all__ = ["SPEC", "WeeklyTotalConfig"]
