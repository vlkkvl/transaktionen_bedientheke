"""Cluster-wise baseline model selection via rolling-origin evaluation."""
from src.models.model_selection.best_baseline import (
    BaselineSelectionResult,
    run_baseline_selection,
    write_results,
)
from src.models.model_selection.config import SelectionConfig

__all__ = [
    "BaselineSelectionResult",
    "SelectionConfig",
    "run_baseline_selection",
    "write_results",
]
