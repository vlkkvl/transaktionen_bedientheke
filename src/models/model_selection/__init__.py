"""Cluster-wise baseline model selection via rolling-origin evaluation."""
from src.models.model_selection.config import SelectionConfig

__all__ = [
    "BaselineSelectionResult",
    "SelectionConfig",
    "run_baseline_selection",
    "write_results",
]


def __getattr__(name: str):
    if name in {"BaselineSelectionResult", "run_baseline_selection", "write_results"}:
        from src.models.model_selection.best_baseline import (
            BaselineSelectionResult,
            run_baseline_selection,
            write_results,
        )

        exports = {
            "BaselineSelectionResult": BaselineSelectionResult,
            "run_baseline_selection": run_baseline_selection,
            "write_results": write_results,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
