"""Configuration domains shared by model families."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastWindow:
    """Expanding-origin backtest settings, independent of model parameters."""

    training_origins: int = 48
    validation_origins: int = 4
    test_origins: int = 20
    refit_interval_origins: int = 4
    origin_spacing_days: int = 7

    def __post_init__(self) -> None:
        invalid = [
            name
            for name, value in self.__dict__.items()
            if int(value) < 1
        ]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution controls which do not change the statistical model."""

    random_state: int = 42
    num_threads: int = 4

    def __post_init__(self) -> None:
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")


@dataclass(frozen=True)
class FitControl:
    """Training-loop controls shared in shape, but owned by each model config."""

    num_boost_round: int = 500
    early_stopping_rounds: int = 40

    def __post_init__(self) -> None:
        if self.num_boost_round < 1 or self.early_stopping_rounds < 1:
            raise ValueError("boosting and early-stopping rounds must be positive")
