"""Small model specification used by family and single-model runners."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


ConfigT = TypeVar("ConfigT")
FitCallable = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class ModelSpec(Generic[ConfigT]):
    """Everything orchestration needs to run one registered model."""

    name: str
    family: str
    config_factory: Callable[[], ConfigT]
    fit: FitCallable

    def make_config(self) -> ConfigT:
        return self.config_factory()
