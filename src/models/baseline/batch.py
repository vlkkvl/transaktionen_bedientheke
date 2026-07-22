"""Compiled batch evaluation of scalar baseline levels over ragged histories."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit

from src.models.baseline.croston import DEFAULT_CROSTON_ALPHA
from src.models.baseline.simple_exponential_smoothing import DEFAULT_SES_ALPHA
from src.models.baseline.tsb import (
    DEFAULT_TSB_DEMAND_ALPHA,
    DEFAULT_TSB_PROBABILITY_ALPHA,
)


SCALAR_MODEL_NAMES = (
    "simple_exponential_smoothing",
    "croston",
    "sba",
    "tsb",
)


@njit(cache=True)
def _forecast_levels(
    values: np.ndarray,
    offsets: np.ndarray,
    ses_alpha: float,
    croston_alpha: float,
    tsb_alpha_d: float,
    tsb_alpha_p: float,
) -> np.ndarray:
    result = np.zeros((len(offsets) - 1, 4), dtype=np.float64)
    for group in range(len(offsets) - 1):
        start = offsets[group]
        end = offsets[group + 1]
        if start == end:
            continue

        ses = values[start]
        probability = 1.0 if values[start] > 0 else 0.0
        croston_size = 0.0
        tsb_size = 0.0
        interval = 0.0
        previous_positive = -1
        has_positive = False

        for position in range(start, end):
            value = values[position]
            relative_position = position - start
            if position > start:
                ses = ses_alpha * value + (1.0 - ses_alpha) * ses
                occurrence = 1.0 if value > 0 else 0.0
                probability = (
                    tsb_alpha_p * occurrence + (1.0 - tsb_alpha_p) * probability
                )

            if value > 0:
                if not has_positive:
                    croston_size = value
                    tsb_size = value
                    interval = relative_position + 1.0
                    has_positive = True
                else:
                    croston_size = (
                        croston_alpha * value
                        + (1.0 - croston_alpha) * croston_size
                    )
                    tsb_size = (
                        tsb_alpha_d * value + (1.0 - tsb_alpha_d) * tsb_size
                    )
                    new_interval = relative_position - previous_positive
                    interval = (
                        croston_alpha * new_interval
                        + (1.0 - croston_alpha) * interval
                    )
                previous_positive = relative_position

        croston = (
            croston_size / interval if interval > 0 else croston_size
        )
        result[group, 0] = max(ses, 0.0)
        result[group, 1] = max(croston, 0.0)
        result[group, 2] = max((1.0 - croston_alpha / 2.0) * croston, 0.0)
        result[group, 3] = max(probability * tsb_size, 0.0)
    return result


def forecast_scalar_levels(
    histories: Sequence[Sequence[float] | np.ndarray],
    ses_alpha: float = DEFAULT_SES_ALPHA,
    croston_alpha: float = DEFAULT_CROSTON_ALPHA,
    tsb_alpha_d: float = DEFAULT_TSB_DEMAND_ALPHA,
    tsb_alpha_p: float = DEFAULT_TSB_PROBABILITY_ALPHA,
) -> np.ndarray:
    """Forecast four scalar levels for each supplied history."""
    arrays = [np.asarray(history, dtype=np.float64) for history in histories]
    lengths = np.fromiter((len(values) for values in arrays), dtype=np.int64)
    offsets = np.empty(len(arrays) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    flat = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)
    flat = np.maximum(
        np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    return _forecast_levels(
        flat,
        offsets,
        float(ses_alpha),
        float(croston_alpha),
        float(tsb_alpha_d),
        float(tsb_alpha_p),
    )
