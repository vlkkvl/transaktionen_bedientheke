"""Explicit registry of supported LightGBM architectures."""

from src.models.lightgbm.base import LightGBMVariantResult
from src.models.lightgbm.common import GlobalLightGBMResult
from src.models.lightgbm.features.builder import (
    GlobalLightGBMConfig,
    GlobalLightGBMFrames,
)
from src.models.lightgbm.l2.model import (
    MODEL_NAME,
    fit_global_lightgbm_frames,
)
from src.models.lightgbm.l2.spec import SPEC as L2
from src.models.lightgbm.tweedie.model import (
    TWEEDIE_MODEL_NAME,
    fit_tweedie_daily,
)
from src.models.lightgbm.tweedie.spec import SPEC as TWEEDIE
from src.models.lightgbm.two_stage.model import (
    TWO_STAGE_MODEL_NAME,
    fit_two_stage,
)
from src.models.lightgbm.two_stage.spec import SPEC as TWO_STAGE
from src.models.lightgbm.two_stage_quantile.model import (
    TWO_STAGE_QUANTILE_MODEL_NAME,
    fit_two_stage_quantile,
)
from src.models.lightgbm.two_stage_quantile.spec import SPEC as TWO_STAGE_QUANTILE
from src.models.lightgbm.weekly_total.model import (
    WEEKLY_MODEL_NAME,
    fit_weekly_total,
)
from src.models.lightgbm.weekly_total.spec import SPEC as WEEKLY_TOTAL

LIGHTGBM_MODELS = (L2, TWEEDIE, TWO_STAGE, TWO_STAGE_QUANTILE, WEEKLY_TOTAL)
LIGHTGBM_MODELS_BY_NAME = {spec.name: spec for spec in LIGHTGBM_MODELS}

LIGHTGBM_MODEL_LABELS = {
    MODEL_NAME: "Daily direct LightGBM (L2)",
    TWEEDIE_MODEL_NAME: "Daily direct LightGBM (Tweedie)",
    TWO_STAGE_MODEL_NAME: "Two-stage LightGBM (occurrence × quantity)",
    TWO_STAGE_QUANTILE_MODEL_NAME: (
        "Two-stage LightGBM (occurrence × quantile quantity P10/P50/P90)"
    ),
    WEEKLY_MODEL_NAME: "7-day total LightGBM × weekday allocation",
}


def fit_all_lightgbm_models(
    frames: GlobalLightGBMFrames,
    config: GlobalLightGBMConfig | None = None,
) -> dict[str, GlobalLightGBMResult | LightGBMVariantResult]:
    """Compatibility helper for fitting every model to prepared frames."""
    config = GlobalLightGBMConfig() if config is None else config
    fitters = (
        (MODEL_NAME, fit_global_lightgbm_frames),
        (TWEEDIE_MODEL_NAME, fit_tweedie_daily),
        (TWO_STAGE_MODEL_NAME, fit_two_stage),
        (TWO_STAGE_QUANTILE_MODEL_NAME, fit_two_stage_quantile),
        (WEEKLY_MODEL_NAME, fit_weekly_total),
    )
    results: dict[str, GlobalLightGBMResult | LightGBMVariantResult] = {}
    for position, (model_name, fitter) in enumerate(fitters, start=1):
        print(
            f"[model {position}/{len(fitters)}] Fitting "
            f"{LIGHTGBM_MODEL_LABELS[model_name]}...",
            flush=True,
        )
        result = fitter(frames, config)
        results[model_name] = result
        print(
            f"[model {position}/{len(fitters)}] Completed {model_name}: "
            f"{len(result.forecasts):,} forecasts",
            flush=True,
        )
    return results

__all__ = [
    "LIGHTGBM_MODELS",
    "LIGHTGBM_MODELS_BY_NAME",
    "LIGHTGBM_MODEL_LABELS",
    "fit_all_lightgbm_models",
]
