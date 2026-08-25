"""Feature definitions and engineering shared by all LightGBM models."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterable, Iterator
import unicodedata

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dateutil.easter import easter

from src.data.preparation.distribute_sales_over_active_days import (
    HOLIDAY_SUBDIVISIONS,
    create_germany_holidays,
    create_germany_ni_holidays,
    store_subdivisions,
)
from src.models.benchmark.config import ROOT, BenchmarkDesign
from src.models.benchmark.evaluation import (
    DEFAULT_DEMAND_COL,
    _create_assessed_origins,
    prepare_daily_rows,
)
from src.models.benchmark.models import create_history_features

TARGET_COLUMN = DEFAULT_DEMAND_COL
NORMALIZED_TARGET_COLUMN = "normalized_actual"
TARGET_SCALE_COLUMN = "target_mean"
DEFAULT_FEATURES_DIR = ROOT / "data" / "processed"
DEFAULT_FEATURES_PATH = DEFAULT_FEATURES_DIR / "lightgbm_features.parquet"
MIN_COMPLETED_GAPS_FOR_P90 = 10
FEATURE_ORIGIN_BATCH_SIZE = 4
EVENT_TRADING_DAY_RADIUS = 10
EVENT_BASELINE_ROWS = 24
EVENT_MIN_ARTICLE_BASELINE_OBSERVATIONS = 30
# Across the 72 production origins, own-event cells contain 2-14 distinct
# historical event dates (median 7). Two or three dates are treated as thin.
EVENT_MIN_POOLED_CELL_DATES = 4
# A position cell holds exactly one date per historical event occurrence, so
# two dates mean two different years and average out single-day shocks.
EVENT_MIN_POSITION_CELL_DATES = 2
# Position estimates are shrunk toward the window-level lift with this prior
# weight (in distinct-date units), so single-year cells stay usable instead of
# being hidden: value = (dates * position + weight * window) / (dates + weight).
EVENT_POSITION_SHRINKAGE_PRIOR_WEIGHT = 1
# Weather. `05_10` established that the usable weather signal is the day-level
# temperature *anomaly* interacted with the article: the seasonal path itself is
# already carried by month/iso_week/trailing means, and the network's within-day
# spatial temperature spread is negligible (median 0.53 C). The normal must be a
# same-calendar-day average over earlier years, never a trailing mean: a trailing
# baseline lags the spring warming and turns a whole spring evaluation window
# into a spurious warm anomaly.
ERA5_WEATHER_PATH = ROOT / "data" / "interim" / "weather" / "era5_grid_daily_weather.csv"
# Promotion context extracted from raw sale lines by
# src/data/preparation/extract_action_history.py: campaign numbers, planned
# validity windows, and unit prices, which the processed daily table drops.
ACTION_DAYS_PATH = ROOT / "data" / "interim" / "actions" / "article_store_day_actions.parquet"
REGULAR_PRICE_PATH = ROOT / "data" / "interim" / "actions" / "article_day_regular_price.parquet"
# Prior weight (in pre-origin action-day observations) shrinking the
# per-article action lift toward the sourcing-group pooled lift.
ARTICLE_ACTION_LIFT_PRIOR_OBS = 24
MARKET_COORDINATES_PATH = ROOT / "data" / "raw" / "maerkte" / "maerkte.csv"
ERA5_GRID_DEGREES = 0.25
WEATHER_SEASONAL_WINDOW_DAYS = 7
WEATHER_MIN_SEASONAL_OBSERVATIONS = 10
# `06_01` shipped the raw anomaly and recovered only ~1/6 of the signal `05_10`
# measured: the boosters must rediscover an article-specific slope by splitting
# jointly on ARTIKEL_ID and the anomaly, which they do only partially. The
# sensitivity below hands that interaction over directly. `06_01` also showed the
# response is warm-side only (origins >= +2 C gained 0.61 pp, origins <= -2 C
# gained nothing), so the regressor is the warm half of the anomaly.
WEATHER_MIN_SENSITIVITY_DAYS = 150
FIXED_EVENT_DATES = {
    "new_year": (1, 1, "Neujahr"),
    "labour_day": (5, 1, "Erster Mai"),
    "german_unity": (10, 3, "Tag der Deutschen Einheit"),
    "reformation_day": (10, 31, "Reformationstag"),
    "christmas_day_1": (12, 25, "Erster Weihnachtstag"),
    "christmas_day_2": (12, 26, "Zweiter Weihnachtstag"),
}
REMOVED_FEATURE_COLUMNS = frozenset(
    {
        "lag_364",
        "lag_371",
        # The 08_03 cross-article action batch, fully reverted. The store-wide
        # flight-intensity count was dropped first (weakest mechanism, and the
        # pooled-bias source on the superseded dataset: occurrence rank
        # 14.8/55, bias -0.97% -> -1.63%). The two class-competition features
        # were then measured null on transactions_fixed (row +0.17 pp, no
        # closure of the quiet-class bias, gain share <= 0.05%) and removed;
        # notebooks/08_calendar_features/08_03 is the executed record.
        "store_other_actions_on_forecast_day",
        "same_class_other_actions_on_forecast_day",
        "same_class_action_share_on_forecast_day",
        # The 08_05 decentral-markdown batch, fully reverted. Mechanisms were
        # real and pre-verified (markdown-day demand 1.61x at forecast/actual
        # 0.59, propensity 1.3%->37.7%, split-half lift corr 0.78), but the
        # event is a same-day store decision: propensity spreads the lift over
        # ~12x more quiet days than event days, so the boosters left the
        # features unused (best rank 17/64, gain share 0.53%) and row WAPE
        # moved +0.08 pp with the targeted rows unchanged (0.593->0.598).
        # This confirms the 09_01 oracle finding that average-lift corrections
        # cannot identify which rows spike; only a same-day data feed could.
        # notebooks/08_calendar_features/08_05 is the executed record.
        "decentral_markdowns_last_28d",
        "days_since_last_decentral_markdown",
        "series_markdown_rate",
        "article_markdown_lift",
        "markdown_expected_lift",
        "spoilage_days_last_28d",
    }
)


def _mothers_day(year: int) -> pd.Timestamp:
    """Return the second Sunday in May."""
    may_first = pd.Timestamp(year=year, month=5, day=1)
    days_to_sunday = (6 - may_first.dayofweek) % 7
    return may_first + pd.Timedelta(days=days_to_sunday + 7)


def _normalize_event_name(event_name: str) -> str:
    """Return a stable identifier for a calendar event name."""
    ascii_name = unicodedata.normalize("NFKD", event_name).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def _annual_events(years: Iterable[int]) -> list[dict[str, Any]]:
    """Return the non-Easter events that receive trading-day anchors."""
    rows: list[dict[str, Any]] = []
    for year in sorted(set(int(value) for value in years)):
        for event_key, (month, day, event_name) in FIXED_EVENT_DATES.items():
            rows.append(
                {
                    "event_key": event_key,
                    "event_name": event_name,
                    "event_date": pd.Timestamp(year=year, month=month, day=day),
                }
            )
        rows.append(
            {
                "event_key": "mothers_day",
                "event_name": "Muttertag",
                "event_date": _mothers_day(year),
            }
        )
    return rows


def _public_holiday_dates(years: Iterable[int]) -> set[pd.Timestamp]:
    holiday_map = create_germany_ni_holidays(range(min(years), max(years) + 1))
    return {pd.Timestamp(day) for day in holiday_map}


def _is_trading_day(day: pd.Timestamp, holidays: set[pd.Timestamp]) -> bool:
    normalized = day.normalize()
    return normalized.dayofweek != 6 and normalized not in holidays


def _trading_day_offset(
    target: pd.Timestamp,
    event_date: pd.Timestamp,
    holidays: set[pd.Timestamp],
) -> int:
    """Return the signed active-day position of target around an event."""
    target = target.normalize()
    event_date = event_date.normalize()
    if target == event_date:
        return 0
    if target < event_date:
        dates = pd.date_range(target, event_date - pd.Timedelta(days=1), freq="D")
        return -sum(_is_trading_day(day, holidays) for day in dates)
    dates = pd.date_range(event_date + pd.Timedelta(days=1), target, freq="D")
    return sum(_is_trading_day(day, holidays) for day in dates)


def _shift_trading_days(
    event_date: pd.Timestamp,
    offset: int,
    holidays: set[pd.Timestamp],
) -> pd.Timestamp:
    """Map a signed trading-day offset onto an event occurrence."""
    if offset == 0:
        return event_date.normalize()
    direction = 1 if offset > 0 else -1
    remaining = abs(int(offset))
    current = event_date.normalize()
    while remaining:
        current += pd.Timedelta(days=direction)
        if _is_trading_day(current, holidays):
            remaining -= 1
    return current


def _closure_block_length(
    event_date: pd.Timestamp,
    holidays: set[pd.Timestamp],
) -> int:
    """Count consecutive closed dates in the block containing an event."""
    event_date = event_date.normalize()

    def closed(day: pd.Timestamp) -> bool:
        return not _is_trading_day(day, holidays)

    if not closed(event_date):
        return 0
    start = event_date
    while closed(start - pd.Timedelta(days=1)):
        start -= pd.Timedelta(days=1)
    end = event_date
    while closed(end + pd.Timedelta(days=1)):
        end += pd.Timedelta(days=1)
    return int((end - start).days + 1)


def _nearest_non_easter_event(
    target: pd.Timestamp,
    events: list[dict[str, Any]],
    holidays: set[pd.Timestamp],
) -> tuple[dict[str, Any], int] | None:
    """Return rule-(a) context only when a non-Easter event is nearest overall."""
    non_easter_by_date = {
        event["event_date"].normalize(): event for event in events
    }
    all_event_dates = set(non_easter_by_date) | {
        day.normalize() for day in holidays
    }
    candidates = []
    for event_date in all_event_dates:
        calendar_distance = abs((event_date - target).days)
        if calendar_distance > 24:
            continue
        offset = _trading_day_offset(target, event_date, holidays)
        candidates.append(
            (
                abs(offset),
                calendar_distance,
                event_date,
                non_easter_by_date.get(event_date),
                offset,
            )
        )
    if not candidates:
        return None
    _, _, _, event, offset = min(candidates, key=lambda value: value[:3])
    if event is None or abs(offset) > EVENT_TRADING_DAY_RADIUS:
        return None
    return event, int(offset)


def _resolve_annual_anchor(
    target: pd.Timestamp,
    events: list[dict[str, Any]],
    holidays: set[pd.Timestamp],
) -> dict[str, Any]:
    """Resolve one anchor and its event context through the shared rule path."""
    target = target.normalize()
    event_match = _nearest_non_easter_event(target, events, holidays)
    if event_match is not None:
        event, trading_offset = event_match
        previous_event = next(
            candidate
            for candidate in events
            if candidate["event_key"] == event["event_key"]
            and candidate["event_date"].year == event["event_date"].year - 1
        )
        return {
            "anchor_period": _shift_trading_days(
                previous_event["event_date"], trading_offset, holidays
            ),
            "anchor_kind": "event",
            "anchor_event_key": event["event_key"],
            "anchor_event_name": event["event_name"],
            "event_trading_day_offset": trading_offset,
            "event_closure_block_length": _closure_block_length(
                event["event_date"], holidays
            ),
        }

    easter_current = pd.Timestamp(easter(target.year))
    easter_previous = pd.Timestamp(easter(target.year - 1))
    if (
        easter_current - pd.Timedelta(days=21)
        <= target
        <= easter_current + pd.Timedelta(days=64)
    ):
        anchor = target - (easter_current - easter_previous)
        anchor_kind = "easter"
    else:
        anchor = target - pd.Timedelta(days=364)
        anchor_kind = "regular"
    return {
        "anchor_period": anchor,
        "anchor_kind": anchor_kind,
        "anchor_event_key": None,
        "anchor_event_name": None,
        "event_trading_day_offset": None,
        "event_closure_block_length": None,
    }


def anchor_date(current_date: object) -> pd.Timestamp:
    """Resolve the prior-year annual anchor for a target date."""
    target = pd.Timestamp(current_date).normalize()
    years = range(target.year - 1, target.year + 2)
    context = _resolve_annual_anchor(
        target,
        _annual_events(years),
        _public_holiday_dates(years),
    )
    return context["anchor_period"]


def _normalize_feature_path(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_materialized_features(feature_path: Path | str = DEFAULT_FEATURES_PATH) -> pd.DataFrame:
    """Load previously materialized row-level LightGBM features."""
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(f"Materialized feature dataset not found: {path}")
    return pd.read_parquet(path)


def materialize_features_for_origins(
    con: duckdb.DuckDBPyConnection,
    *,
    origins: Iterable[object],
    design: BenchmarkDesign,
    feature_path: Path | str = DEFAULT_FEATURES_PATH,
    origin_batch_size: int = FEATURE_ORIGIN_BATCH_SIZE,
    return_frame: bool = True,
) -> pd.DataFrame | None:
    """Calculate and persist features without querying every origin at once."""
    normalized_origins = pd.DatetimeIndex(origins).normalize().unique().sort_values()
    if len(normalized_origins) == 0:
        raise ValueError("At least one origin is required")
    if origin_batch_size < 1:
        raise ValueError("origin_batch_size must be positive")

    path = _normalize_feature_path(feature_path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    writer: pq.ParquetWriter | None = None
    retained_frames: list[pd.DataFrame] = []
    try:
        for start in range(0, len(normalized_origins), origin_batch_size):
            batch_origins = normalized_origins[start : start + origin_batch_size]
            batch = make_feature_frame(con, batch_origins, design)
            batch["reason_closed"] = batch["reason_closed"].astype("string")
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            if return_frame:
                retained_frames.append(batch)
        if writer is None:
            raise RuntimeError("Feature materialization produced no origin batches")
        writer.close()
        writer = None
        temporary_path.replace(path)
    except Exception:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    if not return_frame:
        return None
    return pd.concat(retained_frames, ignore_index=True)


FEATURE_COLUMNS = (
    # IDs
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    # Calendar and forecast position
    "target_weekday",
    "iso_week",
    "month",
    "closed_days_next_1",
    "closed_days_next_2",
    "closed_days_next_3",
    "closed_days_prev_1",
    "closed_days_prev_2",
    "closed_days_prev_3",
    "days_to_nearest_event",
    "holiday_event_window",
    "event_name",
    # Known action schedule
    "action_on_forecast_day",
    "action_during_horizon",
    "action_depth_on_forecast_day",
    "action_flight_day",
    "same_class_promoted_depth",
    # Historical action behavior (strictly before the origin)
    "article_action_lift",
    "action_expected_lift",
    "article_typical_action_depth",
    "days_since_last_action",
    "actions_last_28d",
    "mean_action_lift_in_sourcing_group",
    # Maturity and demand gaps
    "active_days_before_origin",
    "demand_days_before_origin",
    "active_zero_demand_gap",
    "demand_rate_last_6",
    "demand_rate_last_12",
    "demand_rate_last_24",
    "historical_p90_gap",
    "current_gap_over_historical_p90_gap",
    # Local demand history
    "same_weekday_lag_7",
    "same_weekday_lag_14",
    "rolling_6_mean",
    "rolling_24_mean",
    "rolling_24_mean_non_event",
    "event_window_share_last_24",
    "rolling_28_demand_rate",
    "same_weekday_mean_4",
    "same_weekday_mean_8",
    # Annual demand history
    "annual_lookup_days_available",
    "same_weekday_last_year_mean",
    "same_week_last_year_mean",
    "product_cross_store_same_weekday_last_year_mean",
    "event_lift_series",
    "event_lift_pooled_occurrence",
    "event_lift_pooled_quantity",
    "event_lift_pooled_total",
    "event_position_lift_occurrence",
    "event_position_lift_quantity",
    "event_position_lift_total",
    # Demand regime
    "ADI",
    "CV2",
    # Cross-sectional context
    "product_cross_store_mean_28",
    "product_weekday_profile_value",
    "store_category_mean_28",
    # Weather
    "temperature_anomaly_c",
    "temperature_expected_lift",
)

DIRECT_FEATURE_COLUMNS = tuple(
    feature
    for feature in FEATURE_COLUMNS
    if feature
    not in {
        "event_lift_pooled_occurrence",
        "event_lift_pooled_quantity",
        "event_position_lift_occurrence",
        "event_position_lift_quantity",
    }
)

FEATURE_DESCRIPTIONS = {
    "ARTIKEL_ID": (
        "Article identifier of the forecast series; encoded as a categorical feature."
    ),
    "MARKT_ID": (
        "Store identifier of the forecast series; encoded as a categorical feature."
    ),
    "sourcing_group": (
        "Sourcing group assigned to the article-store series during daily-row "
        "preparation, such as FCM or Pseudo; encoded as a categorical feature."
    ),
    "category_id": (
        "Product-category identifier of the forecast series; encoded as a categorical "
        "feature."
    ),
    "target_weekday": (
        "ISO weekday of the forecast target date (Monday = 1, Sunday = 7); encoded "
        "as a categorical feature."
    ),
    "iso_week": (
        "ISO week number of the forecast target date; encoded as a categorical feature."
    ),
    "month": (
        "Calendar-month number of the forecast target date (January = 1); encoded as "
        "a categorical feature."
    ),
    "closed_days_next_1": (
        "Indicator that the store is closed on target + 1 calendar day. A store-date "
        "is closed when no article row at that store is active; a missing store-date "
        "is treated as not closed."
    ),
    "closed_days_next_2": (
        "Count of closed store-dates in the two calendar dates target + 1 through "
        "target + 2. A date is closed when no article row at the store is active; "
        "missing store-dates do not increase the count."
    ),
    "closed_days_next_3": (
        "Count of closed store-dates in the three calendar dates target + 1 through "
        "target + 3. A date is closed when no article row at the store is active; "
        "missing store-dates do not increase the count."
    ),
    "closed_days_prev_1": (
        "Indicator that the store is closed on target - 1 calendar day. A store-date "
        "is closed when no article row at that store is active; a missing store-date "
        "is treated as not closed."
    ),
    "closed_days_prev_2": (
        "Count of closed store-dates in the two calendar dates target - 2 through "
        "target - 1. A date is closed when no article row at the store is active; "
        "missing store-dates do not increase the count."
    ),
    "closed_days_prev_3": (
        "Count of closed store-dates in the three calendar dates target - 3 through "
        "target - 1. A date is closed when no article row at the store is active; "
        "missing store-dates do not increase the count."
    ),
    "days_to_nearest_event": (
        "Signed number of calendar days from the target date to the nearest "
        "Niedersachsen public holiday or Muttertag event: event date minus target "
        "date. Positive values indicate dates before the event and negative values "
        "dates after it; ties are resolved in favor of the earlier event."
    ),
    "holiday_event_window": (
        "Categorical position relative to the nearest Niedersachsen public holiday or "
        "Muttertag: event date, 1-3 calendar days before, 1-3 calendar days after, or "
        "none when the absolute offset exceeds three days."
    ),
    "event_name": (
        "Name of the nearest Niedersachsen public holiday or Muttertag when its "
        "absolute calendar-day offset is at most three; otherwise the category none."
    ),
    "action_on_forecast_day": (
        "Article-store promotion indicator recorded for the forecast target date; no "
        "filter on target-day activity is applied."
    ),
    "action_during_horizon": (
        "Maximum article-store promotion indicator across all observed target rows in "
        "the configured forecast horizon for the origin; inactive target rows remain "
        "in the maximum."
    ),
    "action_depth_on_forecast_day": (
        "One minus the article's promoted unit price on the forecast target date "
        "divided by its mean non-action unit price over the 28 calendar days "
        "strictly before the origin. The promoted price belongs to the same "
        "known-ahead promotion schedule as action_on_forecast_day; the regular "
        "reference uses only pre-origin days. Zero on days without an action; "
        "missing when the action's price or the regular reference is unknown."
    ),
    "action_flight_day": (
        "Position of the forecast target date inside its promotion's planned "
        "validity window: target date minus GUELTIG_VON plus one, capped at 28. "
        "The validity window belongs to the same known-ahead promotion schedule "
        "as action_on_forecast_day. Zero on days without an action; missing when "
        "an action has no recorded window."
    ),
    "same_class_promoted_depth": (
        "Deepest historically-typical discount among the articles of the same "
        "Warenklasse promoted at the store on the forecast target date, the "
        "article itself included. Promotion membership comes from the known-ahead "
        "schedule; each article's typical depth is measured only on its action "
        "days strictly before the origin. Zero when no promoted class article has "
        "a measurable depth history."
    ),
    "article_action_lift": (
        "Per-article promotion lift: the article's mean demand on active action "
        "rows divided by its mean demand on active non-action rows minus one, "
        "both pooled over stores and restricted to rows strictly before the "
        "origin, shrunk toward mean_action_lift_in_sourcing_group with a prior "
        "weight of 24 action-day observations. Missing only when neither the "
        "article nor its sourcing group has any usable history."
    ),
    "action_expected_lift": (
        "article_action_lift on rows whose forecast target date has a recorded "
        "promotion, and zero otherwise — the precomputed interaction of the "
        "known action schedule with the article's historical promotion response."
    ),
    "article_typical_action_depth": (
        "Mean historical discount depth of the article: one minus the promoted "
        "unit price divided by the article's mean non-action unit price of the "
        "28 days before each action day, averaged over all action days strictly "
        "before the origin. Missing when the article has no measurable action "
        "price history."
    ),
    "days_since_last_action": (
        "Calendar-day difference between the origin and the most recent observed row "
        "with action_flag = 1 strictly before the origin. Activity is not filtered; "
        "the value is missing when no earlier action exists."
    ),
    "actions_last_28d": (
        "Sum of action_flag over observed article-store rows in the half-open calendar "
        "interval [origin - 28 days, origin). Activity is not filtered, missing dates "
        "are not synthesized, and the result is zero when no rows exist."
    ),
    "mean_action_lift_in_sourcing_group": (
        "Across all article-store rows in the sourcing group strictly before the "
        "origin, the mean demand on active promotion observations divided by the mean "
        "demand on active non-promotion observations, minus one. Inactive observations "
        "are excluded from both denominators; active zero-demand observations are "
        "included. The value is missing if either group is absent or the regular-day "
        "mean is zero."
    ),
    "active_days_before_origin": (
        "Count of observed article-store rows with is_active = true strictly before "
        "the origin. Each daily row contributes at most one; inactive and missing "
        "calendar dates do not contribute."
    ),
    "demand_days_before_origin": (
        "Count of observed article-store rows that are active and have demand > 0 "
        "strictly before the origin; inactive, active zero-demand, and missing dates "
        "do not contribute."
    ),
    "active_zero_demand_gap": (
        "Number of active observed rows after the most recent active positive-demand "
        "row and strictly before the origin. Inactive and missing calendar dates do "
        "not increase the gap; before the first positive-demand row, all active "
        "observations are counted."
    ),
    "demand_rate_last_6": (
        "Share of positive-demand observations among the final six active article-store "
        "rows strictly before the origin. Inactive and missing calendar dates are "
        "excluded before the active-row window is formed."
    ),
    "demand_rate_last_12": (
        "Share of positive-demand observations among the final 12 active article-store "
        "rows strictly before the origin. Inactive and missing calendar dates are "
        "excluded before the active-row window is formed."
    ),
    "demand_rate_last_24": (
        "Share of positive-demand observations among the final 24 active article-store "
        "rows strictly before the origin. Inactive and missing calendar dates are "
        "excluded before the active-row window is formed."
    ),
    "historical_p90_gap": (
        "Reference length for completed gaps between consecutive active positive-demand "
        "observations, measured as the number of intervening active observations. With "
        "at least ten completed gaps strictly before the origin it is their continuous "
        "90th percentile; otherwise it is their maximum, or the current active gap when "
        "none is completed. Inactive and missing calendar dates do not lengthen gaps."
    ),
    "current_gap_over_historical_p90_gap": (
        "active_zero_demand_gap / historical_p90_gap. Both quantities are measured in "
        "active observations rather than calendar days. The result is zero when the "
        "current gap is zero and missing when a positive gap has no nonzero reference."
    ),
    "same_weekday_lag_7": (
        "Article-store demand on the exact calendar date target - 7 days. An active "
        "lookup date contributes its recorded demand; an inactive or missing lookup "
        "date produces a missing feature."
    ),
    "same_weekday_lag_14": (
        "Article-store demand on the exact calendar date target - 14 days. An active "
        "lookup date contributes its recorded demand; an inactive or missing lookup "
        "date produces a missing feature."
    ),
    "rolling_6_mean": (
        "Arithmetic mean of demand over the final six active article-store rows "
        "strictly before the origin. Inactive and missing dates are removed before "
        "forming the window, so the denominator is six when sufficient history exists."
    ),
    "rolling_24_mean": (
        "Arithmetic mean of demand over the final 24 active article-store rows "
        "strictly before the origin. Inactive and missing dates are removed before "
        "forming the window, so the denominator is 24 when sufficient history exists."
    ),
    "rolling_24_mean_non_event": (
        "Arithmetic mean of demand over the final 24 active article-store rows "
        "strictly before the origin whose holiday-event window is none. Event-window, "
        "inactive, and missing dates are removed before the window is formed, so "
        "holiday run-up and post-event echo days cannot inflate the level estimate."
    ),
    "event_window_share_last_24": (
        "Share of the final 24 active article-store rows strictly before the origin "
        "that fall inside a calendar holiday-event window. It signals how strongly "
        "run-up demand can contaminate the trailing windows that include event days."
    ),
    "rolling_28_demand_rate": (
        "Share of active observations with demand > 0 within the final 28 observed "
        "article-store rows strictly before the origin. The denominator includes only "
        "active rows, including active zero-demand rows; inactive rows and missing "
        "dates are excluded."
    ),
    "same_weekday_mean_4": (
        "Arithmetic mean over up to the final four active article-store observations "
        "strictly before the origin whose ISO weekday matches the target weekday and "
        "whose holiday-event window is none. Event-window, inactive, and missing dates "
        "are removed before the four-row window is formed."
    ),
    "same_weekday_mean_8": (
        "Arithmetic mean over up to the final eight active article-store observations "
        "strictly before the origin whose ISO weekday matches the target weekday and "
        "whose holiday-event window is none. Event-window, inactive, and missing dates "
        "are removed before the eight-row window is formed."
    ),
    "annual_lookup_days_available": (
        "Number of observed active article-store lookup dates used by the annual "
        "same-weekday mean. The usual five-date anchor-and-spoke lookup yields 0-5. "
        "For Easter-aligned and regular anchors, a date is usable only when its "
        "holiday-event window and event name match the target's context. "
        "When the nearest event overall is a configured non-Easter event within ten "
        "trading days, only its mapped anchor is eligible, so the value is 0 or 1 on "
        "the same 0-5 scale."
    ),
    "same_weekday_last_year_mean": (
        "Mean article-store demand over active observed dates at the resolved annual "
        "anchor and its +/-7 and +/-14-day spokes. For Easter-aligned and regular "
        "anchors, candidates must match the target's holiday-event window and event "
        "name. When the nearest event overall is a configured non-Easter event within "
        "ten trading days, the feature is its mapped anchor date alone, with no "
        "spokes or weekday snapping."
    ),
    "same_week_last_year_mean": (
        "Arithmetic mean of article-store demand over active observed rows in the "
        "Monday-Sunday week resolved from the shared annual anchor. Inactive and "
        "missing dates are excluded. The feature is missing for targets using a "
        "non-Easter event anchor."
    ),
    "product_cross_store_same_weekday_last_year_mean": (
        "Article demand averaged across active stores on the shared annual anchor and "
        "eligible +/-7 and +/-14-day spokes, then averaged across dates. For regular "
        "and Easter-aligned anchors, candidates must match the target's holiday-event "
        "window and event name. When the nearest event overall is a configured "
        "non-Easter event within ten trading days, only its mapped anchor date is "
        "used."
    ),
    "event_lift_series": (
        "Demand on the mapped prior-year event anchor divided by the article-store "
        "mean over the 24 active rows ending immediately before that anchor. It is "
        "available only for an active mapped anchor when the target is inside a "
        "calendar holiday-event window."
    ),
    "event_lift_pooled_occurrence": (
        "Ratio of summed positive-demand indicators to summed non-event occurrence "
        "baselines for the same event and closure-block length. Baselines are keyed "
        "by each historical observation's weekday and article, with a weekday, "
        "sourcing-group, and category fallback for article cells below 30 "
        "observations. Weekday is not part of the pooled event key. Event cells below "
        "four distinct historical event dates yield missing values; estimates never "
        "borrow observations from another event. Only observations strictly before "
        "the forecast origin enter the estimate."
    ),
    "event_lift_pooled_quantity": (
        "Ratio of summed positive event demand to summed non-event positive-demand "
        "baselines for the same event and closure-block length. Baselines are keyed "
        "by each historical observation's weekday and article, with a weekday, "
        "sourcing-group, and category fallback for article cells below 30 "
        "observations. Zero-demand observations are excluded from both sums. Event "
        "cells below four distinct historical event dates yield missing values; "
        "estimates never borrow observations from another event. Only observations "
        "strictly before the forecast origin enter the estimate."
    ),
    "event_lift_pooled_total": (
        "Total pooled demand lift for direct models, defined exactly as "
        "event_lift_pooled_occurrence multiplied by event_lift_pooled_quantity. Only "
        "observations strictly before the forecast origin enter either component."
    ),
    "event_position_lift_occurrence": (
        "Pooled occurrence lift restricted to historical observations sharing the "
        "target's exact signed calendar-day offset to the event, in addition to the "
        "event and closure-block-length keys of event_lift_pooled_occurrence. "
        "Baselines are identical to the window-level feature. The position estimate "
        "is shrunk toward the gated window-level lift with a one-date prior weight; "
        "without a window-level value, cells with fewer than two distinct historical "
        "event dates yield missing values. Only observations strictly before the "
        "forecast origin enter the estimate."
    ),
    "event_position_lift_quantity": (
        "Pooled positive-quantity lift restricted to historical observations sharing "
        "the target's exact signed calendar-day offset to the event, in addition to "
        "the event and closure-block-length keys of event_lift_pooled_quantity. "
        "Baselines are identical to the window-level feature. The position estimate "
        "is shrunk toward the gated window-level lift with a one-date prior weight; "
        "without a window-level value, cells with fewer than two distinct historical "
        "event dates yield missing values. Only observations strictly before the "
        "forecast origin enter the estimate."
    ),
    "event_position_lift_total": (
        "Total position-keyed demand lift for direct models, defined exactly as "
        "event_position_lift_occurrence multiplied by event_position_lift_quantity. "
        "Only observations strictly before the forecast origin enter either "
        "component."
    ),
    "ADI": (
        "Count of active article-store observations divided by the count of active "
        "positive-demand observations, using all rows strictly before the origin. "
        "Active zero-demand rows contribute only to the numerator; inactive and missing "
        "dates contribute to neither. The value is missing when no positive-demand "
        "observation exists."
    ),
    "CV2": (
        "Population variance of demand divided by squared mean demand, calculated over "
        "all active positive-demand article-store observations strictly before the "
        "origin. Only those positive observations enter the moment denominators; "
        "inactive, zero-demand, and missing dates are excluded."
    ),
    "product_cross_store_mean_28": (
        "For each article-date, demand is first averaged across active stores, with "
        "active zero-demand stores included and inactive stores excluded. The feature "
        "is the arithmetic mean of the available daily means in the final 28 observed "
        "article-date rows strictly before the origin. Days with no active store yield "
        "null and are excluded from the outer denominator; missing article-dates are "
        "absent and can make the row window span more than 28 calendar days."
    ),
    "product_weekday_profile_value": (
        "Weekday demand multiplier centred on 1.0. For each eligible article-date, "
        "demand is first averaged across active stores. The numerator is the mean over "
        "the final eight eligible open dates matching the target weekday; the "
        "denominator is the mean over the final 48 eligible open dates. Dates in a "
        "holiday-event window, fully closed dates, inactive stores, and missing dates "
        "are excluded. The value is missing when either mean is unavailable or the "
        "denominator is zero."
    ),
    "store_category_mean_28": (
        "For each store-category-date, demand is first averaged across active article "
        "rows, including active zeros and excluding inactive articles. The feature is "
        "the arithmetic mean of the available daily category means in the final 28 "
        "observed store-category-date rows strictly before the origin. Dates with no "
        "active article yield null and are excluded from the outer denominator; missing "
        "dates are absent and can make the row window span more than 28 calendar days."
    ),
    "temperature_anomaly_c": (
        "Daily mean 2 m temperature at the store on the forecast day, minus a seasonal "
        "normal for that store and calendar day, in degrees Celsius. The store is "
        "assigned the nearest 0.25 degree ERA5 grid cell. The normal is the mean "
        "temperature over dates within seven calendar days of the target day-of-year, "
        "taken strictly before the origin, and is null unless at least ten such dates "
        "exist; the feature is then null as well. Positive values mean the day is "
        "warmer than normal for its time of year. The seasonal path itself is already "
        "carried by month, ISO week, and the trailing means, so only the deviation is "
        "exposed here. A trailing-window baseline is deliberately not used: it lags the "
        "seasonal cycle and would report a systematic warm anomaly through spring."
    ),
    "temperature_expected_lift": (
        "The article's own temperature sensitivity multiplied by the warm part of the "
        "store's temperature anomaly on the forecast day, so the value is the expected "
        "relative demand change and is zero on days at or below the seasonal normal. "
        "The sensitivity is refitted at every origin from article-date rows strictly "
        "before it: demand is averaged across active stores, divided by that article's "
        "own weekday mean inside the same window so weekday structure cannot appear as a "
        "temperature response, and regressed on the warm part of the network temperature "
        "anomaly. At least 150 article-dates are required. Each slope is then shrunk "
        "toward zero by empirical Bayes, keeping the share of its variance that is "
        "signal, with the between-article variance estimated separately at every origin "
        "as the spread of the fitted slopes minus their mean sampling variance. Articles "
        "without a slope contribute zero. Only the warm half is used because the batch "
        "in 06_01 found gains on origins warmer than normal and none on colder ones."
    ),
}

if set(FEATURE_DESCRIPTIONS) != set(FEATURE_COLUMNS):
    missing = sorted(set(FEATURE_COLUMNS) - set(FEATURE_DESCRIPTIONS))
    unexpected = sorted(set(FEATURE_DESCRIPTIONS) - set(FEATURE_COLUMNS))
    raise RuntimeError(
        "LightGBM feature descriptions do not match FEATURE_COLUMNS: "
        f"missing={missing}, unexpected={unexpected}"
    )

CATEGORICAL_FEATURES = (
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    "target_weekday",
    "iso_week",
    "month",
    "holiday_event_window",
    "event_name",
)

FORECAST_ID_COLUMNS = (
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    "origin",
    "period",
    "active_days_before_origin",
    "demand_days_before_origin",
    "recent_occurrence_rate",
    "calendar_days_since_last_demand",
    "seasonal_mase_scale",
    "actual",
    "is_active",
    "reason_closed",
)

DIAGNOSTIC_COLUMNS = (
    "is_public_holiday",
    "action_on_forecast_day",
    "action_during_horizon",
    "days_since_last_action",
    "actions_last_28d",
    "mean_action_lift_in_sourcing_group",
    "rolling_28_demand_rate",
    "active_zero_demand_gap",
    "historical_p90_gap",
    "current_gap_over_historical_p90_gap",
)


def _required_materialized_columns() -> set[str]:
    return {
        *FEATURE_COLUMNS,
        *FORECAST_ID_COLUMNS,
        *DIAGNOSTIC_COLUMNS,
        NORMALIZED_TARGET_COLUMN,
        TARGET_SCALE_COLUMN,
    }


def _feature_cache_covers(
    con: duckdb.DuckDBPyConnection,
    feature_path: Path | str,
    origins: pd.DatetimeIndex,
) -> bool:
    """Check cached columns and origins without loading the full parquet."""
    path = Path(feature_path)
    if not path.exists():
        return False
    try:
        columns = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
        }
        if (
            not _required_materialized_columns().issubset(columns)
            or not REMOVED_FEATURE_COLUMNS.isdisjoint(columns)
        ):
            return False
        cached_origins = pd.DatetimeIndex(
            pd.to_datetime(
                [
                    row[0]
                    for row in con.execute(
                        "SELECT DISTINCT origin FROM read_parquet(?)", [str(path)]
                    ).fetchall()
                ]
            )
        ).normalize()
    except Exception:
        return False
    return pd.Index(origins).isin(cached_origins).all()


@dataclass(frozen=True)
class GlobalLightGBMConfig:
    """Expanding weekly design shared by the global LightGBM architectures."""

    training_origins: int = 48
    validation_origins: int = 4
    test_origins: int = 20
    refit_interval_origins: int = 4
    origin_spacing_days: int = 7
    num_boost_round: int = 500
    early_stopping_rounds: int = 40
    random_state: int = 42
    num_threads: int = 4

    def __post_init__(self) -> None:
        positive = {
            "training_origins": self.training_origins,
            "validation_origins": self.validation_origins,
            "test_origins": self.test_origins,
            "refit_interval_origins": self.refit_interval_origins,
            "origin_spacing_days": self.origin_spacing_days,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_threads": self.num_threads,
        }
        invalid = [name for name, value in positive.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")

    @property
    def history_origins(self) -> int:
        """Number of completed weekly origins required for the first refit."""
        return self.training_origins + self.validation_origins

    @property
    def refit_interval_days(self) -> int:
        """Calendar days between model refits."""
        return self.refit_interval_origins * self.origin_spacing_days


@dataclass(frozen=True)
class LightGBMOriginWindow:
    """Origin dates assigned to one expanding-window model refit."""

    training: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    evaluation: pd.DatetimeIndex


def iter_lightgbm_origin_windows(
    initial_history_origins: Iterable[object],
    evaluation_origins: Iterable[object],
    config: GlobalLightGBMConfig,
) -> Iterator[LightGBMOriginWindow]:
    """Yield expanding training splits with four-origin validation blocks."""
    initial = (
        pd.DatetimeIndex(initial_history_origins)
        .normalize()
        .unique()
        .sort_values()
    )
    evaluation = (
        pd.DatetimeIndex(evaluation_origins)
        .normalize()
        .unique()
        .sort_values()
    )
    if len(initial) < config.history_origins:
        raise ValueError(
            "Initial history has fewer origins than the configured training and "
            "validation windows"
        )

    for position in range(0, len(evaluation), config.refit_interval_origins):
        available = initial.append(evaluation[:position]).unique().sort_values()
        validation_start = len(available) - config.validation_origins
        yield LightGBMOriginWindow(
            training=available[:validation_start],
            validation=available[validation_start:],
            evaluation=evaluation[
                position : position + config.refit_interval_origins
            ],
        )


@dataclass
class GlobalLightGBMFrames:
    """Origin-aligned frames used by every global LightGBM variant.

    ``origin_frames`` is populated on the top-level backtest container. Each
    child contains one four-week test block, all training origins available at
    that refit, and the latest four completed validation origins.
    """

    training: pd.DataFrame
    validation: pd.DataFrame
    evaluation: pd.DataFrame
    training_origins: pd.DatetimeIndex
    evaluation_origin: pd.Timestamp | None = None
    origin_frames: tuple["GlobalLightGBMFrames", ...] = ()


def _store_daily_temperature() -> pd.DataFrame:
    """Return daily mean temperature per store, or an empty frame when unavailable.

    Stores are assigned to the nearest 0.25 degree ERA5 grid cell, the same
    mapping used in ``notebooks/01_data_understanding/01_05_weather.ipynb``.
    Missing inputs yield an empty frame so the feature degrades to NULL rather
    than breaking the build on installations without the weather cache.
    """
    if not ERA5_WEATHER_PATH.exists() or not MARKET_COORDINATES_PATH.exists():
        return pd.DataFrame(
            columns=["MARKT_ID", "period", "temperature_c"]
        ).astype({"MARKT_ID": "int64", "temperature_c": "float64"})
    stores = pd.read_csv(
        MARKET_COORDINATES_PATH, usecols=["MARKT_ID", "LONGITUDE", "LATITUDE"]
    ).dropna(subset=["LONGITUDE", "LATITUDE"])
    weather = pd.read_csv(
        ERA5_WEATHER_PATH,
        usecols=["date", "era5_latitude", "era5_longitude", "temperature_mean_c"],
        parse_dates=["date"],
    ).dropna(subset=["temperature_mean_c"])
    if stores.empty or weather.empty:
        return pd.DataFrame(
            columns=["MARKT_ID", "period", "temperature_c"]
        ).astype({"MARKT_ID": "int64", "temperature_c": "float64"})

    def nearest_cell(values: pd.Series) -> pd.Series:
        return (
            np.floor(values.astype(float) / ERA5_GRID_DEGREES + 0.5)
            * ERA5_GRID_DEGREES
        ).round(2)

    stores["era5_latitude"] = nearest_cell(stores["LATITUDE"])
    stores["era5_longitude"] = nearest_cell(stores["LONGITUDE"])
    joined = stores.merge(
        weather, on=["era5_latitude", "era5_longitude"], how="inner"
    )
    return pd.DataFrame(
        {
            "MARKT_ID": joined["MARKT_ID"].astype("int64"),
            "period": joined["date"].dt.date,
            "temperature_c": joined["temperature_mean_c"].astype("float64"),
        }
    ).drop_duplicates(["MARKT_ID", "period"])


@lru_cache(maxsize=4)
def _article_product_classes_cached(data_dir: str) -> pd.DataFrame:
    directory = Path(data_dir)
    empty = pd.DataFrame(
        columns=["ARTIKEL_ID", "product_class"]
    ).astype({"ARTIKEL_ID": "int64", "product_class": "object"})
    if not directory.exists() or not any(directory.glob("*.parquet")):
        return empty
    scratch = duckdb.connect()
    try:
        frame = scratch.execute(
            """
            SELECT
                ARTIKEL_ID::BIGINT AS ARTIKEL_ID,
                ANY_VALUE(N_WARENKLASSE_KBEZ) AS product_class
            FROM read_parquet(?)
            WHERE (is_fcm OR is_pseudo)
              AND WGR_ID IN (890, 900)
              AND N_WARENKLASSE_KBEZ IS NOT NULL
            GROUP BY ARTIKEL_ID
            """,
            [str(directory / "*.parquet")],
        ).fetchdf()
    finally:
        scratch.close()
    return frame if not frame.empty else empty


def _article_product_classes(design: BenchmarkDesign) -> pd.DataFrame:
    """Map every article to its Warenklasse from the processed transactions.

    The Warenklasse (``N_WARENKLASSE_KBEZ``, e.g. Schweinefleisch or Bratwurst)
    is a static article attribute that ``benchmark_daily_rows`` does not carry,
    so the mapping is read from the processed transaction parquet directly.
    Currently unused by the feature build — the 08_03 class-competition
    features were reverted as null — but retained for class-keyed follow-ups
    (e.g. depth-weighted competition). Missing inputs yield an empty frame so
    consumers degrade to zero/NULL rather than breaking the build, mirroring
    the weather cache behaviour.
    """
    return _article_product_classes_cached(str(design.data_dir)).copy()


def _article_store_action_days() -> pd.DataFrame:
    """Per article-store-day promotion record from the raw sale lines.

    Carries the campaign number, the planned validity window
    (``GUELTIG_VON``/``GUELTIG_BIS``) and the mean promoted unit price of the
    day. The planned window and price belong to the same known-ahead central
    promotion schedule as ``action_on_forecast_day``. A missing extract yields
    an empty frame so the features degrade to zero/NULL, mirroring the weather
    cache behaviour.
    """
    if not ACTION_DAYS_PATH.exists():
        return pd.DataFrame(
            columns=[
                "ARTIKEL_ID", "MARKT_ID", "period",
                "aktionsnummer", "gueltig_von", "gueltig_bis",
                "action_unit_price",
            ]
        ).astype({"ARTIKEL_ID": "int64", "MARKT_ID": "int64",
                  "action_unit_price": "float64"})
    return pd.read_parquet(
        ACTION_DAYS_PATH,
        columns=["ARTIKEL_ID", "MARKT_ID", "period", "aktionsnummer",
                 "gueltig_von", "gueltig_bis", "action_unit_price"],
    )


def _article_day_regular_prices() -> pd.DataFrame:
    """Per article-day mean non-action unit price across stores."""
    if not REGULAR_PRICE_PATH.exists():
        return pd.DataFrame(
            columns=["ARTIKEL_ID", "period", "regular_unit_price"]
        ).astype({"ARTIKEL_ID": "int64", "regular_unit_price": "float64"})
    return pd.read_parquet(
        REGULAR_PRICE_PATH,
        columns=["ARTIKEL_ID", "period", "regular_unit_price"],
    )


def _holiday_calendar(
    start: object, end: object, subdivision: str | None = None
) -> pd.DataFrame:
    """Build holiday and retail-event calendar features for one Bundesland.

    ``subdivision`` defaults to Niedersachsen. When it is supplied the returned
    frame carries a ``subdivision`` column so per-store calendars can be stacked.
    """
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    dates = pd.date_range(start_date, end_date, freq="D")
    years = range(start_date.year - 1, end_date.year + 2)
    holiday_map = (
        create_germany_ni_holidays(years)
        if subdivision is None
        else create_germany_holidays(subdivision, years)
    )
    public_holiday_dates = {pd.Timestamp(day) for day in holiday_map}
    events = [(pd.Timestamp(day), str(name)) for day, name in holiday_map.items()]
    events.extend(
        (_mothers_day(year), "Muttertag")
        for year in range(start_date.year - 1, end_date.year + 2)
    )
    rows: list[dict[str, Any]] = []
    for target in dates:
        nearest_date, nearest_name = min(
            events, key=lambda event: (abs((event[0] - target).days), event[0])
        )
        delta = int((nearest_date - target).days)
        if delta == 0:
            window = "holiday"
        elif 0 < delta <= 3:
            window = "before_holiday_1_3d"
        elif -3 <= delta < 0:
            window = "after_holiday_1_3d"
        else:
            window = "none"
        rows.append(
            {
                "period": target.date(),
                "is_public_holiday": target in public_holiday_dates,
                "days_to_nearest_event": delta,
                "holiday_event_window": window,
                "event_name": nearest_name if abs(delta) <= 3 else "none",
                "calendar_event_key": (
                    _normalize_event_name(nearest_name) if window != "none" else None
                ),
                "calendar_event_date": (
                    nearest_date.date() if window != "none" else None
                ),
                "calendar_event_closure_block_length": (
                    _closure_block_length(nearest_date, public_holiday_dates)
                    if window != "none"
                    else None
                ),
            }
        )
    calendar = pd.DataFrame(rows)
    if subdivision is not None:
        calendar.insert(0, "subdivision", subdivision)
    return calendar


def _annual_anchor_calendar(start: object, end: object) -> pd.DataFrame:
    """Build the shared annual anchor and event context for every date."""
    dates = pd.date_range(
        pd.Timestamp(start).normalize(),
        pd.Timestamp(end).normalize(),
        freq="D",
    )
    years = range(dates.min().year - 1, dates.max().year + 2)
    holidays = _public_holiday_dates(years)
    events = _annual_events(years)
    rows: list[dict[str, Any]] = []
    for target in dates:
        context = _resolve_annual_anchor(target, events, holidays)
        rows.append(
            {
                "target_period": target.date(),
                **{
                    key: value.date() if key == "anchor_period" else value
                    for key, value in context.items()
                },
            }
        )
    return pd.DataFrame(rows)


def create_feature_tables(
    con: duckdb.DuckDBPyConnection,
    *,
    origins: Iterable[object],
    design: BenchmarkDesign,
) -> None:
    """Create per-origin feature snapshots from ``benchmark_daily_rows``.

    The expensive history resolution (ASOF joins over the full daily table)
    runs here exactly once for every supplied origin; the per-origin feature
    query afterwards only performs cheap equality joins against the resulting
    snapshot tables. Feature queries may therefore only request origins that
    were supplied to this function.
    """
    bounds = con.execute(
        "SELECT MIN(period), MAX(period) FROM benchmark_daily_rows"
    ).fetchone()
    if bounds is None or bounds[0] is None:
        raise RuntimeError("benchmark_daily_rows is empty")
    con.register("ml_snapshot_origin_frame", _normalized_origins(origins))
    con.execute(
        "CREATE OR REPLACE TABLE ml_snapshot_origins AS "
        "SELECT * FROM ml_snapshot_origin_frame"
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ml_target_dates AS
        SELECT DISTINCT
            (o.origin + target_offset.day_offset * INTERVAL 1 DAY)::DATE
                AS target_period
        FROM ml_snapshot_origins AS o
        CROSS JOIN range({int(design.forecast_horizon_days)})
            AS target_offset(day_offset)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_series AS
        SELECT DISTINCT
            ARTIKEL_ID, MARKT_ID, sourcing_group, category_id
        FROM benchmark_daily_rows
        """
    )
    con.register("ml_calendar_frame", _holiday_calendar(bounds[0], bounds[1]))
    con.execute(
        "CREATE OR REPLACE TABLE ml_calendar AS SELECT * FROM ml_calendar_frame"
    )
    # The store network spans Niedersachsen and Nordrhein-Westfalen, whose public
    # holidays differ (Fronleichnam and Allerheiligen in NW only, Reformationstag
    # in NI only). `ml_calendar` above stays on the Niedersachsen calendar because
    # the history and cross-store aggregate tables it feeds are pooled over stores
    # and have no Bundesland to key on. The row-level event context below is
    # per-store, so a store is told about its own holidays — which is what drives
    # the pre-holiday stock-up features.
    con.register("ml_store_subdivision_frame", store_subdivisions())
    con.execute(
        "CREATE OR REPLACE TABLE ml_store_subdivision AS "
        "SELECT * FROM ml_store_subdivision_frame"
    )
    con.register(
        "ml_subdivision_calendar_frame",
        pd.concat(
            [
                _holiday_calendar(bounds[0], bounds[1], subdivision=subdivision)
                for subdivision in HOLIDAY_SUBDIVISIONS
            ],
            ignore_index=True,
        ),
    )
    con.execute(
        "CREATE OR REPLACE TABLE ml_subdivision_calendar AS "
        "SELECT * FROM ml_subdivision_calendar_frame"
    )
    # Store-day temperature anomaly relative to a same-calendar-day normal built
    # only from dates strictly before the origin, so the feature obeys the same
    # leakage rule as every historical feature.
    con.register("ml_store_temperature_frame", _store_daily_temperature())
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_store_temperature AS
        SELECT
            MARKT_ID,
            period::DATE AS period,
            temperature_c,
            EXTRACT(DOY FROM period::DATE)::INTEGER AS day_of_year
        FROM ml_store_temperature_frame
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ml_store_weather_anomalies AS
        WITH target_store_days AS (
            SELECT DISTINCT
                s.MARKT_ID,
                o.origin,
                t.target_period AS period,
                EXTRACT(DOY FROM t.target_period)::INTEGER AS day_of_year
            FROM (SELECT DISTINCT MARKT_ID FROM ml_series) AS s
            CROSS JOIN ml_snapshot_origins AS o
            CROSS JOIN range({int(design.forecast_horizon_days)})
                AS target_offset(day_offset)
            INNER JOIN ml_target_dates AS t
                ON t.target_period
                    = (o.origin + target_offset.day_offset * INTERVAL 1 DAY)::DATE
        ),
        seasonal_normals AS (
            SELECT
                d.MARKT_ID,
                d.origin,
                d.period,
                AVG(h.temperature_c) AS seasonal_normal_c,
                COUNT(*) AS normal_observations
            FROM target_store_days AS d
            INNER JOIN ml_store_temperature AS h
                ON h.MARKT_ID = d.MARKT_ID
                AND h.period < d.origin
                AND LEAST(
                        ABS(h.day_of_year - d.day_of_year),
                        365 - ABS(h.day_of_year - d.day_of_year)
                    ) <= {int(WEATHER_SEASONAL_WINDOW_DAYS)}
            GROUP BY d.MARKT_ID, d.origin, d.period
        )
        SELECT
            d.MARKT_ID,
            d.origin,
            d.period,
            CASE
                WHEN n.normal_observations
                        >= {int(WEATHER_MIN_SEASONAL_OBSERVATIONS)}
                    THEN current.temperature_c - n.seasonal_normal_c
            END AS temperature_anomaly_c
        FROM target_store_days AS d
        LEFT JOIN seasonal_normals AS n
            ON n.MARKT_ID = d.MARKT_ID
            AND n.origin = d.origin
            AND n.period = d.period
        LEFT JOIN ml_store_temperature AS current
            ON current.MARKT_ID = d.MARKT_ID
            AND current.period = d.period
        """
    )
    # Article-level warm-anomaly sensitivity. Fitting uses a network-mean anomaly
    # (the network's within-day spatial spread is ~0.5 C, so pooling stores costs
    # nothing and keeps this cheap), while the exposed feature multiplies the
    # article's slope by the row's own store-level warm anomaly.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_network_temperature AS
        SELECT
            period,
            AVG(temperature_c) AS temperature_c,
            EXTRACT(DOY FROM period)::INTEGER AS day_of_year
        FROM ml_store_temperature
        GROUP BY period
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ml_origin_network_anomaly AS
        WITH candidate_days AS (
            SELECT o.origin, t.period, t.temperature_c, t.day_of_year
            FROM ml_snapshot_origins AS o
            INNER JOIN ml_network_temperature AS t
                ON t.period < o.origin
        ),
        normals AS (
            SELECT
                d.origin,
                d.period,
                d.temperature_c,
                AVG(h.temperature_c) AS seasonal_normal_c,
                COUNT(*) AS normal_observations
            FROM candidate_days AS d
            INNER JOIN ml_network_temperature AS h
                ON h.period < d.origin
                AND LEAST(
                        ABS(h.day_of_year - d.day_of_year),
                        365 - ABS(h.day_of_year - d.day_of_year)
                    ) <= {int(WEATHER_SEASONAL_WINDOW_DAYS)}
            GROUP BY d.origin, d.period, d.temperature_c
        )
        SELECT
            origin,
            period,
            GREATEST(temperature_c - seasonal_normal_c, 0) AS warm_anomaly_c
        FROM normals
        WHERE normal_observations >= {int(WEATHER_MIN_SEASONAL_OBSERVATIONS)}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_article_daily_demand AS
        SELECT
            ARTIKEL_ID,
            period,
            EXTRACT(ISODOW FROM period)::INTEGER AS weekday,
            AVG(demand) AS kg
        FROM benchmark_daily_rows
        WHERE is_active
        GROUP BY ARTIKEL_ID, period
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ml_article_temperature_sensitivity AS
        WITH observations AS (
            SELECT
                a.origin,
                d.ARTIKEL_ID,
                d.weekday,
                d.kg,
                a.warm_anomaly_c AS x
            FROM ml_origin_network_anomaly AS a
            INNER JOIN ml_article_daily_demand AS d
                ON d.period = a.period
        ),
        -- Demand is expressed relative to the article's own weekday mean inside
        -- the same pre-origin window, so weekday structure cannot masquerade as
        -- a temperature response.
        weekday_means AS (
            SELECT origin, ARTIKEL_ID, weekday, AVG(kg) AS weekday_mean
            FROM observations
            GROUP BY origin, ARTIKEL_ID, weekday
        ),
        relative AS (
            SELECT
                o.origin,
                o.ARTIKEL_ID,
                o.x,
                o.kg / NULLIF(w.weekday_mean, 0) AS y
            FROM observations AS o
            INNER JOIN weekday_means AS w
                USING (origin, ARTIKEL_ID, weekday)
            WHERE w.weekday_mean > 0
        ),
        moments AS (
            SELECT
                origin,
                ARTIKEL_ID,
                COUNT(*) AS n,
                SUM(x) AS sum_x,
                SUM(y) AS sum_y,
                SUM(x * x) AS sum_xx,
                SUM(x * y) AS sum_xy,
                SUM(y * y) AS sum_yy
            FROM relative
            GROUP BY origin, ARTIKEL_ID
        ),
        slopes AS (
            SELECT
                origin,
                ARTIKEL_ID,
                n,
                sum_xx - sum_x * sum_x / n AS sxx,
                sum_yy - sum_y * sum_y / n AS syy,
                sum_xy - sum_x * sum_y / n AS sxy
            FROM moments
            WHERE n >= {int(WEATHER_MIN_SENSITIVITY_DAYS)}
        ),
        estimates AS (
            SELECT
                origin,
                ARTIKEL_ID,
                sxy / sxx AS beta,
                GREATEST(syy - (sxy * sxy) / sxx, 0) / ((n - 2) * sxx)
                    AS beta_variance
            FROM slopes
            WHERE sxx > 0 AND n > 2
        ),
        -- Empirical Bayes: the spread of the fitted slopes is the true spread
        -- plus sampling noise, so tau^2 = Var(beta) - mean(Var(beta_hat)) and
        -- each slope keeps the share of its variance that is signal.
        origin_priors AS (
            SELECT
                origin,
                GREATEST(VAR_SAMP(beta) - AVG(beta_variance), 0) AS tau_squared
            FROM estimates
            GROUP BY origin
        )
        SELECT
            e.origin,
            e.ARTIKEL_ID,
            e.beta AS article_temperature_slope,  -- kept for diagnostics
            COALESCE(
                e.beta * p.tau_squared
                    / NULLIF(p.tau_squared + e.beta_variance, 0),
                0
            ) AS article_temperature_sensitivity
        FROM estimates AS e
        INNER JOIN origin_priors AS p USING (origin)
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE ml_store_closure_features AS
        WITH store_calendar AS (
            SELECT
                MARKT_ID,
                period,
                NOT BOOL_OR(is_active) AS is_closed
            FROM benchmark_daily_rows
            GROUP BY MARKT_ID, period
        )
        -- Calendar-day RANGE frames skip missing store-dates, matching the
        -- COALESCE(FALSE) semantics of exact-offset self joins.
        SELECT
            MARKT_ID,
            period,
            COALESCE(COUNT_IF(is_closed) OVER next_1, 0)::INTEGER
                AS closed_days_next_1,
            COALESCE(COUNT_IF(is_closed) OVER next_2, 0)::INTEGER
                AS closed_days_next_2,
            COALESCE(COUNT_IF(is_closed) OVER next_3, 0)::INTEGER
                AS closed_days_next_3,
            COALESCE(COUNT_IF(is_closed) OVER previous_1, 0)::INTEGER
                AS closed_days_prev_1,
            COALESCE(COUNT_IF(is_closed) OVER previous_2, 0)::INTEGER
                AS closed_days_prev_2,
            COALESCE(COUNT_IF(is_closed) OVER previous_3, 0)::INTEGER
                AS closed_days_prev_3
        FROM store_calendar
        WINDOW
            next_1 AS (
                PARTITION BY MARKT_ID ORDER BY period
                RANGE BETWEEN INTERVAL 1 DAY FOLLOWING
                    AND INTERVAL 1 DAY FOLLOWING
            ),
            next_2 AS (
                PARTITION BY MARKT_ID ORDER BY period
                RANGE BETWEEN INTERVAL 1 DAY FOLLOWING
                    AND INTERVAL 2 DAY FOLLOWING
            ),
            next_3 AS (
                PARTITION BY MARKT_ID ORDER BY period
                RANGE BETWEEN INTERVAL 1 DAY FOLLOWING
                    AND INTERVAL 3 DAY FOLLOWING
            ),
            previous_1 AS (
                PARTITION BY MARKT_ID ORDER BY period
                RANGE BETWEEN INTERVAL 1 DAY PRECEDING
                    AND INTERVAL 1 DAY PRECEDING
            ),
            previous_2 AS (
                PARTITION BY MARKT_ID ORDER BY period
                RANGE BETWEEN INTERVAL 2 DAY PRECEDING
                    AND INTERVAL 1 DAY PRECEDING
            ),
            previous_3 AS (
                PARTITION BY MARKT_ID ORDER BY period
                RANGE BETWEEN INTERVAL 3 DAY PRECEDING
                    AND INTERVAL 1 DAY PRECEDING
            )
        """
    )
    con.register(
        "ml_annual_anchor_frame",
        _annual_anchor_calendar(bounds[0], bounds[1]),
    )
    con.execute(
        "CREATE OR REPLACE TABLE ml_annual_anchors AS "
        "SELECT * FROM ml_annual_anchor_frame"
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_annual_reference_dates AS
        SELECT
            anchors.target_period,
            anchors.anchor_kind,
            (anchors.anchor_period + spokes.day_offset * INTERVAL 1 DAY)::DATE
                AS reference_period
        FROM ml_annual_anchors AS anchors
        CROSS JOIN (VALUES (-14), (-7), (0), (7), (14)) AS spokes(day_offset)
        WHERE anchors.anchor_kind <> 'event' OR spokes.day_offset = 0
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_series_annual_features AS
        SELECT
            history.ARTIKEL_ID,
            history.MARKT_ID,
            candidate.target_period,
            COUNT(*) FILTER (
                WHERE history.is_active
                    AND (
                        candidate.anchor_kind = 'event'
                        OR (
                            reference_calendar.holiday_event_window
                                = target_calendar.holiday_event_window
                            AND reference_calendar.event_name
                                = target_calendar.event_name
                        )
                    )
            )::INTEGER AS annual_lookup_days_available,
            AVG(history.demand) FILTER (
                WHERE history.is_active
                    AND (
                        candidate.anchor_kind = 'event'
                        OR (
                            reference_calendar.holiday_event_window
                                = target_calendar.holiday_event_window
                            AND reference_calendar.event_name
                                = target_calendar.event_name
                        )
                    )
            ) AS same_weekday_last_year_mean
        FROM ml_annual_reference_dates AS candidate
        INNER JOIN ml_target_dates USING (target_period)
        INNER JOIN benchmark_daily_rows AS history
            ON history.period = candidate.reference_period
        INNER JOIN ml_calendar AS target_calendar
            ON candidate.target_period = target_calendar.period
        INNER JOIN ml_calendar AS reference_calendar
            ON candidate.reference_period = reference_calendar.period
        GROUP BY
            history.ARTIKEL_ID,
            history.MARKT_ID,
            candidate.target_period
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_series_weekly_annual_features AS
        WITH requested_reference_weeks AS (
            SELECT
                anchors.target_period,
                (
                    anchors.anchor_period
                    - DATE_DIFF(
                        'day',
                        DATE_TRUNC('week', anchors.target_period),
                        anchors.target_period
                    ) * INTERVAL 1 DAY
                )::DATE AS reference_week_start
            FROM ml_annual_anchors AS anchors
            INNER JOIN ml_target_dates USING (target_period)
            WHERE anchors.anchor_kind <> 'event'
        )
        SELECT
            history.ARTIKEL_ID,
            history.MARKT_ID,
            requested.target_period,
            AVG(history.demand) FILTER (WHERE history.is_active)
                AS same_week_last_year_mean
        FROM requested_reference_weeks AS requested
        INNER JOIN benchmark_daily_rows AS history
            ON history.period BETWEEN requested.reference_week_start
                AND requested.reference_week_start + INTERVAL 6 DAY
        GROUP BY history.ARTIKEL_ID, history.MARKT_ID, requested.target_period
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_series_lag_features AS
        -- Single-date RANGE frames return the exact period - 7/14 row when it
        -- exists and is active, and NULL for inactive or missing lookup dates.
        -- The window runs over the full history; only rows on requested
        -- target dates are stored because only those are ever joined.
        SELECT * FROM (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                period,
                MAX(demand) FILTER (WHERE is_active) OVER lag_7
                    AS same_weekday_lag_7,
                MAX(demand) FILTER (WHERE is_active) OVER lag_14
                    AS same_weekday_lag_14
            FROM benchmark_daily_rows
            WINDOW
                lag_7 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period
                    RANGE BETWEEN INTERVAL 7 DAY PRECEDING
                        AND INTERVAL 7 DAY PRECEDING
                ),
                lag_14 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period
                    RANGE BETWEEN INTERVAL 14 DAY PRECEDING
                        AND INTERVAL 14 DAY PRECEDING
                )
        )
        WHERE period IN (SELECT target_period FROM ml_target_dates)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ml_active_series_event_history AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            sourcing_group,
            category_id,
            period AS feature_date,
            demand,
            COUNT(*) OVER previous_24 AS previous_active_days,
            AVG(demand) OVER previous_24 AS previous_demand_mean
        FROM benchmark_daily_rows
        WHERE is_active
        WINDOW previous_24 AS (
            PARTITION BY ARTIKEL_ID, MARKT_ID
            ORDER BY period ROWS BETWEEN {EVENT_BASELINE_ROWS} PRECEDING
                AND 1 PRECEDING
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ml_series_event_lifts AS
        SELECT
            history.ARTIKEL_ID,
            history.MARKT_ID,
            anchors.target_period,
            history.demand / NULLIF(history.previous_demand_mean, 0)
                AS event_lift_series
        FROM ml_annual_anchors AS anchors
        INNER JOIN ml_target_dates USING (target_period)
        INNER JOIN ml_calendar AS calendar
            ON anchors.target_period = calendar.period
        INNER JOIN ml_active_series_event_history AS history
            ON anchors.anchor_period = history.feature_date
        WHERE calendar.holiday_event_window <> 'none'
            AND history.previous_active_days = {EVENT_BASELINE_ROWS}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ml_pooled_event_statistics AS
        WITH daily AS (
            SELECT
                history.feature_date,
                calendar.calendar_event_key,
                calendar.calendar_event_closure_block_length,
                calendar.days_to_nearest_event AS event_day_offset,
                EXTRACT(ISODOW FROM history.feature_date)::INTEGER AS event_weekday,
                history.ARTIKEL_ID,
                history.sourcing_group,
                history.category_id,
                COUNT(*) AS event_observation_count,
                COUNT_IF(history.demand > 0) AS event_positive_observation_count,
                SUM(history.demand) FILTER (WHERE history.demand > 0)
                    AS event_positive_demand_sum
            FROM ml_active_series_event_history AS history
            INNER JOIN ml_calendar AS calendar
                ON history.feature_date = calendar.period
            WHERE calendar.holiday_event_window <> 'none'
                AND history.previous_active_days = {EVENT_BASELINE_ROWS}
            GROUP BY
                feature_date,
                calendar_event_key,
                calendar_event_closure_block_length,
                event_day_offset,
                event_weekday,
                history.ARTIKEL_ID,
                history.sourcing_group,
                history.category_id
        )
        SELECT * FROM daily
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ml_pooled_article_baseline_statistics AS
        WITH daily AS (
            SELECT
                history.feature_date,
                EXTRACT(ISODOW FROM history.feature_date)::INTEGER
                    AS baseline_weekday,
                history.ARTIKEL_ID,
                COUNT(*) AS observation_count,
                COUNT_IF(history.demand > 0) AS positive_observation_count,
                SUM(history.demand) FILTER (WHERE history.demand > 0)
                    AS positive_demand_sum
            FROM ml_active_series_event_history AS history
            INNER JOIN ml_calendar AS calendar
                ON history.feature_date = calendar.period
            WHERE calendar.holiday_event_window = 'none'
                AND history.previous_active_days = {EVENT_BASELINE_ROWS}
            GROUP BY feature_date, baseline_weekday, history.ARTIKEL_ID
        )
        SELECT
            feature_date,
            baseline_weekday,
            ARTIKEL_ID,
            SUM(observation_count) OVER history AS baseline_observation_count,
            SUM(positive_observation_count) OVER history
                AS baseline_positive_observation_count,
            SUM(positive_demand_sum) OVER history AS baseline_positive_demand_sum
        FROM daily
        WINDOW history AS (
            PARTITION BY baseline_weekday, ARTIKEL_ID
            ORDER BY feature_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ml_pooled_coarse_baseline_statistics AS
        WITH daily AS (
            SELECT
                history.feature_date,
                EXTRACT(ISODOW FROM history.feature_date)::INTEGER
                    AS baseline_weekday,
                history.sourcing_group,
                history.category_id,
                COUNT(*) AS observation_count,
                COUNT_IF(history.demand > 0) AS positive_observation_count,
                SUM(history.demand) FILTER (WHERE history.demand > 0)
                    AS positive_demand_sum
            FROM ml_active_series_event_history AS history
            INNER JOIN ml_calendar AS calendar
                ON history.feature_date = calendar.period
            WHERE calendar.holiday_event_window = 'none'
                AND history.previous_active_days = {EVENT_BASELINE_ROWS}
            GROUP BY
                feature_date,
                baseline_weekday,
                history.sourcing_group,
                history.category_id
        )
        SELECT
            feature_date,
            baseline_weekday,
            sourcing_group,
            category_id,
            SUM(observation_count) OVER history AS baseline_observation_count,
            SUM(positive_observation_count) OVER history
                AS baseline_positive_observation_count,
            SUM(positive_demand_sum) OVER history AS baseline_positive_demand_sum
        FROM daily
        WINDOW history AS (
            PARTITION BY baseline_weekday, sourcing_group, category_id
            ORDER BY feature_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_active_series_recent_features AS
        SELECT
            history.ARTIKEL_ID,
            history.MARKT_ID,
            history.period AS feature_date,
            AVG(history.demand) OVER active_6 AS rolling_6_mean,
            AVG(history.demand) OVER active_24 AS rolling_24_mean,
            AVG((history.demand > 0)::INTEGER) OVER active_6 AS demand_rate_last_6,
            AVG((history.demand > 0)::INTEGER) OVER active_12
                AS demand_rate_last_12,
            AVG((history.demand > 0)::INTEGER) OVER active_24
                AS demand_rate_last_24,
            AVG((calendar.holiday_event_window <> 'none')::INTEGER) OVER active_24
                AS event_window_share_last_24
        FROM benchmark_daily_rows AS history
        INNER JOIN ml_calendar AS calendar ON history.period = calendar.period
        WHERE history.is_active
        WINDOW
            active_6 AS (
                PARTITION BY history.ARTIKEL_ID, history.MARKT_ID
                ORDER BY history.period ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
            ),
            active_12 AS (
                PARTITION BY history.ARTIKEL_ID, history.MARKT_ID
                ORDER BY history.period ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
            ),
            active_24 AS (
                PARTITION BY history.ARTIKEL_ID, history.MARKT_ID
                ORDER BY history.period ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_active_series_non_event_features AS
        -- The trailing window here counts only active rows outside calendar
        -- event windows, so run-up and post-event echo days cannot inflate it.
        SELECT
            history.ARTIKEL_ID,
            history.MARKT_ID,
            history.period AS feature_date,
            AVG(history.demand) OVER non_event_24 AS rolling_24_mean_non_event
        FROM benchmark_daily_rows AS history
        INNER JOIN ml_calendar AS calendar ON history.period = calendar.period
        WHERE history.is_active
            AND calendar.holiday_event_window = 'none'
        WINDOW non_event_24 AS (
            PARTITION BY history.ARTIKEL_ID, history.MARKT_ID
            ORDER BY history.period ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ml_series_features AS
        WITH row_counts AS (
            SELECT
                d.*,
                COUNT_IF(is_active) OVER lifetime AS active_days,
                COUNT_IF(is_active AND demand > 0) OVER lifetime AS demand_days,
                SUM(CASE WHEN is_active AND demand > 0 THEN demand ELSE 0 END)
                    OVER lifetime
                    AS positive_sum,
                SUM(
                    CASE WHEN is_active AND demand > 0 THEN demand * demand ELSE 0 END
                )
                    OVER lifetime AS positive_square_sum,
                AVG(demand) FILTER (WHERE is_active AND demand > 0) OVER lifetime
                    AS target_mean,
                MAX(CASE WHEN is_active AND demand > 0 THEN period END) OVER lifetime
                    AS last_positive_period,
                MAX(CASE WHEN action_flag = 1 THEN period END) OVER lifetime
                    AS last_action_period,
                AVG((demand > 0)::INTEGER) FILTER (WHERE is_active) OVER trailing_28
                    AS rolling_28_demand_rate,
                COALESCE(
                    SUM(action_flag) OVER trailing_calendar_28,
                    0
                )::INTEGER AS actions_last_28d
            FROM benchmark_daily_rows AS d
            WINDOW
                lifetime AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ),
                trailing_28 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
                ),
                trailing_calendar_28 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period
                    RANGE BETWEEN INTERVAL 27 DAY PRECEDING AND CURRENT ROW
                )
        ),
        windowed AS (
            SELECT
                *,
                MAX(
                    CASE WHEN is_active AND demand > 0 THEN active_days END
                ) OVER lifetime AS last_positive_active_day
            FROM row_counts
            WINDOW lifetime AS (
                PARTITION BY ARTIKEL_ID, MARKT_ID
                ORDER BY period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        ),
        regimes AS (
            SELECT
                *,
                active_days::DOUBLE / NULLIF(demand_days, 0) AS ADI,
                GREATEST(
                    positive_square_sum / NULLIF(demand_days, 0)
                        - POWER(positive_sum / NULLIF(demand_days, 0), 2),
                    0
                ) / NULLIF(POWER(positive_sum / NULLIF(demand_days, 0), 2), 0)
                    AS CV2
            FROM windowed
        )
        SELECT
            r.ARTIKEL_ID,
            r.MARKT_ID,
            r.period AS feature_date,
            r.active_days,
            r.demand_days,
            r.target_mean,
            r.last_positive_period,
            r.last_action_period,
            r.is_active AND r.demand > 0 AS is_positive_sale,
            r.active_days - COALESCE(r.last_positive_active_day, 0)
                AS active_zero_demand_gap,
            r.actions_last_28d,
            r.rolling_28_demand_rate,
            r.ADI,
            r.CV2
        FROM regimes AS r
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_gap_statistics AS
        WITH positive_gaps AS (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                feature_date AS gap_feature_date,
                active_days
                    - LAG(active_days) OVER (
                        PARTITION BY ARTIKEL_ID, MARKT_ID ORDER BY feature_date
                    )
                    - 1 AS completed_gap
            FROM ml_series_features
            WHERE is_positive_sale
        )
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            gap_feature_date,
            COUNT(completed_gap) OVER gap_history AS completed_gap_count,
            QUANTILE_CONT(completed_gap, 0.9) OVER gap_history
                AS historical_gap_p90,
            MAX(completed_gap) OVER gap_history AS historical_max_gap
        FROM positive_gaps
        WINDOW gap_history AS (
            PARTITION BY ARTIKEL_ID, MARKT_ID
            ORDER BY gap_feature_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_sourcing_group_action_features AS
        WITH daily AS (
            SELECT
                sourcing_group,
                period,
                SUM(demand) FILTER (WHERE is_active AND action_flag = 1)
                    AS action_demand,
                COUNT(*) FILTER (WHERE is_active AND action_flag = 1)
                    AS action_observations,
                SUM(demand) FILTER (WHERE is_active AND action_flag = 0)
                    AS regular_demand,
                COUNT(*) FILTER (WHERE is_active AND action_flag = 0)
                    AS regular_observations
            FROM benchmark_daily_rows
            GROUP BY sourcing_group, period
        ),
        cumulative AS (
            SELECT
                sourcing_group,
                period AS feature_date,
                SUM(action_demand) OVER history AS action_demand_sum,
                SUM(action_observations) OVER history AS action_observations,
                SUM(regular_demand) OVER history AS regular_demand_sum,
                SUM(regular_observations) OVER history AS regular_observations
            FROM daily
            WINDOW history AS (
                PARTITION BY sourcing_group
                ORDER BY period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            sourcing_group,
            feature_date,
            (
                action_demand_sum / NULLIF(action_observations, 0)
            ) / NULLIF(
                regular_demand_sum / NULLIF(regular_observations, 0),
                0
            ) - 1.0 AS mean_action_lift_in_sourcing_group
        FROM cumulative
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_series_weekday_features AS
        SELECT
            history.ARTIKEL_ID,
            history.MARKT_ID,
            history.period AS feature_date,
            EXTRACT(ISODOW FROM history.period)::INTEGER AS target_weekday,
            AVG(history.demand) OVER weekday_4 AS same_weekday_mean_4,
            AVG(history.demand) OVER weekday_8 AS same_weekday_mean_8
        FROM benchmark_daily_rows AS history
        INNER JOIN ml_calendar AS calendar ON history.period = calendar.period
        WHERE history.is_active
            AND calendar.holiday_event_window = 'none'
        WINDOW
            weekday_4 AS (
                PARTITION BY
                    history.ARTIKEL_ID,
                    history.MARKT_ID,
                    EXTRACT(ISODOW FROM history.period)
                ORDER BY history.period ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ),
            weekday_8 AS (
                PARTITION BY
                    history.ARTIKEL_ID,
                    history.MARKT_ID,
                    EXTRACT(ISODOW FROM history.period)
                ORDER BY history.period ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
            )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_product_daily AS
        SELECT
            ARTIKEL_ID,
            period,
            SUM(demand) AS product_demand,
            COUNT(*) AS observed_stores,
            COUNT_IF(is_active) AS active_stores,
            AVG(demand) FILTER (WHERE is_active) AS cross_store_mean
        FROM benchmark_daily_rows
        GROUP BY ARTIKEL_ID, period
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_product_features AS
        SELECT
            ARTIKEL_ID,
            period AS feature_date,
            AVG(cross_store_mean) OVER (
                PARTITION BY ARTIKEL_ID
                ORDER BY period ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
            ) AS product_cross_store_mean_28
        FROM ml_product_daily
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_product_open_features AS
        SELECT
            product.ARTIKEL_ID,
            product.period AS feature_date,
            AVG(product.cross_store_mean) OVER (
                PARTITION BY product.ARTIKEL_ID
                ORDER BY product.period ROWS BETWEEN 47 PRECEDING AND CURRENT ROW
            ) AS product_cross_store_open_mean_48
        FROM ml_product_daily AS product
        INNER JOIN ml_calendar AS calendar ON product.period = calendar.period
        WHERE product.active_stores > 0
            AND calendar.holiday_event_window = 'none'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_product_annual_features AS
        SELECT
            history.ARTIKEL_ID,
            candidate.target_period,
            AVG(history.cross_store_mean)
                FILTER (
                    WHERE candidate.anchor_kind = 'event'
                        OR (
                            reference_calendar.holiday_event_window
                                = target_calendar.holiday_event_window
                            AND reference_calendar.event_name
                                = target_calendar.event_name
                        )
                )
                AS product_cross_store_same_weekday_last_year_mean
        FROM ml_annual_reference_dates AS candidate
        INNER JOIN ml_target_dates USING (target_period)
        INNER JOIN ml_product_daily AS history
            ON history.period = candidate.reference_period
        INNER JOIN ml_calendar AS target_calendar
            ON candidate.target_period = target_calendar.period
        INNER JOIN ml_calendar AS reference_calendar
            ON candidate.reference_period = reference_calendar.period
        GROUP BY history.ARTIKEL_ID, candidate.target_period
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_product_weekday_features AS
        SELECT
            product.ARTIKEL_ID,
            product.period AS feature_date,
            EXTRACT(ISODOW FROM product.period)::INTEGER AS target_weekday,
            AVG(product.cross_store_mean) OVER (
                PARTITION BY
                    product.ARTIKEL_ID,
                    EXTRACT(ISODOW FROM product.period)
                ORDER BY product.period ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
            ) AS product_weekday_cross_store_mean_8
        FROM ml_product_daily AS product
        INNER JOIN ml_calendar AS calendar ON product.period = calendar.period
        WHERE product.active_stores > 0
            AND calendar.holiday_event_window = 'none'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_store_category_features AS
        WITH daily AS (
            SELECT
                MARKT_ID,
                category_id,
                period,
                AVG(demand) FILTER (WHERE is_active)
                    AS category_cross_product_mean
            FROM benchmark_daily_rows
            GROUP BY MARKT_ID, category_id, period
        )
        SELECT
            MARKT_ID,
            category_id,
            period AS feature_date,
            AVG(category_cross_product_mean) OVER (
                PARTITION BY MARKT_ID, category_id
                ORDER BY period ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
            ) AS store_category_mean_28
        FROM daily
        """
    )

    # Resolve every history ASOF join once for all snapshot origins. The
    # per-origin feature query then only equality-joins these snapshots, so
    # the full-history tables can be dropped afterwards to free memory.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ml_origin_features AS
        WITH origin_series AS (
            SELECT s.*, o.origin
            FROM ml_series AS s
            CROSS JOIN ml_snapshot_origins AS o
        ),
        origin_history_base AS (
            SELECT
                s.*,
                f.* EXCLUDE (ARTIKEL_ID, MARKT_ID),
                DATE_DIFF('day', f.last_positive_period, s.origin)
                    AS calendar_days_since_last_demand,
                DATE_DIFF('day', f.last_action_period, s.origin)
                    AS days_since_last_action
            FROM origin_series AS s
            ASOF LEFT JOIN ml_series_features AS f
                ON s.ARTIKEL_ID = f.ARTIKEL_ID
                AND s.MARKT_ID = f.MARKT_ID
                AND s.origin > f.feature_date
            -- Prune immature series before every later ASOF join so those
            -- joins run only over series that survive the maturity gate.
            WHERE f.active_days >= {int(design.min_active_days)}
        ),
        origin_history_with_rates AS (
            SELECT
                h.*,
                rates.rolling_6_mean,
                rates.rolling_24_mean,
                rates.demand_rate_last_6,
                rates.demand_rate_last_12,
                rates.demand_rate_last_24,
                rates.event_window_share_last_24
            FROM origin_history_base AS h
            ASOF LEFT JOIN ml_active_series_recent_features AS rates
                ON h.ARTIKEL_ID = rates.ARTIKEL_ID
                AND h.MARKT_ID = rates.MARKT_ID
                AND h.origin > rates.feature_date
        ),
        origin_history_with_non_event_rates AS (
            SELECT
                h.*,
                non_event.rolling_24_mean_non_event
            FROM origin_history_with_rates AS h
            ASOF LEFT JOIN ml_active_series_non_event_features AS non_event
                ON h.ARTIKEL_ID = non_event.ARTIKEL_ID
                AND h.MARKT_ID = non_event.MARKT_ID
                AND h.origin > non_event.feature_date
        ),
        origin_history_with_gaps AS (
            SELECT
                h.*,
                g.completed_gap_count,
                g.historical_gap_p90,
                g.historical_max_gap
            FROM origin_history_with_non_event_rates AS h
            ASOF LEFT JOIN ml_gap_statistics AS g
                ON h.ARTIKEL_ID = g.ARTIKEL_ID
                AND h.MARKT_ID = g.MARKT_ID
                AND h.origin > g.gap_feature_date
        ),
        origin_history_with_reference AS (
            SELECT
                h.* EXCLUDE (
                    completed_gap_count,
                    historical_gap_p90,
                    historical_max_gap
                ),
                CASE
                    WHEN h.completed_gap_count >= {MIN_COMPLETED_GAPS_FOR_P90}
                        THEN h.historical_gap_p90
                    ELSE COALESCE(h.historical_max_gap, h.active_zero_demand_gap)
                END AS historical_p90_gap
            FROM origin_history_with_gaps AS h
        ),
        origin_history AS (
            SELECT
                h.*,
                CASE
                    WHEN h.active_zero_demand_gap = 0 THEN 0.0
                    ELSE h.active_zero_demand_gap::DOUBLE
                        / NULLIF(h.historical_p90_gap, 0)
                END AS current_gap_over_historical_p90_gap
            FROM origin_history_with_reference AS h
        ),
        origin_history_with_product AS (
            SELECT h.*, x.product_cross_store_mean_28
            FROM origin_history AS h
            ASOF LEFT JOIN ml_product_features AS x
                ON h.ARTIKEL_ID = x.ARTIKEL_ID
                AND h.origin > x.feature_date
        ),
        origin_history_with_product_profile AS (
            SELECT h.*, x.product_cross_store_open_mean_48
            FROM origin_history_with_product AS h
            ASOF LEFT JOIN ml_product_open_features AS x
                ON h.ARTIKEL_ID = x.ARTIKEL_ID
                AND h.origin > x.feature_date
        ),
        origin_history_with_store_category AS (
            SELECT h.*, x.store_category_mean_28
            FROM origin_history_with_product_profile AS h
            ASOF LEFT JOIN ml_store_category_features AS x
                ON h.MARKT_ID = x.MARKT_ID
                AND h.category_id = x.category_id
                AND h.origin > x.feature_date
        ),
        origin_history_with_group_action AS (
            SELECT h.*, a.mean_action_lift_in_sourcing_group
            FROM origin_history_with_store_category AS h
            ASOF LEFT JOIN ml_sourcing_group_action_features AS a
                ON h.sourcing_group = a.sourcing_group
                AND h.origin > a.feature_date
        )
        SELECT * FROM origin_history_with_group_action
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_origin_weekday_features AS
        WITH probe AS (
            SELECT
                series_origin.ARTIKEL_ID,
                series_origin.MARKT_ID,
                series_origin.origin,
                weekday.target_weekday::INTEGER AS target_weekday
            FROM (
                SELECT DISTINCT ARTIKEL_ID, MARKT_ID, origin
                FROM ml_origin_features
            ) AS series_origin
            CROSS JOIN range(1, 8) AS weekday(target_weekday)
        )
        SELECT
            probe.ARTIKEL_ID,
            probe.MARKT_ID,
            probe.origin,
            probe.target_weekday,
            w.same_weekday_mean_4,
            w.same_weekday_mean_8
        FROM probe
        ASOF LEFT JOIN ml_series_weekday_features AS w
            ON probe.ARTIKEL_ID = w.ARTIKEL_ID
            AND probe.MARKT_ID = w.MARKT_ID
            AND probe.target_weekday = w.target_weekday
            AND probe.origin > w.feature_date
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_origin_product_weekday_features AS
        WITH probe AS (
            SELECT
                product_origin.ARTIKEL_ID,
                product_origin.origin,
                weekday.target_weekday::INTEGER AS target_weekday
            FROM (
                SELECT DISTINCT ARTIKEL_ID, origin
                FROM ml_origin_features
            ) AS product_origin
            CROSS JOIN range(1, 8) AS weekday(target_weekday)
        )
        SELECT
            probe.ARTIKEL_ID,
            probe.origin,
            probe.target_weekday,
            x.product_weekday_cross_store_mean_8
        FROM probe
        ASOF LEFT JOIN ml_product_weekday_features AS x
            ON probe.ARTIKEL_ID = x.ARTIKEL_ID
            AND probe.target_weekday = x.target_weekday
            AND probe.origin > x.feature_date
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ml_origin_event_lifts AS
        WITH probe AS (
            SELECT
                origin,
                (
                    origin + target_offset.day_offset * INTERVAL 1 DAY
                )::DATE AS target_period
            FROM ml_snapshot_origins
            CROSS JOIN range({int(design.forecast_horizon_days)})
                AS target_offset(day_offset)
        ),
        event_probe AS (
            SELECT
                probe.origin,
                probe.target_period,
                calendar.calendar_event_key,
                calendar.calendar_event_closure_block_length,
                calendar.days_to_nearest_event AS event_day_offset
            FROM probe
            INNER JOIN ml_calendar AS calendar
                ON probe.target_period = calendar.period
            WHERE calendar.holiday_event_window <> 'none'
        ),
        event_cells AS (
            SELECT DISTINCT
                origin,
                calendar_event_key,
                calendar_event_closure_block_length
            FROM event_probe
        ),
        event_history_at_origin AS (
            SELECT
                cells.origin,
                cells.calendar_event_key,
                cells.calendar_event_closure_block_length,
                pooled.feature_date,
                pooled.event_day_offset,
                pooled.event_weekday,
                pooled.ARTIKEL_ID,
                pooled.sourcing_group,
                pooled.category_id,
                pooled.event_observation_count,
                pooled.event_positive_observation_count,
                pooled.event_positive_demand_sum
            FROM event_cells AS cells
            LEFT JOIN ml_pooled_event_statistics AS pooled
                ON cells.calendar_event_key = pooled.calendar_event_key
                AND cells.calendar_event_closure_block_length
                    = pooled.calendar_event_closure_block_length
                AND pooled.feature_date < cells.origin
        ),
        context_at_origin AS (
            SELECT
                event.*,
                article.baseline_observation_count
                    AS article_baseline_observation_count,
                article.baseline_positive_observation_count
                    AS article_baseline_positive_observation_count,
                article.baseline_positive_demand_sum
                    AS article_baseline_positive_demand_sum,
                coarse.baseline_observation_count
                    AS coarse_baseline_observation_count,
                coarse.baseline_positive_observation_count
                    AS coarse_baseline_positive_observation_count,
                coarse.baseline_positive_demand_sum
                    AS coarse_baseline_positive_demand_sum
            FROM event_history_at_origin AS event
            ASOF LEFT JOIN ml_pooled_article_baseline_statistics AS article
                ON event.ARTIKEL_ID = article.ARTIKEL_ID
                AND event.event_weekday = article.baseline_weekday
                AND event.origin > article.feature_date
            ASOF LEFT JOIN ml_pooled_coarse_baseline_statistics AS coarse
                ON event.event_weekday = coarse.baseline_weekday
                AND event.sourcing_group = coarse.sourcing_group
                AND event.category_id = coarse.category_id
                AND event.origin > coarse.feature_date
        ),
        resolved_baselines AS (
            SELECT
                context.*,
                CASE
                    WHEN article_baseline_observation_count
                            >= {EVENT_MIN_ARTICLE_BASELINE_OBSERVATIONS}
                        AND article_baseline_positive_observation_count > 0
                    THEN article_baseline_positive_observation_count::DOUBLE
                        / article_baseline_observation_count
                    ELSE coarse_baseline_positive_observation_count::DOUBLE
                        / NULLIF(coarse_baseline_observation_count, 0)
                END AS baseline_occurrence_rate,
                CASE
                    WHEN article_baseline_observation_count
                            >= {EVENT_MIN_ARTICLE_BASELINE_OBSERVATIONS}
                        AND article_baseline_positive_observation_count > 0
                        AND article_baseline_positive_demand_sum > 0
                    THEN article_baseline_positive_demand_sum
                        / article_baseline_positive_observation_count
                    ELSE coarse_baseline_positive_demand_sum
                        / NULLIF(coarse_baseline_positive_observation_count, 0)
                END AS baseline_positive_mean
            FROM context_at_origin AS context
        ),
        contributions AS (
            SELECT
                origin,
                calendar_event_key,
                calendar_event_closure_block_length,
                event_day_offset,
                feature_date,
                event_observation_count,
                CASE WHEN baseline_occurrence_rate > 0
                    THEN event_positive_observation_count END
                    AS occurrence_numerator,
                CASE WHEN baseline_occurrence_rate > 0
                    THEN event_observation_count * baseline_occurrence_rate END
                    AS occurrence_denominator,
                CASE WHEN baseline_positive_mean > 0
                    THEN event_positive_demand_sum END AS quantity_numerator,
                CASE WHEN baseline_positive_mean > 0
                    THEN event_positive_observation_count * baseline_positive_mean END
                    AS quantity_denominator
            FROM resolved_baselines
        ),
        cell_components AS (
            SELECT
                origin,
                calendar_event_key,
                calendar_event_closure_block_length,
                SUM(event_observation_count)::BIGINT AS row_count,
                COUNT(DISTINCT feature_date)::INTEGER AS date_count,
                SUM(occurrence_numerator)::DOUBLE
                    / NULLIF(SUM(occurrence_denominator), 0)
                    AS occurrence_lift,
                SUM(quantity_numerator)
                    / NULLIF(SUM(quantity_denominator), 0)
                    AS quantity_lift
            FROM contributions
            GROUP BY
                origin,
                calendar_event_key,
                calendar_event_closure_block_length
        ),
        position_components AS (
            SELECT
                origin,
                calendar_event_key,
                calendar_event_closure_block_length,
                event_day_offset,
                COUNT(DISTINCT feature_date)::INTEGER AS date_count,
                SUM(occurrence_numerator)::DOUBLE
                    / NULLIF(SUM(occurrence_denominator), 0)
                    AS occurrence_lift,
                SUM(quantity_numerator)
                    / NULLIF(SUM(quantity_denominator), 0)
                    AS quantity_lift
            FROM contributions
            GROUP BY
                origin,
                calendar_event_key,
                calendar_event_closure_block_length,
                event_day_offset
        ),
        selected AS (
            SELECT
                probe.origin,
                probe.target_period,
                cell.row_count AS event_lift_pooled_cell_row_count,
                cell.date_count AS event_lift_pooled_cell_date_count,
                CASE
                    WHEN cell.date_count >= {EVENT_MIN_POOLED_CELL_DATES}
                    THEN cell.occurrence_lift
                END AS event_lift_pooled_occurrence,
                CASE
                    WHEN cell.date_count >= {EVENT_MIN_POOLED_CELL_DATES}
                    THEN cell.quantity_lift
                END AS event_lift_pooled_quantity,
                position.date_count AS event_position_lift_cell_date_count,
                CASE
                    WHEN position.occurrence_lift IS NOT NULL
                        AND cell.date_count >= {EVENT_MIN_POOLED_CELL_DATES}
                        AND cell.occurrence_lift IS NOT NULL
                    THEN (
                        position.date_count * position.occurrence_lift
                        + {EVENT_POSITION_SHRINKAGE_PRIOR_WEIGHT}
                            * cell.occurrence_lift
                    ) / (
                        position.date_count
                        + {EVENT_POSITION_SHRINKAGE_PRIOR_WEIGHT}
                    )
                    WHEN position.date_count >= {EVENT_MIN_POSITION_CELL_DATES}
                    THEN position.occurrence_lift
                END AS event_position_lift_occurrence,
                CASE
                    WHEN position.quantity_lift IS NOT NULL
                        AND cell.date_count >= {EVENT_MIN_POOLED_CELL_DATES}
                        AND cell.quantity_lift IS NOT NULL
                    THEN (
                        position.date_count * position.quantity_lift
                        + {EVENT_POSITION_SHRINKAGE_PRIOR_WEIGHT}
                            * cell.quantity_lift
                    ) / (
                        position.date_count
                        + {EVENT_POSITION_SHRINKAGE_PRIOR_WEIGHT}
                    )
                    WHEN position.date_count >= {EVENT_MIN_POSITION_CELL_DATES}
                    THEN position.quantity_lift
                END AS event_position_lift_quantity
            FROM event_probe AS probe
            LEFT JOIN cell_components AS cell
                ON probe.origin = cell.origin
                AND probe.calendar_event_key = cell.calendar_event_key
                AND probe.calendar_event_closure_block_length
                    = cell.calendar_event_closure_block_length
            LEFT JOIN position_components AS position
                ON probe.origin = position.origin
                AND probe.calendar_event_key = position.calendar_event_key
                AND probe.calendar_event_closure_block_length
                    = position.calendar_event_closure_block_length
                AND probe.event_day_offset = position.event_day_offset
        )
        SELECT
            origin,
            target_period,
            event_lift_pooled_cell_row_count,
            event_lift_pooled_cell_date_count,
            event_lift_pooled_occurrence,
            event_lift_pooled_quantity,
            event_lift_pooled_occurrence * event_lift_pooled_quantity
                AS event_lift_pooled_total,
            event_position_lift_cell_date_count,
            event_position_lift_occurrence,
            event_position_lift_quantity,
            event_position_lift_occurrence * event_position_lift_quantity
                AS event_position_lift_total
        FROM selected
        """
    )
    # The feature query runs on separate cursors when origins are materialized
    # in parallel. Cursors cannot see this connection's TEMP tables, so the
    # two benchmark-owned inputs it needs are copied into the shared schema.
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_target_rows AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            period,
            demand,
            is_active,
            reason_closed,
            action_flag
        FROM benchmark_daily_rows
        WHERE period IN (SELECT target_period FROM ml_target_dates)
        """
    )
    history_tables = {
        row[0] for row in con.execute("SHOW TABLES").fetchall()
    }
    if "benchmark_origin_history" not in history_tables:
        raise RuntimeError(
            "benchmark_origin_history is missing; run create_history_features "
            "and _create_assessed_origins before create_feature_tables"
        )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_origin_history_scale AS
        SELECT ARTIKEL_ID, MARKT_ID, origin, seasonal_mase_scale
        FROM benchmark_origin_history
        """
    )
    # Promotion context from the raw-line extract. Campaign windows and prices
    # on the target day belong to the same known-ahead promotion schedule as
    # action_on_forecast_day; every statistic below that summarizes history is
    # restricted to rows strictly before the origin.
    con.register("ml_article_class_frame", _article_product_classes(design))
    con.execute(
        "CREATE OR REPLACE TABLE ml_article_class AS "
        "SELECT * FROM ml_article_class_frame"
    )
    con.register("ml_action_days_frame", _article_store_action_days())
    con.register("ml_regular_prices_frame", _article_day_regular_prices())
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_action_article_days AS
        SELECT
            ARTIKEL_ID,
            period::DATE AS period,
            MIN(gueltig_von)::DATE AS gueltig_von,
            AVG(action_unit_price) AS action_unit_price
        FROM ml_action_days_frame
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_regular_prices AS
        SELECT ARTIKEL_ID, period::DATE AS period, regular_unit_price
        FROM ml_regular_prices_frame
        WHERE regular_unit_price > 0
        """
    )
    # Trailing 28-day regular price per origin, the depth reference for the
    # target-day discount; strictly pre-origin.
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_article_regular_price_28 AS
        SELECT
            o.origin,
            r.ARTIKEL_ID,
            AVG(r.regular_unit_price) AS regular_price_28d
        FROM ml_snapshot_origins AS o
        JOIN ml_regular_prices AS r
            ON r.period < o.origin
            AND r.period >= (o.origin - INTERVAL 28 DAY)
        GROUP BY 1, 2
        """
    )
    # Historical discount depth per action day (each day referenced against
    # the article's regular price of the 28 days before that action day), and
    # its strictly-pre-origin per-article mean.
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_article_depth_stats AS
        WITH day_depth AS (
            SELECT
                a.ARTIKEL_ID,
                a.period,
                1.0 - a.action_unit_price / NULLIF(AVG(r.regular_unit_price), 0)
                    AS discount_depth
            FROM ml_action_article_days AS a
            JOIN ml_regular_prices AS r
                ON r.ARTIKEL_ID = a.ARTIKEL_ID
                AND r.period < a.period
                AND r.period >= (a.period - INTERVAL 28 DAY)
            WHERE a.action_unit_price IS NOT NULL
            GROUP BY a.ARTIKEL_ID, a.period, a.action_unit_price
        )
        SELECT
            o.origin,
            d.ARTIKEL_ID,
            AVG(d.discount_depth) AS typical_depth,
            COUNT(*) AS depth_obs
        FROM ml_snapshot_origins AS o
        JOIN day_depth AS d ON d.period < o.origin
        WHERE d.discount_depth IS NOT NULL
        GROUP BY 1, 2
        """
    )
    # Per-article action lift over all strictly-pre-origin active rows,
    # pooled across stores; shrinkage toward the sourcing-group lift happens
    # in the feature query where both values meet.
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_article_action_stats AS
        WITH daily AS (
            SELECT
                ARTIKEL_ID,
                period,
                SUM(demand) FILTER (WHERE is_active AND action_flag = 1)
                    AS action_demand,
                COUNT(*) FILTER (WHERE is_active AND action_flag = 1)
                    AS action_obs,
                SUM(demand) FILTER (WHERE is_active AND action_flag = 0)
                    AS regular_demand,
                COUNT(*) FILTER (WHERE is_active AND action_flag = 0)
                    AS regular_obs
            FROM benchmark_daily_rows
            GROUP BY 1, 2
        )
        SELECT
            o.origin,
            d.ARTIKEL_ID,
            SUM(d.action_obs) AS action_obs,
            (SUM(d.action_demand) / NULLIF(SUM(d.action_obs), 0))
                / NULLIF(
                    SUM(d.regular_demand) / NULLIF(SUM(d.regular_obs), 0), 0
                ) - 1.0 AS raw_lift
        FROM ml_snapshot_origins AS o
        JOIN daily AS d ON d.period < o.origin
        GROUP BY 1, 2
        """
    )
    # Deepest historically-typical discount among the class articles promoted
    # at the store on each target day (the article itself included).
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ml_class_promoted_depth AS
        SELECT
            tp.origin,
            t.MARKT_ID,
            t.period,
            cls.product_class,
            MAX(ds.typical_depth) AS max_class_depth
        FROM (
            SELECT
                o.origin,
                (o.origin + off.day_offset * INTERVAL 1 DAY)::DATE AS period
            FROM ml_snapshot_origins AS o
            CROSS JOIN range({int(design.forecast_horizon_days)})
                AS off(day_offset)
        ) AS tp
        JOIN ml_target_rows AS t
            ON t.period = tp.period AND t.action_flag = 1
        JOIN ml_article_class AS cls ON t.ARTIKEL_ID = cls.ARTIKEL_ID
        JOIN ml_article_depth_stats AS ds
            ON ds.origin = tp.origin AND ds.ARTIKEL_ID = t.ARTIKEL_ID
        GROUP BY 1, 2, 3, 4
        """
    )
    for resolved_table in (
        "ml_active_series_recent_features",
        "ml_active_series_non_event_features",
        "ml_active_series_event_history",
        "ml_pooled_event_statistics",
        "ml_pooled_article_baseline_statistics",
        "ml_pooled_coarse_baseline_statistics",
        "ml_series_features",
        "ml_gap_statistics",
        "ml_series_weekday_features",
        "ml_product_features",
        "ml_product_open_features",
        "ml_product_weekday_features",
        "ml_store_category_features",
        "ml_sourcing_group_action_features",
        "ml_product_daily",
        "ml_annual_reference_dates",
    ):
        con.execute(f"DROP TABLE IF EXISTS {resolved_table}")


def _normalized_origins(origins: Iterable[object]) -> pd.DataFrame:
    values = pd.DatetimeIndex(origins).normalize().unique().sort_values()
    if len(values) == 0:
        raise ValueError("At least one origin is required")
    return pd.DataFrame({"origin": values.date})


def _feature_query_statement(
    con: duckdb.DuckDBPyConnection,
    origins: Iterable[object],
    design: BenchmarkDesign,
    *,
    order_results: bool = True,
) -> tuple[str, list[object]]:
    """Return the registered-origin feature query and its parameters."""
    con.register("ml_requested_origin_frame", _normalized_origins(origins))
    con.execute(
        "CREATE OR REPLACE TEMP TABLE ml_requested_origins AS "
        "SELECT * FROM ml_requested_origin_frame"
    )
    try:
        uncovered = con.execute(
            """
            SELECT COUNT(*) FROM ml_requested_origins
            WHERE origin NOT IN (SELECT origin FROM ml_snapshot_origins)
            """
        ).fetchone()[0]
    except duckdb.CatalogException as error:
        raise RuntimeError(
            "Feature snapshots are missing; run create_feature_tables with "
            "the requested origins first"
        ) from error
    if uncovered:
        raise ValueError(
            "Feature query requested origins outside the snapshot set; "
            "rerun create_feature_tables with all required origins"
        )
    return (
        f"""
        WITH origin_history_with_action_lift AS (
            SELECT h.*
            FROM ml_origin_features AS h
            INNER JOIN ml_requested_origins USING (origin)
        ),
        target_dates AS (
            SELECT
                h.*,
                (
                    h.origin
                    + target_offset.day_offset * INTERVAL 1 DAY
                )::DATE AS target_period
            FROM origin_history_with_action_lift AS h
            CROSS JOIN range({int(design.forecast_horizon_days)})
                AS target_offset(day_offset)
        ),
        targets AS (
            SELECT
                h.*,
                t.period,
                t.demand AS actual,
                t.is_active,
                t.reason_closed,
                EXTRACT(ISODOW FROM t.period)::INTEGER AS target_weekday,
                EXTRACT(WEEK FROM t.period)::INTEGER AS iso_week,
                EXTRACT(MONTH FROM t.period)::INTEGER AS month,
                closure.closed_days_next_1,
                closure.closed_days_next_2,
                closure.closed_days_next_3,
                closure.closed_days_prev_1,
                closure.closed_days_prev_2,
                closure.closed_days_prev_3,
                t.action_flag::INTEGER AS action_on_forecast_day,
                MAX(t.action_flag) OVER (
                    PARTITION BY h.ARTIKEL_ID, h.MARKT_ID, h.origin
                )::INTEGER AS action_during_horizon,
                CASE
                    WHEN t.action_flag = 1 AND act.action_unit_price IS NOT NULL
                        AND reg.regular_price_28d > 0
                    THEN 1.0 - act.action_unit_price / reg.regular_price_28d
                    WHEN t.action_flag = 1 THEN NULL
                    ELSE 0.0
                END AS action_depth_on_forecast_day,
                CASE
                    WHEN t.action_flag = 1 AND act.gueltig_von IS NOT NULL
                        AND t.period >= act.gueltig_von
                    THEN LEAST(
                        DATE_DIFF('day', act.gueltig_von, t.period) + 1, 28
                    )
                    WHEN t.action_flag = 1 THEN NULL
                    ELSE 0
                END AS action_flight_day,
                CASE
                    WHEN lift.raw_lift IS NULL
                        AND h.mean_action_lift_in_sourcing_group IS NULL
                    THEN NULL
                    ELSE (
                        COALESCE(lift.raw_lift, 0)
                            * COALESCE(lift.action_obs, 0)
                        + COALESCE(h.mean_action_lift_in_sourcing_group, 0)
                            * {ARTICLE_ACTION_LIFT_PRIOR_OBS}
                    ) / (
                        COALESCE(lift.action_obs, 0)
                        + {ARTICLE_ACTION_LIFT_PRIOR_OBS}
                    )
                END AS article_action_lift,
                dep.typical_depth AS article_typical_action_depth,
                COALESCE(comp.max_class_depth, 0) AS same_class_promoted_depth,
                lags.same_weekday_lag_7,
                lags.same_weekday_lag_14
            FROM target_dates AS h
            INNER JOIN ml_target_rows AS t
                ON h.ARTIKEL_ID = t.ARTIKEL_ID
                AND h.MARKT_ID = t.MARKT_ID
                AND t.period = h.target_period
            INNER JOIN ml_store_closure_features AS closure
                ON t.MARKT_ID = closure.MARKT_ID
                AND t.period = closure.period
            LEFT JOIN ml_action_article_days AS act
                ON act.ARTIKEL_ID = t.ARTIKEL_ID
                AND act.period = t.period
            LEFT JOIN ml_article_regular_price_28 AS reg
                ON reg.origin = h.origin
                AND reg.ARTIKEL_ID = t.ARTIKEL_ID
            LEFT JOIN ml_article_action_stats AS lift
                ON lift.origin = h.origin
                AND lift.ARTIKEL_ID = t.ARTIKEL_ID
            LEFT JOIN ml_article_depth_stats AS dep
                ON dep.origin = h.origin
                AND dep.ARTIKEL_ID = t.ARTIKEL_ID
            LEFT JOIN ml_article_class AS cls
                ON cls.ARTIKEL_ID = t.ARTIKEL_ID
            LEFT JOIN ml_class_promoted_depth AS comp
                ON comp.origin = h.origin
                AND comp.MARKT_ID = t.MARKT_ID
                AND comp.period = t.period
                AND comp.product_class = cls.product_class
            LEFT JOIN ml_series_lag_features AS lags
                ON t.ARTIKEL_ID = lags.ARTIKEL_ID
                AND t.MARKT_ID = lags.MARKT_ID
                AND t.period = lags.period
        ),
        with_annual_history AS (
            SELECT
                t.*,
                COALESCE(a.annual_lookup_days_available, 0)::INTEGER
                    AS annual_lookup_days_available,
                a.same_weekday_last_year_mean,
                w.same_week_last_year_mean,
                x.product_cross_store_same_weekday_last_year_mean,
                e.event_lift_series,
                pooled.event_lift_pooled_occurrence,
                pooled.event_lift_pooled_quantity,
                pooled.event_lift_pooled_total,
                pooled.event_position_lift_occurrence,
                pooled.event_position_lift_quantity,
                pooled.event_position_lift_total
            FROM targets AS t
            LEFT JOIN ml_series_annual_features AS a
                ON t.ARTIKEL_ID = a.ARTIKEL_ID
                AND t.MARKT_ID = a.MARKT_ID
                AND t.period = a.target_period
            LEFT JOIN ml_series_weekly_annual_features AS w
                ON t.ARTIKEL_ID = w.ARTIKEL_ID
                AND t.MARKT_ID = w.MARKT_ID
                AND t.period = w.target_period
            LEFT JOIN ml_product_annual_features AS x
                ON t.ARTIKEL_ID = x.ARTIKEL_ID
                AND t.period = x.target_period
            LEFT JOIN ml_series_event_lifts AS e
                ON t.ARTIKEL_ID = e.ARTIKEL_ID
                AND t.MARKT_ID = e.MARKT_ID
                AND t.period = e.target_period
            LEFT JOIN ml_origin_event_lifts AS pooled
                ON t.origin = pooled.origin
                AND t.period = pooled.target_period
        ),
        with_weekday AS (
            SELECT t.*, w.same_weekday_mean_4, w.same_weekday_mean_8
            FROM with_annual_history AS t
            LEFT JOIN (
                SELECT * FROM ml_origin_weekday_features
                WHERE origin IN (SELECT origin FROM ml_requested_origins)
            ) AS w
                ON t.ARTIKEL_ID = w.ARTIKEL_ID
                AND t.MARKT_ID = w.MARKT_ID
                AND t.origin = w.origin
                AND t.target_weekday = w.target_weekday
        ),
        with_product_weekday AS (
            SELECT p.*, x.product_weekday_cross_store_mean_8
            FROM with_weekday AS p
            LEFT JOIN (
                SELECT * FROM ml_origin_product_weekday_features
                WHERE origin IN (SELECT origin FROM ml_requested_origins)
            ) AS x
                ON p.ARTIKEL_ID = x.ARTIKEL_ID
                AND p.origin = x.origin
                AND p.target_weekday = x.target_weekday
        )
        SELECT
            p.ARTIKEL_ID,
            p.MARKT_ID,
            p.sourcing_group,
            p.category_id,
            p.origin,
            p.period,
            p.target_weekday,
            p.iso_week,
            p.month,
            p.closed_days_next_1,
            p.closed_days_next_2,
            p.closed_days_next_3,
            p.closed_days_prev_1,
            p.closed_days_prev_2,
            p.closed_days_prev_3,
            p.is_active,
            p.reason_closed,
            c.is_public_holiday,
            c.days_to_nearest_event,
            c.holiday_event_window,
            c.event_name,
            p.action_on_forecast_day,
            p.action_during_horizon,
            p.action_depth_on_forecast_day,
            p.action_flight_day,
            p.article_action_lift,
            CASE
                WHEN p.action_on_forecast_day = 1 THEN p.article_action_lift
                ELSE 0.0
            END AS action_expected_lift,
            p.article_typical_action_depth,
            p.same_class_promoted_depth,
            p.days_since_last_action,
            p.actions_last_28d,
            p.mean_action_lift_in_sourcing_group,
            p.active_days AS active_days_before_origin,
            p.demand_days AS demand_days_before_origin,
            p.active_zero_demand_gap,
            p.demand_rate_last_6,
            p.demand_rate_last_12,
            p.demand_rate_last_24,
            p.historical_p90_gap,
            p.current_gap_over_historical_p90_gap,
            p.same_weekday_lag_7,
            p.same_weekday_lag_14,
            p.rolling_6_mean,
            p.rolling_24_mean,
            p.rolling_24_mean_non_event,
            p.event_window_share_last_24,
            p.rolling_28_demand_rate,
            p.same_weekday_mean_4,
            p.same_weekday_mean_8,
            p.annual_lookup_days_available,
            p.same_weekday_last_year_mean,
            p.same_week_last_year_mean,
            p.product_cross_store_same_weekday_last_year_mean,
            p.event_lift_series,
            p.event_lift_pooled_occurrence,
            p.event_lift_pooled_quantity,
            p.event_lift_pooled_total,
            p.event_position_lift_occurrence,
            p.event_position_lift_quantity,
            p.event_position_lift_total,
            p.ADI,
            p.CV2,
            p.product_cross_store_mean_28,
            p.product_weekday_cross_store_mean_8
                / NULLIF(p.product_cross_store_open_mean_48, 0)
                AS product_weekday_profile_value,
            p.store_category_mean_28,
            w.temperature_anomaly_c,
            COALESCE(sensitivity.article_temperature_sensitivity, 0)
                * GREATEST(w.temperature_anomaly_c, 0)
                AS temperature_expected_lift,
            p.rolling_28_demand_rate AS recent_occurrence_rate,
            p.calendar_days_since_last_demand,
            h.seasonal_mase_scale,
            p.actual,
            CASE
                WHEN p.target_mean > 0 THEN p.actual / p.target_mean
                ELSE p.actual
            END AS normalized_actual,
            CASE
                WHEN p.target_mean > 0 THEN p.target_mean
                ELSE 1.0
            END AS target_mean
        FROM with_product_weekday AS p
        LEFT JOIN ml_store_subdivision AS store_state
            ON p.MARKT_ID = store_state.MARKT_ID
        INNER JOIN ml_subdivision_calendar AS c
            ON c.period = p.period
            AND c.subdivision = COALESCE(store_state.subdivision, 'NI')
        LEFT JOIN ml_store_weather_anomalies AS w
            ON p.MARKT_ID = w.MARKT_ID
            AND p.origin = w.origin
            AND p.period = w.period
        LEFT JOIN ml_article_temperature_sensitivity AS sensitivity
            ON p.ARTIKEL_ID = sensitivity.ARTIKEL_ID
            AND p.origin = sensitivity.origin
        LEFT JOIN ml_origin_history_scale AS h
            ON p.ARTIKEL_ID = h.ARTIKEL_ID
            AND p.MARKT_ID = h.MARKT_ID
            AND p.origin = h.origin
        {"ORDER BY p.origin, p.ARTIKEL_ID, p.MARKT_ID, p.period" if order_results else ""}
        """,
        [],
    )


def _execute_feature_query(
    con: duckdb.DuckDBPyConnection,
    origins: Iterable[object],
    design: BenchmarkDesign,
    *,
    order_results: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Execute the feature query and leave its result ready for fetching."""
    query, parameters = _feature_query_statement(
        con, origins, design, order_results=order_results
    )
    return con.execute(query, parameters)


def make_feature_frame(
    con: duckdb.DuckDBPyConnection,
    origins: Iterable[object],
    design: BenchmarkDesign,
) -> pd.DataFrame:
    """Return direct-horizon feature rows for mature series at supplied origins."""
    return _execute_feature_query(con, origins, design).fetchdf()


def historical_training_origins(
    con: duckdb.DuckDBPyConnection,
    design: BenchmarkDesign,
    evaluation_start: object,
    origin_count: int,
) -> pd.DatetimeIndex:
    """Return origin-aligned historical samples strictly before evaluation."""
    first_observed = pd.Timestamp(
        con.execute("SELECT MIN(period) FROM benchmark_daily_rows").fetchone()[0]
    )
    evaluation_start = pd.Timestamp(evaluation_start).normalize()
    latest = evaluation_start - pd.Timedelta(days=design.origin_spacing_days)
    earliest = first_observed + pd.Timedelta(days=design.min_active_days)
    candidates: list[pd.Timestamp] = []
    origin = latest
    while origin >= earliest and len(candidates) < int(origin_count):
        candidates.append(origin)
        origin -= pd.Timedelta(days=design.origin_spacing_days)
    if len(candidates) < int(origin_count):
        raise RuntimeError(
            "Insufficient historical origins to train the global model: "
            f"required {int(origin_count)}, found {len(candidates)}"
        )
    return pd.DatetimeIndex(sorted(candidates), name="origin")


def get_or_materialize_feature_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    origins: Iterable[object],
    design: BenchmarkDesign,
    feature_path: Path | str = DEFAULT_FEATURES_PATH,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Return a feature frame for the requested origins, reusing disk cache."""
    required_origins = pd.DatetimeIndex(origins).normalize().unique().sort_values()
    if len(required_origins) == 0:
        raise ValueError("At least one origin is required")

    path = _normalize_feature_path(feature_path)
    if not force_recompute and path.exists():
        try:
            feature_frame = pd.read_parquet(path)
        except Exception:
            feature_frame = materialize_features_for_origins(
                con,
                origins=required_origins,
                design=design,
                feature_path=path,
            )
        else:
            existing_origins = pd.to_datetime(feature_frame["origin"]).dt.normalize().unique()
            required_columns = _required_materialized_columns()
            if (
                not pd.Index(required_origins).isin(existing_origins).all()
                or not required_columns.issubset(feature_frame.columns)
                or not REMOVED_FEATURE_COLUMNS.isdisjoint(feature_frame.columns)
            ):
                feature_frame = materialize_features_for_origins(
                    con,
                    origins=required_origins,
                    design=design,
                    feature_path=path,
                )
    else:
        feature_frame = materialize_features_for_origins(
            con,
            origins=required_origins,
            design=design,
            feature_path=path,
        )

    return feature_frame.loc[
        pd.to_datetime(feature_frame["origin"]).dt.normalize().isin(required_origins)
    ].copy()


def prepare_global_lightgbm_frames(
    *,
    design: BenchmarkDesign,
    evaluation_origins: Iterable[object],
    connection: duckdb.DuckDBPyConnection | None = None,
    data_dir: Path | None = None,
    config: GlobalLightGBMConfig | None = None,
    feature_dataset_path: Path | str = DEFAULT_FEATURES_PATH,
    force_feature_recompute: bool = False,
) -> GlobalLightGBMFrames:
    """Materialize leakage-safe weekly frames for expanding four-week refits.

    The first refit uses ``training_origins`` followed by
    ``validation_origins``. At each later refit, the previous validation block
    joins the expanding training set and the latest completed origins become
    validation. The fitted model evaluates ``refit_interval_origins`` weekly
    feature snapshots before it is refit.
    """
    config = GlobalLightGBMConfig() if config is None else config
    data_dir = design.data_dir if data_dir is None else Path(data_dir)
    con = duckdb.connect() if connection is None else connection
    # num_threads only limits LightGBM training; DuckDB keeps its default of
    # one thread per logical processor for the feature build.
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "benchmark_daily_rows" not in tables:
        prepare_daily_rows(con, data_dir=data_dir)
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    normalized_origins = pd.DatetimeIndex(evaluation_origins).normalize().unique()
    normalized_origins = normalized_origins.sort_values()
    if len(normalized_origins) == 0:
        raise ValueError("evaluation_origins cannot be empty")
    if design.origin_spacing_days != config.origin_spacing_days:
        raise ValueError(
            "LightGBM forecasts and features must use origins spaced "
            f"{config.origin_spacing_days} days apart"
        )
    if design.forecast_horizon_days != config.origin_spacing_days:
        raise ValueError("The LightGBM forecast horizon must be one full week")
    if len(normalized_origins) > config.test_origins:
        raise ValueError(
            f"LightGBM accepts at most {config.test_origins} test origins"
        )
    if len(normalized_origins) > 1:
        origin_deltas = normalized_origins[1:] - normalized_origins[:-1]
        if (origin_deltas != pd.Timedelta(days=config.origin_spacing_days)).any():
            raise ValueError(
                "LightGBM test forecasts and feature snapshots must be "
                f"generated every {config.origin_spacing_days} days"
            )
        previous_windows_end = normalized_origins[:-1] + pd.Timedelta(
            days=design.forecast_horizon_days
        )
        if (previous_windows_end > normalized_origins[1:]).any():
            raise ValueError(
                "Evaluation origins overlap: a preceding target window is not "
                "fully observed before the next refit"
            )
    if "benchmark_origin_history" not in tables:
        create_history_features(con)
        _create_assessed_origins(con, normalized_origins, design)

    con.execute("DROP TABLE IF EXISTS benchmark_row_features")
    con.execute("DROP TABLE IF EXISTS benchmark_weekday_features")
    con.execute("DROP TABLE IF EXISTS benchmark_positive_features")
    initial_history_origins = historical_training_origins(
        con,
        design,
        normalized_origins.min(),
        config.history_origins,
    )
    all_history_origins = initial_history_origins.append(normalized_origins[:-1])
    all_history_origins = all_history_origins.unique().sort_values()
    all_frame_origins = all_history_origins.append(normalized_origins)
    all_frame_origins = all_frame_origins.unique().sort_values()
    if force_feature_recompute or not _feature_cache_covers(
        con, feature_dataset_path, all_frame_origins
    ):
        create_feature_tables(con, origins=all_frame_origins, design=design)
    feature_frame = get_or_materialize_feature_frame(
        con,
        origins=all_frame_origins,
        design=design,
        feature_path=feature_dataset_path,
        force_recompute=force_feature_recompute,
    )
    history_frame = feature_frame.loc[
        pd.to_datetime(feature_frame["origin"]).dt.normalize().isin(all_history_origins)
    ].copy()
    evaluation_frame = feature_frame.loc[
        pd.to_datetime(feature_frame["origin"]).dt.normalize().isin(normalized_origins)
    ].copy()

    history_origin_values = pd.to_datetime(history_frame["origin"]).dt.normalize()
    evaluation_origin_values = pd.to_datetime(evaluation_frame["origin"]).dt.normalize()
    origin_frames: list[GlobalLightGBMFrames] = []
    for window in iter_lightgbm_origin_windows(
        initial_history_origins, normalized_origins, config
    ):
        training = history_frame.loc[
            history_origin_values.isin(window.training) & history_frame["is_active"]
        ].copy()
        validation = history_frame.loc[
            history_origin_values.isin(window.validation)
            & history_frame["is_active"]
        ].copy()
        evaluation = evaluation_frame.loc[
            evaluation_origin_values.isin(window.evaluation)
        ].copy()
        if training.empty or validation.empty or evaluation.empty:
            raise RuntimeError(
                "Training, validation, and evaluation samples must be nonempty "
                f"for refit origin {window.evaluation.min().date()}"
            )
        origin_frames.append(
            GlobalLightGBMFrames(
                training=training,
                validation=validation,
                evaluation=evaluation,
                training_origins=window.training.append(window.validation),
                evaluation_origin=pd.Timestamp(window.evaluation.min()),
            )
        )

    first = origin_frames[0]
    return GlobalLightGBMFrames(
        training=first.training,
        validation=first.validation,
        evaluation=evaluation_frame,
        training_origins=initial_history_origins,
        origin_frames=tuple(origin_frames),
    )
