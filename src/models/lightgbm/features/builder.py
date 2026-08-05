"""Feature definitions and engineering shared by all LightGBM models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dateutil.easter import easter

from src.data.preparation.distribute_sales_over_active_days import (
    create_germany_ni_holidays,
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
MAX_EVENT_OFFSET_DAYS = 10
FEATURE_ORIGIN_BATCH_SIZE = 4
REMOVED_FEATURE_COLUMNS = frozenset({"lag_364", "lag_371"})


def get_last_year_offset(current_date: object) -> int:
    """Return the Easter-aligned annual-history offset for a target date."""
    target = pd.Timestamp(current_date).normalize()
    easter_current = pd.Timestamp(easter(target.year))
    easter_previous = pd.Timestamp(easter(target.year - 1))
    easter_window_start = easter_current - pd.Timedelta(days=21)
    easter_window_end = easter_current + pd.Timedelta(days=64)
    if easter_window_start <= target <= easter_window_end:
        return int((easter_current - easter_previous).days)
    return 364


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
    # Historical action behavior (strictly before the origin)
    "days_since_last_action",
    "actions_last_28d",
    "mean_action_lift_in_sourcing_group",
    # Maturity and demand gaps
    "active_days_before_origin",
    "demand_days_before_origin",
    "demand_day_ratio",
    "active_zero_demand_gap",
    "demand_days_last_7",
    "demand_days_last_28",
    "demand_days_last_60",
    "historical_p90_gap",
    "current_gap_over_historical_p90_gap",
    # Local demand history
    "same_weekday_lag_7",
    "same_weekday_lag_14",
    "rolling_7_mean",
    "rolling_28_mean",
    "rolling_28_demand_rate",
    "same_weekday_mean_4",
    "same_weekday_mean_8",
    # Annual demand history
    "has_annual_history",
    "same_weekday_last_year_mean",
    "same_week_last_year_mean",
    "product_cross_store_same_weekday_last_year_mean",
    "same_event_offset_last_year_mean",
    # Demand regime
    "ADI",
    "CV2",
    # Cross-sectional context
    "product_cross_store_mean_28",
    "product_weekday_profile_value",
    "store_category_mean_28",
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
        "Niedersachsen public holiday: holiday date minus target date. Positive values "
        "indicate dates before the holiday and negative values dates after it; ties "
        "are resolved in favor of the earlier holiday."
    ),
    "holiday_event_window": (
        "Categorical position of the target date relative to its nearest Niedersachsen "
        "public holiday: holiday, 1-3 calendar days before, 1-3 calendar days after, "
        "or none when the absolute offset exceeds three days."
    ),
    "event_name": (
        "Name of the nearest Niedersachsen public holiday when its absolute calendar-"
        "day offset from the target is at most three; otherwise the category none."
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
    "demand_day_ratio": (
        "demand_days_before_origin / active_days_before_origin. The denominator contains "
        "only active observed rows, including active zero-demand rows; the ratio is "
        "missing when no active history exists."
    ),
    "active_zero_demand_gap": (
        "Number of active observed rows after the most recent active positive-demand "
        "row and strictly before the origin. Inactive and missing calendar dates do "
        "not increase the gap; before the first positive-demand row, all active "
        "observations are counted."
    ),
    "demand_days_last_7": (
        "Count of active positive-demand observations in the seven-calendar-day window "
        "ending on the last observed row before the origin. Inactive, zero-demand, and "
        "missing dates do not contribute."
    ),
    "demand_days_last_28": (
        "Count of active positive-demand observations in the 28-calendar-day window "
        "ending on the last observed row before the origin. Inactive, zero-demand, and "
        "missing dates do not contribute."
    ),
    "demand_days_last_60": (
        "Count of active positive-demand observations in the 60-calendar-day window "
        "ending on the last observed row before the origin. Inactive, zero-demand, and "
        "missing dates do not contribute."
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
        "Article-store demand on the exact calendar date target - 7 days. No activity "
        "filter is applied, so an observed inactive date contributes its recorded zero; "
        "a missing date produces a missing feature."
    ),
    "same_weekday_lag_14": (
        "Article-store demand on the exact calendar date target - 14 days. No activity "
        "filter is applied, so an observed inactive date contributes its recorded zero; "
        "a missing date produces a missing feature."
    ),
    "rolling_7_mean": (
        "Arithmetic mean of demand over the final seven observed article-store rows "
        "strictly before the origin. The denominator is the number of non-null demand "
        "values, without an activity filter: inactive rows recorded with zero are "
        "included, while missing calendar dates are absent and can make the row window "
        "extend more than seven calendar days."
    ),
    "rolling_28_mean": (
        "Arithmetic mean of demand over the final 28 observed article-store rows "
        "strictly before the origin. The denominator is the number of non-null demand "
        "values, without an activity filter: inactive rows recorded with zero are "
        "included, while missing calendar dates are absent and can make the row window "
        "extend more than 28 calendar days."
    ),
    "rolling_28_demand_rate": (
        "Share of active observations with demand > 0 within the final 28 observed "
        "article-store rows strictly before the origin. The denominator includes only "
        "active rows, including active zero-demand rows; inactive rows and missing "
        "dates are excluded."
    ),
    "same_weekday_mean_4": (
        "Arithmetic mean over up to the final four active article-store observations "
        "strictly before the origin whose ISO weekday matches the target weekday. The "
        "denominator is the available active observations, including zero-demand rows; "
        "inactive and missing dates are excluded."
    ),
    "same_weekday_mean_8": (
        "Arithmetic mean over up to the final eight active article-store observations "
        "strictly before the origin whose ISO weekday matches the target weekday. The "
        "denominator is the available active observations, including zero-demand rows; "
        "inactive and missing dates are excluded."
    ),
    "has_annual_history": (
        "Indicator that an article-store row exists on the exact calendar date target "
        "- A. Activity is not required, so an inactive row counts as history; a missing "
        "row does not. A is the day difference between current and previous Easter for "
        "targets from 21 days before through 64 days after Easter, and 364 otherwise."
    ),
    "same_weekday_last_year_mean": (
        "Arithmetic mean of article-store demand on the five exact calendar dates "
        "target - (A - 14), target - (A - 7), target - A, target - (A + 7), and "
        "target - (A + 14). A is the Easter-aware annual offset. The feature is only "
        "constructed when the target - A row exists, regardless of that row's activity. "
        "The averaging denominator contains only available active lookup rows among the "
        "five dates, including active zero-demand rows. Inactive and missing lookup "
        "dates are excluded; the result is missing when none of the five is active."
    ),
    "same_week_last_year_mean": (
        "Arithmetic mean of article-store demand over active observed rows in the "
        "reference Monday-Sunday calendar week whose Monday is target-week Monday - A "
        "days. A is the Easter-aware annual offset. The denominator is the number of "
        "available active daily rows, including active zero-demand rows; inactive and "
        "missing dates are excluded. The result is missing when the reference week has "
        "no active row."
    ),
    "product_cross_store_same_weekday_last_year_mean": (
        "For each of the five exact dates target - (A - 14), target - (A - 7), target "
        "- A, target - (A + 7), and target - (A + 14), article demand is first averaged "
        "across active stores. Each daily denominator therefore contains active store "
        "rows, including active zero-demand rows, while inactive stores are excluded. "
        "The feature is the arithmetic mean of the available non-null daily means; "
        "dates with no active store or no article row are excluded. The feature is "
        "only constructed when an article row exists at some store on target - A; "
        "that central row need not be active."
    ),
    "same_event_offset_last_year_mean": (
        "When the target is at most ten calendar days from its nearest Niedersachsen "
        "holiday, the signed target-to-holiday offset is mapped to the same holiday in "
        "the preceding year. The feature is the arithmetic mean of article-store demand "
        "over active observed rows in the seven-calendar-day interval centered on that "
        "mapped date. The denominator contains only available active rows, including "
        "active zero-demand rows; inactive and missing dates are excluded. The result "
        "is missing when the target's absolute holiday offset exceeds ten days, no row "
        "exists on the mapped center date, or the centered interval has no active row."
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
        "Total article demand across stores on up to the final eight observed dates "
        "strictly before the origin that match the target weekday and have at least one "
        "active store, divided by total article demand across the final 56 observed "
        "article-date rows before the origin. The numerator's date eligibility uses "
        "activity, but each eligible date's demand sum includes every observed store "
        "row; inactive rows contribute their recorded zero. Missing dates are absent "
        "from both row windows. The value defaults to 1/7 when the denominator is not "
        "positive."
    ),
    "store_category_mean_28": (
        "For each store-category-date, demand is first averaged across active article "
        "rows, including active zeros and excluding inactive articles. The feature is "
        "the arithmetic mean of the available daily category means in the final 28 "
        "observed store-category-date rows strictly before the origin. Dates with no "
        "active article yield null and are excluded from the outer denominator; missing "
        "dates are absent and can make the row window span more than 28 calendar days."
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


def _holiday_calendar(start: object, end: object) -> pd.DataFrame:
    """Build Niedersachsen holiday/event-window features for target dates."""
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    dates = pd.date_range(start_date, end_date, freq="D")
    holiday_map = create_germany_ni_holidays(
        range(start_date.year - 1, end_date.year + 2)
    )
    events = [(pd.Timestamp(day), str(name)) for day, name in holiday_map.items()]
    events_by_year_and_name = {
        (event_date.year, event_name): event_date
        for event_date, event_name in events
    }

    rows: list[dict[str, Any]] = []
    for target in dates:
        nearest_date, nearest_name = min(
            events, key=lambda event: (abs((event[0] - target).days), event[0])
        )
        delta = int((nearest_date - target).days)
        previous_event_date = events_by_year_and_name.get(
            (nearest_date.year - 1, nearest_name)
        )
        previous_event_offset_date = (
            previous_event_date - pd.Timedelta(days=delta)
            if previous_event_date is not None
            else pd.NaT
        )
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
                "is_public_holiday": delta == 0,
                "days_to_nearest_event": delta,
                "holiday_event_window": window,
                "event_name": nearest_name if abs(delta) <= 3 else "none",
                "previous_event_offset_date": (
                    previous_event_offset_date.date()
                    if pd.notna(previous_event_offset_date)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _annual_offset_calendar(start: object, end: object) -> pd.DataFrame:
    """Build the annual-history offset for every possible target date."""
    dates = pd.date_range(
        pd.Timestamp(start).normalize(),
        pd.Timestamp(end).normalize(),
        freq="D",
    )
    return pd.DataFrame(
        {
            "target_period": dates.date,
            "last_year_offset": [get_last_year_offset(day) for day in dates],
        }
    )


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
        "ml_annual_offset_frame",
        _annual_offset_calendar(bounds[0], bounds[1]),
    )
    con.execute(
        "CREATE OR REPLACE TABLE ml_annual_offsets AS "
        "SELECT * FROM ml_annual_offset_frame"
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_annual_reference_dates AS
        SELECT
            offsets.target_period,
            offsets.last_year_offset,
            annual_offsets.annual_offset,
            (
                offsets.target_period
                - annual_offsets.annual_offset * INTERVAL 1 DAY
            )::DATE AS reference_period
        FROM ml_annual_offsets AS offsets
        CROSS JOIN (
            VALUES
                (offsets.last_year_offset - 14),
                (offsets.last_year_offset - 7),
                (offsets.last_year_offset),
                (offsets.last_year_offset + 7),
                (offsets.last_year_offset + 14)
        ) AS annual_offsets(annual_offset)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_series_annual_features AS
        WITH annual_references AS (
            SELECT
                history.ARTIKEL_ID,
                history.MARKT_ID,
                offsets.target_period
            FROM ml_annual_offsets AS offsets
            INNER JOIN ml_target_dates USING (target_period)
            INNER JOIN benchmark_daily_rows AS history
                ON history.period = offsets.target_period
                    - offsets.last_year_offset * INTERVAL 1 DAY
        )
        SELECT
            reference.ARTIKEL_ID,
            reference.MARKT_ID,
            reference.target_period,
            AVG(history.demand) FILTER (WHERE history.is_active)
                AS same_weekday_last_year_mean
        FROM annual_references AS reference
        INNER JOIN ml_annual_reference_dates AS candidate
            ON reference.target_period = candidate.target_period
        LEFT JOIN benchmark_daily_rows AS history
            ON reference.ARTIKEL_ID = history.ARTIKEL_ID
            AND reference.MARKT_ID = history.MARKT_ID
            AND history.period = candidate.reference_period
        GROUP BY
            reference.ARTIKEL_ID,
            reference.MARKT_ID,
            reference.target_period
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_series_weekly_annual_features AS
        WITH requested_reference_weeks AS (
            SELECT DISTINCT
                (
                    DATE_TRUNC('week', offsets.target_period)
                    - offsets.last_year_offset * INTERVAL 1 DAY
                )::DATE AS reference_week_start
            FROM ml_annual_offsets AS offsets
            INNER JOIN ml_target_dates USING (target_period)
        )
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            DATE_TRUNC('week', period)::DATE AS reference_week_start,
            AVG(demand) FILTER (WHERE is_active) AS same_week_last_year_mean
        FROM benchmark_daily_rows
        WHERE DATE_TRUNC('week', period)::DATE
            IN (SELECT reference_week_start FROM requested_reference_weeks)
        GROUP BY ARTIKEL_ID, MARKT_ID, DATE_TRUNC('week', period)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_series_lag_features AS
        -- Single-date RANGE frames return the exact period - 7/14 row when it
        -- exists and NULL otherwise, matching a LEFT JOIN on the exact date.
        -- The window runs over the full history; only rows on requested
        -- target dates are stored because only those are ever joined.
        SELECT * FROM (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                period,
                MAX(demand) OVER lag_7 AS same_weekday_lag_7,
                MAX(demand) OVER lag_14 AS same_weekday_lag_14
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
        CREATE OR REPLACE TABLE ml_series_centered_7_features AS
        -- The centered mean is computed over the full history; only reference
        -- dates reachable from requested target dates are stored because the
        -- event-offset join can only probe those.
        SELECT * FROM (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                period AS reference_period,
                AVG(demand) FILTER (WHERE is_active) OVER (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period
                    RANGE BETWEEN INTERVAL 3 DAY PRECEDING
                        AND INTERVAL 3 DAY FOLLOWING
                ) AS centered_7_demand_mean
            FROM benchmark_daily_rows
        )
        WHERE reference_period IN (
            SELECT event_calendar.previous_event_offset_date
            FROM ml_calendar AS event_calendar
            INNER JOIN ml_target_dates
                ON event_calendar.period = ml_target_dates.target_period
            WHERE event_calendar.previous_event_offset_date IS NOT NULL
                AND ABS(event_calendar.days_to_nearest_event)
                    <= {MAX_EVENT_OFFSET_DAYS}
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
                AVG(demand) OVER trailing_7 AS rolling_7_mean,
                AVG(demand) OVER trailing_28 AS rolling_28_mean,
                AVG((demand > 0)::INTEGER) FILTER (WHERE is_active) OVER trailing_28
                    AS rolling_28_demand_rate,
                COUNT_IF(is_active AND demand > 0) OVER trailing_calendar_7
                    AS demand_days_last_7,
                COUNT_IF(is_active AND demand > 0) OVER trailing_calendar_28
                    AS demand_days_last_28,
                COALESCE(
                    SUM(action_flag) OVER trailing_calendar_28,
                    0
                )::INTEGER AS actions_last_28d,
                COUNT_IF(is_active AND demand > 0) OVER trailing_calendar_60
                    AS demand_days_last_60
            FROM benchmark_daily_rows AS d
            WINDOW
                lifetime AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ),
                trailing_7 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ),
                trailing_28 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
                ),
                trailing_calendar_7 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period
                    RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW
                ),
                trailing_calendar_28 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period
                    RANGE BETWEEN INTERVAL 27 DAY PRECEDING AND CURRENT ROW
                ),
                trailing_calendar_60 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period
                    RANGE BETWEEN INTERVAL 59 DAY PRECEDING AND CURRENT ROW
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
            r.demand_days::DOUBLE / NULLIF(r.active_days, 0) AS demand_day_ratio,
            r.target_mean,
            r.last_positive_period,
            r.last_action_period,
            r.is_active AND r.demand > 0 AS is_positive_sale,
            r.active_days - COALESCE(r.last_positive_active_day, 0)
                AS active_zero_demand_gap,
            r.demand_days_last_7,
            r.demand_days_last_28,
            r.actions_last_28d,
            r.demand_days_last_60,
            r.rolling_7_mean,
            r.rolling_28_mean,
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
            ARTIKEL_ID,
            MARKT_ID,
            period AS feature_date,
            EXTRACT(ISODOW FROM period)::INTEGER AS target_weekday,
            AVG(demand) OVER weekday_4 AS same_weekday_mean_4,
            AVG(demand) OVER weekday_8 AS same_weekday_mean_8
        FROM benchmark_daily_rows
        WHERE is_active
        WINDOW
            weekday_4 AS (
                PARTITION BY ARTIKEL_ID, MARKT_ID, EXTRACT(ISODOW FROM period)
                ORDER BY period ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ),
            weekday_8 AS (
                PARTITION BY ARTIKEL_ID, MARKT_ID, EXTRACT(ISODOW FROM period)
                ORDER BY period ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
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
            ) AS product_cross_store_mean_28,
            SUM(product_demand) OVER (
                PARTITION BY ARTIKEL_ID
                ORDER BY period ROWS BETWEEN 55 PRECEDING AND CURRENT ROW
            ) AS product_demand_56
        FROM ml_product_daily
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE ml_product_annual_features AS
        WITH annual_references AS (
            SELECT
                history.ARTIKEL_ID,
                offsets.target_period
            FROM ml_annual_offsets AS offsets
            INNER JOIN ml_target_dates USING (target_period)
            INNER JOIN ml_product_daily AS history
                ON history.period = offsets.target_period
                    - offsets.last_year_offset * INTERVAL 1 DAY
        )
        SELECT
            reference.ARTIKEL_ID,
            reference.target_period,
            AVG(history.cross_store_mean)
                AS product_cross_store_same_weekday_last_year_mean
        FROM annual_references AS reference
        INNER JOIN ml_annual_reference_dates AS candidate
            ON reference.target_period = candidate.target_period
        LEFT JOIN ml_product_daily AS history
            ON reference.ARTIKEL_ID = history.ARTIKEL_ID
            AND history.period = candidate.reference_period
        GROUP BY reference.ARTIKEL_ID, reference.target_period
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_product_weekday_features AS
        SELECT
            ARTIKEL_ID,
            period AS feature_date,
            EXTRACT(ISODOW FROM period)::INTEGER AS target_weekday,
            SUM(product_demand) OVER (
                PARTITION BY ARTIKEL_ID, EXTRACT(ISODOW FROM period)
                ORDER BY period ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
            ) AS product_weekday_demand_8
        FROM ml_product_daily
        WHERE active_stores > 0
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
        origin_history_with_gaps AS (
            SELECT
                h.*,
                g.completed_gap_count,
                g.historical_gap_p90,
                g.historical_max_gap
            FROM origin_history_base AS h
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
            SELECT h.*, x.product_cross_store_mean_28, x.product_demand_56
            FROM origin_history AS h
            ASOF LEFT JOIN ml_product_features AS x
                ON h.ARTIKEL_ID = x.ARTIKEL_ID
                AND h.origin > x.feature_date
        ),
        origin_history_with_store_category AS (
            SELECT h.*, x.store_category_mean_28
            FROM origin_history_with_product AS h
            ASOF LEFT JOIN ml_store_category_features AS x
                ON h.MARKT_ID = x.MARKT_ID
                AND h.category_id = x.category_id
                AND h.origin > x.feature_date
        )
        SELECT h.*, a.mean_action_lift_in_sourcing_group
        FROM origin_history_with_store_category AS h
        ASOF LEFT JOIN ml_sourcing_group_action_features AS a
            ON h.sourcing_group = a.sourcing_group
            AND h.origin > a.feature_date
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
            x.product_weekday_demand_8
        FROM probe
        ASOF LEFT JOIN ml_product_weekday_features AS x
            ON probe.ARTIKEL_ID = x.ARTIKEL_ID
            AND probe.target_weekday = x.target_weekday
            AND probe.origin > x.feature_date
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
    for resolved_table in (
        "ml_series_features",
        "ml_gap_statistics",
        "ml_series_weekday_features",
        "ml_product_features",
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
            LEFT JOIN ml_series_lag_features AS lags
                ON t.ARTIKEL_ID = lags.ARTIKEL_ID
                AND t.MARKT_ID = lags.MARKT_ID
                AND t.period = lags.period
        ),
        with_annual_history AS (
            SELECT
                t.*,
                (a.target_period IS NOT NULL)::INTEGER AS has_annual_history,
                a.same_weekday_last_year_mean,
                w.same_week_last_year_mean,
                x.product_cross_store_same_weekday_last_year_mean,
                e.centered_7_demand_mean AS same_event_offset_last_year_mean
            FROM targets AS t
            INNER JOIN ml_annual_offsets AS annual_offset
                ON t.period = annual_offset.target_period
            LEFT JOIN ml_series_annual_features AS a
                ON t.ARTIKEL_ID = a.ARTIKEL_ID
                AND t.MARKT_ID = a.MARKT_ID
                AND t.period = a.target_period
            LEFT JOIN ml_series_weekly_annual_features AS w
                ON t.ARTIKEL_ID = w.ARTIKEL_ID
                AND t.MARKT_ID = w.MARKT_ID
                AND DATE_TRUNC('week', t.period)
                    - annual_offset.last_year_offset * INTERVAL 1 DAY
                    = w.reference_week_start
            LEFT JOIN ml_product_annual_features AS x
                ON t.ARTIKEL_ID = x.ARTIKEL_ID
                AND t.period = x.target_period
            LEFT JOIN ml_calendar AS event_calendar
                ON t.period = event_calendar.period
            LEFT JOIN ml_series_centered_7_features AS e
                ON t.ARTIKEL_ID = e.ARTIKEL_ID
                AND t.MARKT_ID = e.MARKT_ID
                AND event_calendar.previous_event_offset_date = e.reference_period
                AND ABS(event_calendar.days_to_nearest_event)
                    <= {MAX_EVENT_OFFSET_DAYS}
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
            SELECT p.*, x.product_weekday_demand_8
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
            p.days_since_last_action,
            p.actions_last_28d,
            p.mean_action_lift_in_sourcing_group,
            p.active_days AS active_days_before_origin,
            p.demand_days AS demand_days_before_origin,
            p.demand_day_ratio,
            p.active_zero_demand_gap,
            p.demand_days_last_7,
            p.demand_days_last_28,
            p.demand_days_last_60,
            p.historical_p90_gap,
            p.current_gap_over_historical_p90_gap,
            p.same_weekday_lag_7,
            p.same_weekday_lag_14,
            p.rolling_7_mean,
            p.rolling_28_mean,
            p.rolling_28_demand_rate,
            p.same_weekday_mean_4,
            p.same_weekday_mean_8,
            p.has_annual_history,
            p.same_weekday_last_year_mean,
            p.same_week_last_year_mean,
            p.product_cross_store_same_weekday_last_year_mean,
            p.same_event_offset_last_year_mean,
            p.ADI,
            p.CV2,
            p.product_cross_store_mean_28,
            CASE
                WHEN p.product_demand_56 > 0
                    THEN p.product_weekday_demand_8 / p.product_demand_56
                ELSE 1.0 / 7.0
            END AS product_weekday_profile_value,
            p.store_category_mean_28,
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
        INNER JOIN ml_calendar AS c USING (period)
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
