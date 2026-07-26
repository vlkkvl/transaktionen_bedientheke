"""Feature definitions and engineering shared by all LightGBM models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
ABSCHRIFTEN_FEATURES_PATH = (
    ROOT / "data" / "interim" / "abschriften" / "abschriften_year_*.parquet"
)
WARENEINGAENGE_FEATURES_PATH = (
    ROOT / "data" / "interim" / "wareneingaenge" / "wareneingaenge_year_*.parquet"
)
MIN_COMPLETED_GAPS_FOR_P90 = 10
FEATURE_ORIGIN_BATCH_SIZE = 4


def _normalize_feature_path(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _parquet_expr_if_available(path: Path) -> str:
    files = sorted(Path(path).parent.glob(Path(path).name))
    if not files:
        return "''"
    return str(path)


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
    "lag_364",
    "lag_371",
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
    # Spoilage
    "spoilage_qty_last_28d",
    "spoilage_days_last_28d",
    "days_since_last_spoilage",
    "has_any_spoilage_history",
    # Goods receipts
    "receipt_qty_pos_last_7d",
    "receipt_qty_pos_last_14d",
    "receipt_qty_pos_last_28d",
    "receipt_days_last_28d",
    "days_since_last_receipt",
    "receipt_qty_net_last_28d",
    "receipt_negative_qty_last_28d",
    "has_receipt_history",
)

FEATURE_DESCRIPTIONS = {
    "ARTIKEL_ID": "Article identifier copied from the target series and encoded categorically.",
    "MARKT_ID": "Store identifier copied from the target series and encoded categorically.",
    "sourcing_group": (
        "Series-level sourcing group derived during daily-row preparation, such as "
        "FCM or Pseudo, and encoded categorically."
    ),
    "category_id": (
        "Product-category identifier copied from the target series and encoded "
        "categorically."
    ),
    "target_weekday": (
        "ISO weekday extracted from the forecast target date, with Monday equal to 1 "
        "and Sunday equal to 7."
    ),
    "iso_week": "ISO calendar-week number extracted from the forecast target date.",
    "month": "Calendar-month number extracted from the forecast target date.",
    "days_to_nearest_event": (
        "Signed calendar-day difference from the target date to the nearest known "
        "Niedersachsen public holiday; positive values are before the holiday."
    ),
    "holiday_event_window": (
        "Categorical holiday position: holiday, one-to-three days before, one-to-three "
        "days after, or none."
    ),
    "event_name": (
        "Name of the nearest Niedersachsen public holiday when it is within three "
        "calendar days of the target date; otherwise none."
    ),
    "action_on_forecast_day": (
        "Promotion flag from the known action schedule on the forecast target date."
    ),
    "action_during_horizon": (
        "Maximum promotion flag across all seven target dates for the series and origin."
    ),
    "days_since_last_action": (
        "Calendar days from the most recent promotion strictly before the origin to "
        "the origin."
    ),
    "actions_last_28d": (
        "Sum of daily promotion flags from origin minus 28 calendar days through the "
        "day before origin."
    ),
    "mean_action_lift_in_sourcing_group": (
        "For the sourcing group, cumulative mean active-day demand on promotion days "
        "divided by cumulative mean active-day demand on non-promotion days, minus one; "
        "only dates strictly before the origin are used."
    ),
    "active_days_before_origin": (
        "Cumulative count of active calendar days for the article-store series strictly "
        "before the origin."
    ),
    "demand_days_before_origin": (
        "Cumulative count of active days with demand greater than zero for the series "
        "strictly before the origin."
    ),
    "demand_day_ratio": (
        "demand_days_before_origin divided by active_days_before_origin."
    ),
    "active_zero_demand_gap": (
        "Number of active days after the series' most recent positive-demand day and "
        "before the origin; inactive calendar days do not increase the gap."
    ),
    "demand_days_last_7": (
        "Count of active positive-demand days from origin minus seven calendar days "
        "through the day before origin."
    ),
    "demand_days_last_28": (
        "Count of active positive-demand days from origin minus 28 calendar days "
        "through the day before origin."
    ),
    "demand_days_last_60": (
        "Count of active positive-demand days from origin minus 60 calendar days "
        "through the day before origin."
    ),
    "historical_p90_gap": (
        "90th percentile of the series' completed active-day zero-demand gaps ending "
        "strictly before the origin; until ten completed gaps exist, the historical "
        "maximum gap is used instead, or the current gap when no completed gap exists."
    ),
    "current_gap_over_historical_p90_gap": (
        "active_zero_demand_gap divided by historical_p90_gap; set to zero when the "
        "current gap is zero and left missing when no nonzero denominator exists."
    ),
    "same_weekday_lag_7": (
        "Series demand on the exact calendar date seven days before the forecast target "
        "date."
    ),
    "same_weekday_lag_14": (
        "Series demand on the exact calendar date 14 days before the forecast target "
        "date."
    ),
    "rolling_7_mean": (
        "Arithmetic mean of series demand over the final seven calendar rows strictly "
        "before the origin, including inactive days recorded with zero demand."
    ),
    "rolling_28_mean": (
        "Arithmetic mean of series demand over the final 28 calendar rows strictly "
        "before the origin, including inactive days recorded with zero demand."
    ),
    "rolling_28_demand_rate": (
        "Share of active days with positive demand among the final 28 calendar rows "
        "strictly before the origin."
    ),
    "same_weekday_mean_4": (
        "Mean demand over the final four active observations matching the target ISO "
        "weekday and occurring strictly before the origin."
    ),
    "same_weekday_mean_8": (
        "Mean demand over the final eight active observations matching the target ISO "
        "weekday and occurring strictly before the origin."
    ),
    "has_annual_history": (
        "One when the series has an observed calendar row exactly 364 days before the "
        "target date, otherwise zero."
    ),
    "lag_364": (
        "Series demand on the exact calendar date 364 days, or 52 full weeks, before "
        "the target date."
    ),
    "lag_371": (
        "Series demand on the exact calendar date 371 days, or 53 full weeks, before "
        "the target date."
    ),
    "same_weekday_last_year_mean": (
        "Mean available series demand on the five matching weekdays at target minus "
        "378, 371, 364, 357, and 350 calendar days."
    ),
    "same_week_last_year_mean": (
        "Mean series demand over the Monday-to-Sunday calendar week whose Monday is "
        "364 days before the Monday of the target week."
    ),
    "product_cross_store_same_weekday_last_year_mean": (
        "For each of the five matching weekdays at target minus 378, 371, 364, 357, "
        "and 350 days, mean article demand is first calculated across active stores; "
        "the feature is the mean of those five cross-store values."
    ),
    "same_event_offset_last_year_mean": (
        "The nearest target-date holiday and signed day offset are mapped to the same "
        "holiday in the previous year; the feature is mean series demand from three "
        "calendar days before through three days after that mapped date."
    ),
    "ADI": (
        "Cumulative active days divided by cumulative positive-demand days for the "
        "series, using all observations strictly before the origin."
    ),
    "CV2": (
        "Population variance of positive demand quantities divided by their squared "
        "mean, using all active positive-demand observations strictly before origin."
    ),
    "product_cross_store_mean_28": (
        "For each of the final 28 calendar dates before origin, article demand is "
        "averaged across active stores; the feature is the mean of those daily values."
    ),
    "product_weekday_profile_value": (
        "Article demand summed across stores over the final eight active occurrences of "
        "the target weekday, divided by total article demand across the final 56 calendar "
        "dates before origin; 1/7 is used when the denominator is zero."
    ),
    "store_category_mean_28": (
        "For each of the final 28 calendar dates before origin, demand is averaged across "
        "active products in the same store and category; the feature is the mean of "
        "those daily category values."
    ),
    "spoilage_qty_last_28d": (
        "Sum of Q-type spoilage quantity for the series from origin minus 28 calendar "
        "days through the day before origin; zero when no records exist."
    ),
    "spoilage_days_last_28d": (
        "Count of distinct dates with Q-type spoilage records for the series during the "
        "28 calendar days before origin."
    ),
    "days_since_last_spoilage": (
        "Calendar days from the most recent Q-type spoilage record strictly before the "
        "origin to the origin."
    ),
    "has_any_spoilage_history": (
        "One when at least one Q-type spoilage record exists for the series strictly "
        "before the origin, otherwise zero."
    ),
    "receipt_qty_pos_last_7d": (
        "Sum of positive goods-receipt quantities for the series during the seven "
        "calendar days before origin."
    ),
    "receipt_qty_pos_last_14d": (
        "Sum of positive goods-receipt quantities for the series during the 14 calendar "
        "days before origin."
    ),
    "receipt_qty_pos_last_28d": (
        "Sum of positive goods-receipt quantities for the series during the 28 calendar "
        "days before origin."
    ),
    "receipt_days_last_28d": (
        "Count of distinct dates with any goods-receipt record for the series during the "
        "28 calendar days before origin."
    ),
    "days_since_last_receipt": (
        "Calendar days from the most recent goods-receipt record strictly before the "
        "origin to the origin."
    ),
    "receipt_qty_net_last_28d": (
        "Signed sum of all goods-receipt quantities for the series during the 28 "
        "calendar days before origin."
    ),
    "receipt_negative_qty_last_28d": (
        "Sum of the absolute quantities of negative goods-receipt records for the "
        "series during the 28 calendar days before origin."
    ),
    "has_receipt_history": (
        "One when at least one goods-receipt record exists for the series strictly "
        "before the origin, otherwise zero."
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
        if not _required_materialized_columns().issubset(columns):
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


def create_feature_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create reusable pre-origin feature tables from ``benchmark_daily_rows``."""
    bounds = con.execute(
        "SELECT MIN(period), MAX(period) FROM benchmark_daily_rows"
    ).fetchone()
    if bounds is None or bounds[0] is None:
        raise RuntimeError("benchmark_daily_rows is empty")
    con.register("ml_calendar_frame", _holiday_calendar(bounds[0], bounds[1]))
    con.execute(
        "CREATE OR REPLACE TEMP TABLE ml_calendar AS SELECT * FROM ml_calendar_frame"
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_series_annual_features AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            (period + INTERVAL 364 DAY)::DATE AS target_period,
            demand AS lag_364,
            LAG(demand) OVER same_weekday_history AS lag_371,
            AVG(demand) OVER same_weekday_window
                AS same_weekday_last_year_mean
        FROM benchmark_daily_rows
        WINDOW
            same_weekday_history AS (
                PARTITION BY ARTIKEL_ID, MARKT_ID, EXTRACT(ISODOW FROM period)
                ORDER BY period
            ),
            same_weekday_window AS (
                PARTITION BY ARTIKEL_ID, MARKT_ID, EXTRACT(ISODOW FROM period)
                ORDER BY period ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
            )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_series_weekly_annual_features AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            (DATE_TRUNC('week', period) + INTERVAL 364 DAY)::DATE
                AS target_week_start,
            AVG(demand) AS same_week_last_year_mean
        FROM benchmark_daily_rows
        GROUP BY ARTIKEL_ID, MARKT_ID, DATE_TRUNC('week', period)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ml_series_centered_7_features AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            period AS reference_period,
            AVG(demand) OVER (
                PARTITION BY ARTIKEL_ID, MARKT_ID
                ORDER BY period
                RANGE BETWEEN INTERVAL 3 DAY PRECEDING
                    AND INTERVAL 3 DAY FOLLOWING
            ) AS centered_7_demand_mean
        FROM benchmark_daily_rows
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
                AVG(demand) FILTER (WHERE is_active) OVER lifetime AS target_mean,
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
        CREATE OR REPLACE TEMP TABLE ml_product_annual_features AS
        SELECT
            ARTIKEL_ID,
            (period + INTERVAL 364 DAY)::DATE AS target_period,
            AVG(cross_store_mean) OVER (
                PARTITION BY ARTIKEL_ID, EXTRACT(ISODOW FROM period)
                ORDER BY period ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
            ) AS product_cross_store_same_weekday_last_year_mean
        FROM ml_product_daily
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

    abs_path = _parquet_expr_if_available(ABSCHRIFTEN_FEATURES_PATH)
    if abs_path == "''":
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE ml_spoilage_q_features AS
            SELECT
                CAST(NULL AS BIGINT) AS ARTIKEL_ID,
                CAST(NULL AS BIGINT) AS MARKT_ID,
                CAST(NULL AS DATE) AS period,
                CAST(NULL AS DOUBLE) AS spoilage_qty
            WHERE FALSE
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE ml_spoilage_q_features AS
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                CAST(DATE AS DATE) AS period,
                CAST(IST_ABSCHRIFTEN_MENGE AS DOUBLE) AS spoilage_qty
            FROM read_parquet('{abs_path}')
            WHERE ABSCHRIFT_ART = 'Q'
            """
        )

    receipt_path = _parquet_expr_if_available(WARENEINGAENGE_FEATURES_PATH)
    if receipt_path == "''":
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE ml_receipt_features AS
            SELECT
                CAST(NULL AS BIGINT) AS ARTIKEL_ID,
                CAST(NULL AS BIGINT) AS MARKT_ID,
                CAST(NULL AS DATE) AS period,
                CAST(NULL AS DOUBLE) AS we_menge_vke
            WHERE FALSE
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE ml_receipt_features AS
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                CAST(DATE AS DATE) AS period,
                CAST(WE_MENGE_VKE AS DOUBLE) AS we_menge_vke
            FROM read_parquet('{receipt_path}')
            """
        )


def _normalized_origins(origins: Iterable[object]) -> pd.DataFrame:
    values = pd.DatetimeIndex(origins).normalize().unique().sort_values()
    if len(values) == 0:
        raise ValueError("At least one origin is required")
    return pd.DataFrame({"origin": values.date})


def make_feature_frame(
    con: duckdb.DuckDBPyConnection,
    origins: Iterable[object],
    design: BenchmarkDesign,
) -> pd.DataFrame:
    """Return direct-horizon feature rows for mature series at supplied origins."""
    con.register("ml_requested_origin_frame", _normalized_origins(origins))
    con.execute(
        "CREATE OR REPLACE TEMP TABLE ml_requested_origins AS "
        "SELECT * FROM ml_requested_origin_frame"
    )
    return con.execute(
        f"""
        WITH series AS (
            SELECT DISTINCT
                ARTIKEL_ID, MARKT_ID, sourcing_group, category_id
            FROM benchmark_daily_rows
        ),
        origin_series AS (
            SELECT s.*, o.origin
            FROM series AS s
            CROSS JOIN ml_requested_origins AS o
        ),
    origin_action_history AS (
        SELECT
            os.*,
            COALESCE(a.actions_last_28d, 0)::INTEGER AS actions_last_28d
        FROM origin_series AS os
        LEFT JOIN LATERAL (
            SELECT SUM(history.action_flag) AS actions_last_28d
            FROM benchmark_daily_rows AS history
            WHERE os.ARTIKEL_ID = history.ARTIKEL_ID
                AND os.MARKT_ID = history.MARKT_ID
                AND history.period >= os.origin - INTERVAL 28 DAY
                AND history.period < os.origin
        ) AS a ON TRUE
    ),
    origin_spoilage_features AS (
        SELECT
            oah.*,
            sf.spoilage_qty_last_28d,
            sf.spoilage_days_last_28d,
            sf.days_since_last_spoilage,
            sf.has_any_spoilage_history
        FROM origin_action_history AS oah
        LEFT JOIN LATERAL (
            SELECT
                COALESCE(SUM(history.spoilage_qty) FILTER (
                    WHERE history.period >= oah.origin - INTERVAL 28 DAY
                ), 0) AS spoilage_qty_last_28d,
                COUNT(DISTINCT history.period) FILTER (
                    WHERE history.period >= oah.origin - INTERVAL 28 DAY
                ) AS spoilage_days_last_28d,
                MIN(DATE_DIFF('day', history.period, oah.origin))
                    AS days_since_last_spoilage,
                CASE WHEN COUNT(history.period) > 0 THEN 1 ELSE 0 END
                    AS has_any_spoilage_history
            FROM ml_spoilage_q_features AS history
            WHERE oah.ARTIKEL_ID = history.ARTIKEL_ID
                AND oah.MARKT_ID = history.MARKT_ID
                AND history.period < oah.origin
        ) AS sf ON TRUE
    ),
    origin_receipt_features AS (
        SELECT
            osf.*,
            rf.receipt_qty_pos_last_7d,
            rf.receipt_qty_pos_last_14d,
            rf.receipt_qty_pos_last_28d,
            rf.receipt_days_last_28d,
            rf.days_since_last_receipt,
            rf.receipt_qty_net_last_28d,
            rf.receipt_negative_qty_last_28d,
            rf.has_receipt_history
        FROM origin_spoilage_features AS osf
        LEFT JOIN LATERAL (
            SELECT
                COALESCE(SUM(history.we_menge_vke) FILTER (
                    WHERE history.period >= osf.origin - INTERVAL 7 DAY
                        AND history.we_menge_vke > 0
                ), 0) AS receipt_qty_pos_last_7d,
                COALESCE(SUM(history.we_menge_vke) FILTER (
                    WHERE history.period >= osf.origin - INTERVAL 14 DAY
                        AND history.we_menge_vke > 0
                ), 0) AS receipt_qty_pos_last_14d,
                COALESCE(SUM(history.we_menge_vke) FILTER (
                    WHERE history.period >= osf.origin - INTERVAL 28 DAY
                        AND history.we_menge_vke > 0
                ), 0) AS receipt_qty_pos_last_28d,
                COUNT(DISTINCT history.period) FILTER (
                    WHERE history.period >= osf.origin - INTERVAL 28 DAY
                ) AS receipt_days_last_28d,
                MIN(DATE_DIFF('day', history.period, osf.origin))
                    AS days_since_last_receipt,
                COALESCE(SUM(history.we_menge_vke) FILTER (
                    WHERE history.period >= osf.origin - INTERVAL 28 DAY
                ), 0) AS receipt_qty_net_last_28d,
                COALESCE(SUM(ABS(history.we_menge_vke)) FILTER (
                    WHERE history.period >= osf.origin - INTERVAL 28 DAY
                        AND history.we_menge_vke < 0
                ), 0) AS receipt_negative_qty_last_28d,
                CASE WHEN COUNT(history.period) > 0 THEN 1 ELSE 0 END
                    AS has_receipt_history
            FROM ml_receipt_features AS history
            WHERE osf.ARTIKEL_ID = history.ARTIKEL_ID
                AND osf.MARKT_ID = history.MARKT_ID
                AND history.period < osf.origin
        ) AS rf ON TRUE
    ),
        origin_history_base AS (
            SELECT
                s.*,
                f.* EXCLUDE (ARTIKEL_ID, MARKT_ID),
                DATE_DIFF('day', f.last_positive_period, s.origin)
                    AS calendar_days_since_last_demand,
                DATE_DIFF('day', f.last_action_period, s.origin)
                    AS days_since_last_action
            FROM origin_receipt_features AS s
            ASOF LEFT JOIN ml_series_features AS f
                ON s.ARTIKEL_ID = f.ARTIKEL_ID
                AND s.MARKT_ID = f.MARKT_ID
                AND s.origin > f.feature_date
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
                t.action_flag::INTEGER AS action_on_forecast_day,
                MAX(t.action_flag) OVER (
                    PARTITION BY h.ARTIKEL_ID, h.MARKT_ID, h.origin
                )::INTEGER AS action_during_horizon
            FROM origin_history AS h
            INNER JOIN benchmark_daily_rows AS t
                ON h.ARTIKEL_ID = t.ARTIKEL_ID
                AND h.MARKT_ID = t.MARKT_ID
                AND t.period >= h.origin
                AND t.period < h.origin + ? * INTERVAL 1 DAY
            WHERE h.active_days >= ?
        ),
        with_calendar_history AS (
            SELECT
                t.*,
                previous_week.demand AS same_weekday_lag_7,
                two_weeks_prior.demand AS same_weekday_lag_14
            FROM targets AS t
            LEFT JOIN benchmark_daily_rows AS previous_week
                ON t.ARTIKEL_ID = previous_week.ARTIKEL_ID
                AND t.MARKT_ID = previous_week.MARKT_ID
                AND previous_week.period = t.period - INTERVAL 7 DAY
            LEFT JOIN benchmark_daily_rows AS two_weeks_prior
                ON t.ARTIKEL_ID = two_weeks_prior.ARTIKEL_ID
                AND t.MARKT_ID = two_weeks_prior.MARKT_ID
                AND two_weeks_prior.period = t.period - INTERVAL 14 DAY
        ),
        with_annual_history AS (
            SELECT
                t.*,
                (a.target_period IS NOT NULL)::INTEGER AS has_annual_history,
                a.lag_364,
                a.lag_371,
                a.same_weekday_last_year_mean,
                w.same_week_last_year_mean,
                x.product_cross_store_same_weekday_last_year_mean,
                e.centered_7_demand_mean AS same_event_offset_last_year_mean
            FROM with_calendar_history AS t
            LEFT JOIN ml_series_annual_features AS a
                ON t.ARTIKEL_ID = a.ARTIKEL_ID
                AND t.MARKT_ID = a.MARKT_ID
                AND t.period = a.target_period
            LEFT JOIN ml_series_weekly_annual_features AS w
                ON t.ARTIKEL_ID = w.ARTIKEL_ID
                AND t.MARKT_ID = w.MARKT_ID
                AND DATE_TRUNC('week', t.period) = w.target_week_start
            LEFT JOIN ml_product_annual_features AS x
                ON t.ARTIKEL_ID = x.ARTIKEL_ID
                AND t.period = x.target_period
            LEFT JOIN ml_calendar AS event_calendar
                ON t.period = event_calendar.period
            LEFT JOIN ml_series_centered_7_features AS e
                ON t.ARTIKEL_ID = e.ARTIKEL_ID
                AND t.MARKT_ID = e.MARKT_ID
                AND event_calendar.previous_event_offset_date = e.reference_period
        ),
        with_weekday AS (
            SELECT t.*, w.same_weekday_mean_4, w.same_weekday_mean_8
            FROM with_annual_history AS t
            ASOF LEFT JOIN ml_series_weekday_features AS w
                ON t.ARTIKEL_ID = w.ARTIKEL_ID
                AND t.MARKT_ID = w.MARKT_ID
                AND t.target_weekday = w.target_weekday
                AND t.origin > w.feature_date
        ),
        with_product AS (
            SELECT p.*, x.product_cross_store_mean_28, x.product_demand_56
            FROM with_weekday AS p
            ASOF LEFT JOIN ml_product_features AS x
                ON p.ARTIKEL_ID = x.ARTIKEL_ID
                AND p.origin > x.feature_date
        ),
        with_product_weekday AS (
            SELECT p.*, x.product_weekday_demand_8
            FROM with_product AS p
            ASOF LEFT JOIN ml_product_weekday_features AS x
                ON p.ARTIKEL_ID = x.ARTIKEL_ID
                AND p.target_weekday = x.target_weekday
                AND p.origin > x.feature_date
        ),
        with_store_category AS (
            SELECT p.*, x.store_category_mean_28
            FROM with_product_weekday AS p
            ASOF LEFT JOIN ml_store_category_features AS x
                ON p.MARKT_ID = x.MARKT_ID
                AND p.category_id = x.category_id
                AND p.origin > x.feature_date
        ),
        with_action_lift AS (
            SELECT p.*, a.mean_action_lift_in_sourcing_group
            FROM with_store_category AS p
            ASOF LEFT JOIN ml_sourcing_group_action_features AS a
                ON p.sourcing_group = a.sourcing_group
                AND p.origin > a.feature_date
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
            p.lag_364,
            p.lag_371,
            p.same_weekday_last_year_mean,
            p.same_week_last_year_mean,
            p.product_cross_store_same_weekday_last_year_mean,
            p.same_event_offset_last_year_mean,
            p.ADI,
            p.CV2,
            p.product_cross_store_mean_28,
            COALESCE(p.spoilage_qty_last_28d, 0) AS spoilage_qty_last_28d,
            COALESCE(p.spoilage_days_last_28d, 0) AS spoilage_days_last_28d,
            p.days_since_last_spoilage,
            p.has_any_spoilage_history,
            COALESCE(p.receipt_qty_pos_last_7d, 0) AS receipt_qty_pos_last_7d,
            COALESCE(p.receipt_qty_pos_last_14d, 0) AS receipt_qty_pos_last_14d,
            COALESCE(p.receipt_qty_pos_last_28d, 0) AS receipt_qty_pos_last_28d,
            COALESCE(p.receipt_days_last_28d, 0) AS receipt_days_last_28d,
            p.days_since_last_receipt,
            COALESCE(p.receipt_qty_net_last_28d, 0) AS receipt_qty_net_last_28d,
            COALESCE(p.receipt_negative_qty_last_28d, 0)
                AS receipt_negative_qty_last_28d,
            p.has_receipt_history,
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
        FROM with_action_lift AS p
        INNER JOIN ml_calendar AS c USING (period)
        LEFT JOIN benchmark_origin_history AS h
            ON p.ARTIKEL_ID = h.ARTIKEL_ID
            AND p.MARKT_ID = h.MARKT_ID
            AND p.origin = h.origin
        ORDER BY p.origin, p.ARTIKEL_ID, p.MARKT_ID, p.period
        """,
        [
            design.forecast_horizon_days,
            design.min_active_days,
        ],
    ).fetchdf()


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
    con.execute(f"PRAGMA threads={int(config.num_threads)}")
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
        create_feature_tables(con)
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
