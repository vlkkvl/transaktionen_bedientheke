"""Feature engineering for global LightGBM demand forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

from src.data.preparation.distribute_sales_over_active_days import (
    create_germany_ni_holidays,
)
from src.models.benchmark.config import BenchmarkDesign, DEFAULT_DATA_DIR
from src.models.benchmark.evaluation import (
    DEFAULT_DEMAND_COL,
    _create_eligible_origins,
    prepare_daily_rows,
)
from src.models.benchmark.models import create_history_features

TARGET_COLUMN = DEFAULT_DEMAND_COL
NORMALIZED_TARGET_COLUMN = "normalized_actual"
TARGET_SCALE_COLUMN = "target_mean"
DEFAULT_FEATURES_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
DEFAULT_FEATURES_PATH = DEFAULT_FEATURES_DIR / "lightgbm_features.parquet"
ROOT_DIR = Path(__file__).resolve().parents[3]
ABSCHRIFTEN_FEATURES_PATH = ROOT_DIR / "data" / "interim" / "abschriften" / "abschriften_year_*.parquet"
WARENEINGAENGE_FEATURES_PATH = ROOT_DIR / "data" / "interim" / "wareneingaenge" / "wareneingaenge_year_*.parquet"


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
) -> pd.DataFrame:
    """Calculate and persist features for all supplied origins."""
    frame = make_feature_frame(con, origins, design)
    path = _normalize_feature_path(feature_path)
    frame.to_parquet(path)
    return frame


FEATURE_COLUMNS = (
    # IDs
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    # Calendar and forecast position
    "target_weekday",
    "horizon_day",
    "iso_week",
    "month",
    "is_active",
    "reason_closed",
    "is_public_holiday",
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
    # Maturity
    "active_days_before_origin",
    "demand_days_before_origin",
    "demand_day_ratio",
    "days_since_last_positive_sale",
    "maturity_segment",
    # Local demand history
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_28_mean",
    "rolling_28_demand_rate",
    "rolling_28_positive_mean",
    "same_weekday_mean_4",
    "same_weekday_mean_8",
    # Demand regime
    "ADI",
    "CV2",
    "zero_share",
    # Cross-sectional context
    "product_cross_store_mean_28",
    "product_weekday_profile_value",
    "store_category_mean_28",
    # Baseline forecasts as features
    "recent_mean_28_forecast",
    "same_weekday_ma_4_forecast",
    "occurrence_positive_quantity_forecast",
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

CATEGORICAL_FEATURES = (
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    "target_weekday",
    "iso_week",
    "month",
    "is_active",
    "reason_closed",
    "is_public_holiday",
    "holiday_event_window",
    "event_name",
    "maturity_segment",
)

FORECAST_ID_COLUMNS = (
    "ARTIKEL_ID",
    "MARKT_ID",
    "sourcing_group",
    "category_id",
    "origin",
    "period",
    "horizon_day",
    "active_days_before_origin",
    "demand_days_before_origin",
    "recent_occurrence_rate",
    "days_since_last_demand",
    "seasonal_mase_scale",
    "actual",
    "is_active",
    "reason_closed",
)

DIAGNOSTIC_COLUMNS = (
    "action_on_forecast_day",
    "action_during_horizon",
    "days_since_last_action",
    "actions_last_28d",
    "mean_action_lift_in_sourcing_group",
    "maturity_segment",
    "rolling_28_demand_rate",
    "days_since_last_positive_sale",
    "recent_mean_28_forecast",
    "same_weekday_ma_4_forecast",
    "occurrence_positive_quantity_forecast",
)


@dataclass(frozen=True)
class GlobalLightGBMConfig:
    """Training controls shared by global LightGBM architectures.

    ``max_training_origins`` is the initial history size, including the
    validation origins. Later refits use an expanding history and therefore
    intentionally exceed this value.
    """

    max_training_origins: int = 12
    validation_origins: int = 2
    num_boost_round: int = 500
    early_stopping_rounds: int = 40
    random_state: int = 42
    num_threads: int = 4

    def __post_init__(self) -> None:
        positive = {
            "max_training_origins": self.max_training_origins,
            "validation_origins": self.validation_origins,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_threads": self.num_threads,
        }
        invalid = [name for name, value in positive.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
        if self.validation_origins >= self.max_training_origins:
            raise ValueError("validation_origins must be smaller than max_training_origins")


@dataclass
class GlobalLightGBMFrames:
    """Origin-aligned frames used by every global LightGBM variant.

    ``origin_frames`` is populated on the top-level backtest container. Each
    child contains one evaluation origin and the expanding history available
    immediately before that origin.
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
                "is_public_holiday": delta == 0,
                "days_to_nearest_event": delta,
                "holiday_event_window": window,
                "event_name": nearest_name if abs(delta) <= 3 else "none",
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
        CREATE OR REPLACE TEMP TABLE ml_series_features AS
        WITH windowed AS (
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
                demand AS lag_1,
                LAG(demand, 6) OVER series_order AS lag_7,
                LAG(demand, 13) OVER series_order AS lag_14,
                AVG(demand) OVER trailing_7 AS rolling_7_mean,
                AVG(demand) OVER trailing_28 AS rolling_28_mean,
                AVG((demand > 0)::INTEGER) FILTER (WHERE is_active) OVER trailing_28
                    AS rolling_28_demand_rate,
                AVG(demand) FILTER (WHERE is_active AND demand > 0) OVER trailing_28
                    AS rolling_28_positive_mean
            FROM benchmark_daily_rows AS d
            WINDOW
                lifetime AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ),
                series_order AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID ORDER BY period
                ),
                trailing_7 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ),
                trailing_28 AS (
                    PARTITION BY ARTIKEL_ID, MARKT_ID
                    ORDER BY period ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
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
            ARTIKEL_ID,
            MARKT_ID,
            period AS feature_date,
            active_days,
            demand_days,
            demand_days::DOUBLE / NULLIF(active_days, 0) AS demand_day_ratio,
            target_mean,
            last_positive_period,
            last_action_period,
            lag_1,
            lag_7,
            lag_14,
            rolling_7_mean,
            rolling_28_mean,
            rolling_28_demand_rate,
            rolling_28_positive_mean,
            ADI,
            CV2,
            1.0 - demand_days::DOUBLE / NULLIF(active_days, 0) AS zero_share
        FROM regimes
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
        CREATE OR REPLACE TEMP TABLE ml_positive_quantity_features AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            period AS feature_date,
            AVG(demand) OVER (
                PARTITION BY ARTIKEL_ID, MARKT_ID
                ORDER BY period ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS positive_quantity_mean_10
        FROM benchmark_daily_rows
        WHERE is_active AND demand > 0
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
        """
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
        origin_history AS (
            SELECT
                s.*,
                f.* EXCLUDE (ARTIKEL_ID, MARKT_ID),
                DATE_DIFF('day', f.last_positive_period, s.origin)
                    AS days_since_last_positive_sale,
                DATE_DIFF('day', f.last_action_period, s.origin)
                    AS days_since_last_action,
                CASE
                    WHEN f.active_days < ? * 2 OR f.demand_days < ? * 2
                        THEN 'near_threshold'
                    WHEN f.active_days < 365 OR f.demand_days < 50
                        THEN 'developing'
                    ELSE 'established'
                END AS maturity_segment
            FROM origin_receipt_features AS s
            ASOF LEFT JOIN ml_series_features AS f
                ON s.ARTIKEL_ID = f.ARTIKEL_ID
                AND s.MARKT_ID = f.MARKT_ID
                AND s.origin > f.feature_date
        ),
        targets AS (
            SELECT
                h.*,
                t.period,
                t.demand AS actual,
                t.is_active,
                t.reason_closed,
                DATE_DIFF('day', h.origin, t.period) + 1 AS horizon_day,
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
            WHERE h.active_days >= ? AND h.demand_days >= ?
        ),
        with_weekday AS (
            SELECT t.*, w.same_weekday_mean_4, w.same_weekday_mean_8
            FROM targets AS t
            ASOF LEFT JOIN ml_series_weekday_features AS w
                ON t.ARTIKEL_ID = w.ARTIKEL_ID
                AND t.MARKT_ID = w.MARKT_ID
                AND t.target_weekday = w.target_weekday
                AND t.origin > w.feature_date
        ),
        with_positive AS (
            SELECT w.*, p.positive_quantity_mean_10
            FROM with_weekday AS w
            ASOF LEFT JOIN ml_positive_quantity_features AS p
                ON w.ARTIKEL_ID = p.ARTIKEL_ID
                AND w.MARKT_ID = p.MARKT_ID
                AND w.origin > p.feature_date
        ),
        with_product AS (
            SELECT p.*, x.product_cross_store_mean_28, x.product_demand_56
            FROM with_positive AS p
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
            p.horizon_day,
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
            p.days_since_last_positive_sale,
            p.maturity_segment,
            p.lag_1,
            p.lag_7,
            p.lag_14,
            p.rolling_7_mean,
            p.rolling_28_mean,
            p.rolling_28_demand_rate,
            p.rolling_28_positive_mean,
            p.same_weekday_mean_4,
            p.same_weekday_mean_8,
            p.ADI,
            p.CV2,
            p.zero_share,
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
            p.rolling_28_mean AS recent_mean_28_forecast,
            COALESCE(p.same_weekday_mean_4, p.rolling_28_mean)
                AS same_weekday_ma_4_forecast,
            p.rolling_28_demand_rate * COALESCE(p.positive_quantity_mean_10, 0)
                AS occurrence_positive_quantity_forecast,
            p.rolling_28_demand_rate AS recent_occurrence_rate,
            p.days_since_last_positive_sale AS days_since_last_demand,
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
            design.min_active_days,
            design.min_demand_days,
            design.forecast_horizon_days,
            design.min_active_days,
            design.min_demand_days,
        ],
    ).fetchdf()


def historical_training_origins(
    con: duckdb.DuckDBPyConnection,
    design: BenchmarkDesign,
    evaluation_start: object,
    max_origins: int,
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
    while origin >= earliest and len(candidates) < int(max_origins):
        candidates.append(origin)
        origin -= pd.Timedelta(days=design.origin_spacing_days)
    if len(candidates) < int(max_origins):
        raise RuntimeError(
            "Insufficient historical origins to train the global model: "
            f"required {int(max_origins)}, found {len(candidates)}"
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
            required_columns = {
                NORMALIZED_TARGET_COLUMN,
                TARGET_SCALE_COLUMN,
                "is_active",
                "reason_closed",
            }
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
    data_dir: Path = DEFAULT_DATA_DIR,
    config: GlobalLightGBMConfig | None = None,
    feature_dataset_path: Path | str = DEFAULT_FEATURES_PATH,
    force_feature_recompute: bool = False,
) -> GlobalLightGBMFrames:
    """Materialize expanding, leakage-safe frames for every forecast origin.

    The first evaluation origin uses ``max_training_origins`` preceding
    origins. The last ``validation_origins`` are held out for early stopping;
    the remainder are used for fitting. After each forecast origin, its now
    observed target window joins the history, so the fitting set expands by
    one origin while the validation window keeps a constant size.
    """
    config = GlobalLightGBMConfig() if config is None else config
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
    if len(normalized_origins) > 1:
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
        _create_eligible_origins(con, normalized_origins, design)

    con.execute("DROP TABLE IF EXISTS benchmark_row_features")
    con.execute("DROP TABLE IF EXISTS benchmark_weekday_features")
    con.execute("DROP TABLE IF EXISTS benchmark_positive_features")
    initial_history_origins = historical_training_origins(
        con,
        design,
        normalized_origins.min(),
        config.max_training_origins,
    )
    create_feature_tables(con)
    all_history_origins = initial_history_origins.append(normalized_origins[:-1])
    all_history_origins = all_history_origins.unique().sort_values()
    all_frame_origins = all_history_origins.append(normalized_origins)
    all_frame_origins = all_frame_origins.unique().sort_values()
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
    for position, evaluation_origin in enumerate(normalized_origins):
        available_origins = initial_history_origins.append(normalized_origins[:position])
        available_origins = available_origins.unique().sort_values()
        validation_origins = available_origins[-config.validation_origins :]
        fitting_origins = available_origins[: -config.validation_origins]

        training = history_frame.loc[
            history_origin_values.isin(fitting_origins) & history_frame["is_active"]
        ].copy()
        validation = history_frame.loc[
            history_origin_values.isin(validation_origins) & history_frame["is_active"]
        ].copy()
        evaluation = evaluation_frame.loc[
            evaluation_origin_values.eq(evaluation_origin)
        ].copy()
        if training.empty or validation.empty or evaluation.empty:
            raise RuntimeError(
                "Training, validation, and evaluation samples must be nonempty "
                f"for origin {evaluation_origin.date()}"
            )
        origin_frames.append(
            GlobalLightGBMFrames(
                training=training,
                validation=validation,
                evaluation=evaluation,
                training_origins=available_origins,
                evaluation_origin=pd.Timestamp(evaluation_origin),
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
