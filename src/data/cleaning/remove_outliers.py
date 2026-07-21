"""Remove audited daily data errors and cap extreme Rabatt demand.

The accompanying EDA shows that a global Tukey deletion rule is not suitable
for these intermittent, strongly right-skewed demand series. Consequently,
only the remaining row key identified during manual review is removed when it
is still present after upstream filtering.

Rabatt is an ex-post clearance signal and is unavailable when the forecast is
made. To estimate baseline demand without unplanned clearance spikes, positive
Rabatt observations above the series-specific positive-demand 95th percentile
are capped at that percentile. Ordinary Rabatt observations and all other
statistical outliers remain unchanged.

Input:  data/interim/transactions_dst_daily_min_demand/*.parquet
Output: data/interim/transactions_dst_daily_min_demand_no_outliers/*.parquet
"""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    clear_parquet_outputs,
    configure_duckdb,
    ident,
    read_parquet_expr,
    require_parquet_files,
    sql_literal,
)


IN_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_min_demand"
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_min_demand_no_outliers"

DEMAND_COL = "ABVERKAUFTE_MENGE_KG"
RABATT_CAP_QUANTILE = 0.95

# Exact row keys established by the manual review in
# notebooks/00_data_cleaning/00_04_outliers_detection_eda.ipynb.
KNOWN_ERROR_ROWS = (
    (328555, 1100011, "2023-10-09"),
)


def temp_output_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.no_outliers.tmp{path.suffix}")


def create_daily_view(con, in_glob: Path) -> None:
    """Create the normalized source view required by the cleaning rules."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW daily_rows AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            CAST(DATE AS DATE) AS period,
            CAST(COALESCE({ident(DEMAND_COL)}, 0) AS DOUBLE) AS demand,
            CAST(COALESCE(RABATT, 0) AS INTEGER) AS discount_flag
        FROM {read_parquet_expr(in_glob)}
        """
    )


def create_known_error_table(con) -> None:
    """Materialize reviewed keys and reject duplicate matches in current input."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE known_error_rows (
            ARTIKEL_ID BIGINT,
            MARKT_ID BIGINT,
            period DATE
        )
        """
    )
    con.executemany(
        "INSERT INTO known_error_rows VALUES (?, ?, CAST(? AS DATE))",
        KNOWN_ERROR_ROWS,
    )

    duplicate_matches = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT e.ARTIKEL_ID, e.MARKT_ID, e.period
            FROM known_error_rows e
            JOIN daily_rows d USING (ARTIKEL_ID, MARKT_ID, period)
            GROUP BY ALL
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicate_matches:
        raise RuntimeError(
            "Known-error audit failed: reviewed keys matched duplicate input rows"
        )

    matched_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM daily_rows d
        JOIN known_error_rows e USING (ARTIKEL_ID, MARKT_ID, period)
        """
    ).fetchone()[0]
    print(
        "Reviewed probable-error keys present after upstream filtering: "
        f"{matched_rows:,}/{len(KNOWN_ERROR_ROWS):,}"
    )


def create_outlier_adjustments(con) -> None:
    """Create row-level remove/cap instructions without generic deletion."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE series_fences AS
        SELECT
            ARTIKEL_ID,
            MARKT_ID,
            QUANTILE_CONT(demand, {RABATT_CAP_QUANTILE})
                FILTER (WHERE demand > 0) AS rabatt_cap
        FROM daily_rows
        GROUP BY ARTIKEL_ID, MARKT_ID
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE outlier_adjustments AS
        SELECT
            d.ARTIKEL_ID,
            d.MARKT_ID,
            d.period,
            CASE
                WHEN e.ARTIKEL_ID IS NOT NULL THEN 'remove_probable_error'
                WHEN d.discount_flag = 1
                 AND d.demand > f.rabatt_cap
                    THEN 'cap_rabatt'
            END AS cleaning_action,
            f.rabatt_cap AS cap_value
        FROM daily_rows d
        JOIN series_fences f USING (ARTIKEL_ID, MARKT_ID)
        LEFT JOIN known_error_rows e USING (ARTIKEL_ID, MARKT_ID, period)
        WHERE e.ARTIKEL_ID IS NOT NULL
           OR (d.discount_flag = 1
               AND d.demand > f.rabatt_cap)
        """
    )


def output_select_sql(path: Path) -> str:
    """Return SQL that removes reviewed errors and caps Rabatt spikes."""
    return f"""
        SELECT t.* REPLACE (
            CASE
                WHEN a.cleaning_action = 'cap_rabatt'
                    THEN LEAST(CAST(t.{ident(DEMAND_COL)} AS DOUBLE), a.cap_value)
                ELSE t.{ident(DEMAND_COL)}
            END AS {ident(DEMAND_COL)}
        )
        FROM {read_parquet_expr(path)} t
        LEFT JOIN outlier_adjustments a
          ON t.ARTIKEL_ID = a.ARTIKEL_ID
         AND t.MARKT_ID = a.MARKT_ID
         AND CAST(t.DATE AS DATE) = a.period
        WHERE a.cleaning_action IS NULL
           OR a.cleaning_action <> 'remove_probable_error'
    """


def main(in_dir: Path = IN_DIR, out_dir: Path = OUT_DIR) -> None:
    input_files = require_parquet_files(in_dir)
    in_glob = in_dir / "transactions_year_*.parquet"

    con = configure_duckdb()
    create_daily_view(con, in_glob)
    create_known_error_table(con)
    create_outlier_adjustments(con)

    error_count, rabatt_count = con.execute(
        """
        SELECT
            COUNT_IF(cleaning_action = 'remove_probable_error'),
            COUNT_IF(cleaning_action = 'cap_rabatt')
        FROM outlier_adjustments
        """
    ).fetchone()
    print(f"Audited probable-error rows to remove: {error_count:,}")
    print(f"Extreme Rabatt rows to cap: {rabatt_count:,}")
    print(
        "Rabatt cap: positive-demand "
        f"{100 * RABATT_CAP_QUANTILE:.0f}th percentile"
    )

    clear_parquet_outputs(out_dir)
    n_in_total = 0
    n_out_total = 0
    n_capped_total = 0
    for path in input_files:
        out_path = out_dir / path.name
        tmp_path = temp_output_path(out_path)
        if tmp_path.exists():
            tmp_path.unlink()

        n_in = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(path)}"
        ).fetchone()[0]
        n_out, n_capped = con.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT_IF(a.cleaning_action = 'cap_rabatt')
            FROM {read_parquet_expr(path)} t
            LEFT JOIN outlier_adjustments a
              ON t.ARTIKEL_ID = a.ARTIKEL_ID
             AND t.MARKT_ID = a.MARKT_ID
             AND CAST(t.DATE AS DATE) = a.period
            WHERE a.cleaning_action IS NULL
               OR a.cleaning_action <> 'remove_probable_error'
            """
        ).fetchone()

        con.execute(
            f"""
            COPY ({output_select_sql(path)})
            TO {sql_literal(tmp_path)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        tmp_path.replace(out_path)

        n_in_total += n_in
        n_out_total += n_out
        n_capped_total += n_capped
        print(
            f"Wrote {out_path.name}: {n_out:,}/{n_in:,} rows retained, "
            f"{n_capped:,} Rabatt rows capped"
        )

    if n_in_total - n_out_total != error_count:
        raise RuntimeError(
            "Row-count audit failed: removed "
            f"{n_in_total - n_out_total:,}, expected {error_count:,}"
        )
    if n_capped_total != rabatt_count:
        raise RuntimeError(
            f"Cap-count audit failed: capped {n_capped_total:,}, "
            f"expected {rabatt_count:,}"
        )

    print(f"Total rows retained: {n_out_total:,}/{n_in_total:,}")


if __name__ == "__main__":
    main()
