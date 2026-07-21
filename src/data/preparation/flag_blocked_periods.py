"""Flag article/mandant days covered by a supplier delivery block.

The raw article block export is a status history.  A ``BLOCKED`` state starts
an unavailable interval and the following ``FREE`` state ends it.  The start
is inclusive and the end is exclusive.  Availability is reconstructed at the
article/mandant grain: supplier/product-group fields describe the source row
but do not create independent product availability states.  If the final
recorded state is ``FREE``, the product remains available afterwards.

Input:  data/interim/transactions_dst_over_days/*.parquet
Output: data/interim/transactions_dst_over_days_blocked/*.parquet
"""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    clear_parquet_outputs,
    configure_duckdb,
    read_parquet_expr,
    require_parquet_files,
    sql_literal,
)


IN_DIR = ROOT / "data" / "interim" / "transactions_dst_over_days"
OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_over_days_blocked"
BLOCK_FILE = ROOT / "data" / "raw" / "artikel_markt_sperre" / "artikel_markt_sperre.csv"
BLOCK_FLAG = "IS_BLOCKED"

RAW_REQUIRED_COLS = {
    "ARTIKEL_ID",
    "APMAND",
    "AENDERUNGSDATUM",
    "AVAILABILITY_STATUS",
}
TRANSACTION_REQUIRED_COLS = {
    "ARTIKEL_ID",
    "MANDANT_ID",
    "DATE",
}


def create_raw_block_view(con, block_file: Path) -> None:
    """Read and validate the columns used from the block event export."""
    if not block_file.is_file():
        raise FileNotFoundError(f"Block history not found: {block_file}")

    block_literal = sql_literal(block_file)
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto({block_literal}, header=true, "
            "sample_size=-1, dateformat='%d.%m.%Y')"
        ).fetchall()
    }
    missing = sorted(RAW_REQUIRED_COLS - columns)
    if missing:
        raise ValueError(f"Block history is missing columns: {missing}")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_block_history AS
        SELECT
            CAST(ARTIKEL_ID AS BIGINT) AS ARTIKEL_ID,
            CAST(APMAND AS BIGINT) AS MANDANT_ID,
            row_number() OVER () AS SOURCE_ROW,
            CAST(AENDERUNGSDATUM AS DATE) AS EVENT_DATE,
            upper(trim(AVAILABILITY_STATUS)) AS AVAILABILITY_STATUS
        FROM read_csv_auto(
            {block_literal},
            header=true,
            sample_size=-1,
            dateformat='%d.%m.%Y'
        )
        WHERE ARTIKEL_ID IS NOT NULL
          AND APMAND IS NOT NULL
          AND AENDERUNGSDATUM IS NOT NULL
          AND upper(trim(AVAILABILITY_STATUS)) IN ('BLOCKED', 'FREE')
        """
    )


def create_blocked_intervals(con) -> None:
    """Convert chronological article/mandant states into blocked intervals."""
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE blocked_intervals AS
        WITH daily_states AS (
            SELECT
                ARTIKEL_ID,
                MANDANT_ID,
                EVENT_DATE,
                arg_max(AVAILABILITY_STATUS, SOURCE_ROW) AS STATUS
            FROM raw_block_history
            GROUP BY ARTIKEL_ID, MANDANT_ID, EVENT_DATE
        ),
        marked_states AS (
            SELECT
                *,
                lag(STATUS) OVER (
                    PARTITION BY ARTIKEL_ID, MANDANT_ID
                    ORDER BY EVENT_DATE
                ) AS PREVIOUS_STATUS
            FROM daily_states
        ),
        status_events AS (
            SELECT *
            FROM marked_states
            WHERE PREVIOUS_STATUS IS NULL OR STATUS <> PREVIOUS_STATUS
        ),
        dated_events AS (
            SELECT
                *,
                lead(EVENT_DATE) OVER (
                    PARTITION BY ARTIKEL_ID, MANDANT_ID
                    ORDER BY EVENT_DATE
                ) AS NEXT_EVENT_DATE
            FROM status_events
        )
        SELECT
            ARTIKEL_ID,
            MANDANT_ID,
            EVENT_DATE AS BLOCK_START,
            NEXT_EVENT_DATE AS BLOCK_END
        FROM dated_events
        WHERE STATUS = 'BLOCKED'
          AND (NEXT_EVENT_DATE IS NULL OR NEXT_EVENT_DATE > EVENT_DATE)
        """
    )


def validate_transaction_schema(con, path: Path) -> None:
    columns = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM {read_parquet_expr(path)}"
        ).fetchall()
    }
    missing = sorted(TRANSACTION_REQUIRED_COLS - columns)
    if missing:
        raise ValueError(f"Transaction input is missing columns: {missing}")
    if BLOCK_FLAG in columns:
        raise ValueError(f"Transaction input already contains {BLOCK_FLAG}")


def flagged_select_sql(path: Path) -> str:
    """Return transaction rows with the article/mandant block flag appended."""
    return f"""
        SELECT
            t.*,
            EXISTS (
                SELECT 1
                FROM blocked_intervals b
                WHERE b.ARTIKEL_ID = t.ARTIKEL_ID
                  AND b.MANDANT_ID = t.MANDANT_ID
                  AND CAST(t.DATE AS DATE) >= b.BLOCK_START
                  AND (
                      b.BLOCK_END IS NULL
                      OR CAST(t.DATE AS DATE) < b.BLOCK_END
                  )
            ) AS {BLOCK_FLAG}
        FROM {read_parquet_expr(path)} t
    """


def main(
    in_dir: Path = IN_DIR,
    out_dir: Path = OUT_DIR,
    block_file: Path = BLOCK_FILE,
    *,
    threads: int = 8,
) -> None:
    input_files = require_parquet_files(in_dir)
    con = configure_duckdb(threads)
    # SOURCE_ROW resolves multiple records on the same date according to the
    # order supplied by the status export.
    con.execute("SET preserve_insertion_order = true")
    create_raw_block_view(con, block_file)
    create_blocked_intervals(con)

    interval_count = con.execute("SELECT COUNT(*) FROM blocked_intervals").fetchone()[0]
    open_count = con.execute(
        "SELECT COUNT(*) FROM blocked_intervals WHERE BLOCK_END IS NULL"
    ).fetchone()[0]
    print(f"Constructed {interval_count:,} blocked intervals ({open_count:,} still open)")

    clear_parquet_outputs(out_dir)
    total_rows = 0
    total_blocked = 0
    for path in input_files:
        validate_transaction_schema(con, path)
        out_path = out_dir / path.name
        started_at = perf_counter()
        sql = flagged_select_sql(path)
        row_count, blocked_count = con.execute(
            f"SELECT COUNT(*), COUNT_IF({BLOCK_FLAG}) FROM ({sql})"
        ).fetchone()
        con.execute(
            f"COPY ({sql}) TO {sql_literal(out_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        total_rows += row_count
        total_blocked += blocked_count
        print(
            f"Wrote {out_path.name}: {row_count:,} rows, "
            f"{blocked_count:,} blocked ({perf_counter() - started_at:.1f}s)"
        )

    print(f"Flagged {total_blocked:,}/{total_rows:,} transaction days as blocked")


if __name__ == "__main__":
    main()
