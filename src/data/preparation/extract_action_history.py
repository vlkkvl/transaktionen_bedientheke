"""Extract per-day promotion context from the raw transaction lines.

The processed daily table keeps only the binary ``AKTION_KENNZEICHEN`` flag;
campaign identity (``AKTIONSNUMMER``), the planned validity window
(``GUELTIG_VON``/``GUELTIG_BIS``) and sale-line unit prices survive only in
``data/raw/transactions``. This module aggregates them once into two small
interim tables that the feature builder reads like the weather cache:

- ``article_store_day_actions.parquet`` — one row per article, store and date
  with at least one action sale line: campaign number, planned validity
  window, and the mean promoted unit price of that day.
- ``article_day_regular_price.parquet`` — one row per article and date with
  the mean non-action unit price across all stores, the reference against
  which discount depth is measured.

Both tables are plain historical records; all leakage discipline (strictly
pre-origin windows) is applied downstream in the feature builder.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from src.models.benchmark.config import ROOT

RAW_TRANSACTIONS_GLOB = ROOT / "data" / "raw" / "transactions" / "*.csv.gz"
ACTIONS_DIR = ROOT / "data" / "interim" / "actions"
ACTION_DAYS_PATH = ACTIONS_DIR / "article_store_day_actions.parquet"
REGULAR_PRICE_PATH = ACTIONS_DIR / "article_day_regular_price.parquet"


def extract_action_history(
    raw_glob: Path | str = RAW_TRANSACTIONS_GLOB,
    actions_path: Path | str = ACTION_DAYS_PATH,
    regular_price_path: Path | str = REGULAR_PRICE_PATH,
) -> None:
    actions_path = Path(actions_path)
    regular_price_path = Path(regular_price_path)
    actions_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    escaped_glob = str(raw_glob).replace("'", "''")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_lines AS
        SELECT
            ARTIKEL_ID::BIGINT AS ARTIKEL_ID,
            MARKT_ID::BIGINT AS MARKT_ID,
            DATE::DATE AS period,
            UMS_VK_PREIS::DOUBLE AS unit_price,
            AKTIONSNUMMER::BIGINT AS aktionsnummer,
            GUELTIG_VON::DATE AS gueltig_von,
            GUELTIG_BIS::DATE AS gueltig_bis,
            (COALESCE(AKTION_KENNZEICHEN, 0) = 1) AS is_action
        FROM read_csv(
            '{escaped_glob}',
            nullstr='NULL',
            header=true,
            auto_detect=true,
            sample_size=20000,
            union_by_name=true
        )
        WHERE (IS_FCM OR IS_PSEUDO) AND WGR_ID IN (890, 900)
        """
    )
    con.execute(
        """
        COPY (
            SELECT
                ARTIKEL_ID,
                MARKT_ID,
                period,
                MAX(aktionsnummer) AS aktionsnummer,
                MIN(gueltig_von) AS gueltig_von,
                MAX(gueltig_bis) AS gueltig_bis,
                AVG(unit_price) FILTER (WHERE unit_price > 0)
                    AS action_unit_price,
                COUNT(*) AS action_lines
            FROM raw_lines
            WHERE is_action
            GROUP BY 1, 2, 3
            ORDER BY ARTIKEL_ID, MARKT_ID, period
        ) TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """.format(actions_path)
    )
    con.execute(
        """
        COPY (
            SELECT
                ARTIKEL_ID,
                period,
                AVG(unit_price) FILTER (WHERE unit_price > 0)
                    AS regular_unit_price,
                COUNT(*) AS regular_lines
            FROM raw_lines
            WHERE NOT is_action
            GROUP BY 1, 2
            HAVING regular_unit_price IS NOT NULL
            ORDER BY ARTIKEL_ID, period
        ) TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """.format(regular_price_path)
    )
    counts = con.execute(
        "SELECT (SELECT COUNT(*) FROM read_parquet(?)) AS action_days, "
        "(SELECT COUNT(*) FROM read_parquet(?)) AS regular_price_days",
        [str(actions_path), str(regular_price_path)],
    ).fetchone()
    print(
        f"wrote {counts[0]:,} action article-store-days to {actions_path} and "
        f"{counts[1]:,} article-day regular prices to {regular_price_path}"
    )


if __name__ == "__main__":
    extract_action_history()
