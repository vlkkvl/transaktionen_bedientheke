from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb

from src.data.cleaning.rules import (
    MARKET_ID_COL,
    closed_market_filter_condition,
    closed_market_ids,
)


class ClosedMarketFilterTest(unittest.TestCase):
    def test_loads_case_insensitive_closed_labels_from_market_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "maerkte.csv"
            path.write_text(
                "MARKT_ID,OEFFNUNGSZEIT_MARKT\n"
                "1,Mo - Sa\n"
                "2,Geschlossen\n"
                "3, dauerhaft GESCHLOSSEN \n",
                encoding="utf-8",
            )

            self.assertEqual(closed_market_ids(path), (2, 3))

    def test_condition_removes_currently_closed_markets(self) -> None:
        closed_id = closed_market_ids()[0]
        condition = closed_market_filter_condition()
        rows = duckdb.connect().execute(
            f"""
            SELECT {MARKET_ID_COL}
            FROM (VALUES ({closed_id}), (9999999), (NULL)) AS t({MARKET_ID_COL})
            WHERE {condition}
            ORDER BY {MARKET_ID_COL} NULLS LAST
            """
        ).fetchall()

        self.assertEqual(rows, [(9999999,), (None,)])

    def test_condition_supports_a_table_alias(self) -> None:
        closed_id = closed_market_ids()[0]
        kept = duckdb.connect().execute(
            f"""
            SELECT {closed_market_filter_condition('t')}
            FROM (SELECT {closed_id} AS {MARKET_ID_COL}) AS t
            """
        ).fetchone()[0]

        self.assertIs(kept, False)


if __name__ == "__main__":
    unittest.main()
