from __future__ import annotations

import unittest

import duckdb
import pandas as pd

from src.data.preparation.distribute_sales_over_active_days import (
    STATIC_COLS,
    build_state_calendar,
    output_select_sql,
)


class CompleteCalendarTest(unittest.TestCase):
    def test_output_retains_sundays_and_holidays_with_closure_flags(self) -> None:
        con = duckdb.connect()
        calendar = build_state_calendar(
            pd.Timestamp("2025-12-20"), pd.Timestamp("2025-12-25")
        )
        con.register("calendar_frame", calendar)
        con.execute("CREATE TEMP TABLE calendar AS SELECT * FROM calendar_frame")

        static_projection = ", ".join(
            f"NULL::VARCHAR AS {column}" for column in STATIC_COLS
        )
        con.execute(
            f"""
            CREATE TEMP TABLE series_state AS
            SELECT * FROM (
                VALUES
                    (1, 1, DATE '2025-12-20', DATE '2025-12-25'),
                    (1, 2,
                        DATE '2025-12-20', DATE '2025-12-25')
            ) AS v(ARTIKEL_ID, MARKT_ID, START_DATE, END_DATE)
            CROSS JOIN (
                SELECT TRUE AS is_fcm, FALSE AS is_pseudo, 'NI' AS subdivision,
                       {static_projection}
            )
            """
        )
        con.execute(
            f"""
            CREATE TEMP TABLE source AS
            SELECT
                1::BIGINT AS ARTIKEL_ID,
                1::BIGINT AS MARKT_ID,
                DATE_D,
                demand::DOUBLE AS UMS_MENGE,
                demand::DOUBLE AS ABVERKAUFTE_MENGE_KG,
                demand::DOUBLE AS UMS_VK_WERT,
                0::TINYINT AS AKTION_KENNZEICHEN,
                0::TINYINT AS RABATT,
                0::TINYINT AS ARTIKELRABATT,
                {static_projection}
            FROM (
                VALUES (DATE '2025-12-20', 1.0), (DATE '2025-12-21', 99.0)
            ) AS v(DATE_D, demand)
            """
        )

        output = con.execute(output_select_sql(2025)).fetchdf()
        output["DATE"] = pd.to_datetime(output["DATE"])

        for _, series in output.groupby("MARKT_ID"):
            self.assertEqual(len(series), 6)
            self.assertTrue(series["DATE"].sort_values().diff().dropna().eq("1D").all())

        for _, series in output.groupby("MARKT_ID"):
            by_date = series.set_index("DATE")
            self.assertFalse(bool(by_date.loc["2025-12-21", "is_active"]))
            self.assertEqual(by_date.loc["2025-12-21", "reason_closed"], "Sunday")
            self.assertEqual(
                by_date.loc["2025-12-21", "ABVERKAUFTE_MENGE_KG"], 0.0
            )
            self.assertFalse(bool(by_date.loc["2025-12-25", "is_active"]))
            self.assertEqual(by_date.loc["2025-12-25", "reason_closed"], "Holiday")


if __name__ == "__main__":
    unittest.main()
