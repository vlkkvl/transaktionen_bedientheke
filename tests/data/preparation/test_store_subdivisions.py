from __future__ import annotations

import unittest

import pandas as pd

from src.data.preparation.distribute_sales_over_active_days import (
    build_state_calendar,
    subdivision_for_postal_code,
)


class StoreSubdivisionTest(unittest.TestCase):
    def test_postal_prefixes_map_to_the_right_bundesland(self) -> None:
        for postal_code in (26506, 27472, 28857, 30161, 31157, 49716, 21745):
            self.assertEqual(subdivision_for_postal_code(postal_code), "NI")
        for postal_code in (32312, 33330, 48431, 57392, 59229):
            self.assertEqual(subdivision_for_postal_code(postal_code), "NW")

    def test_border_municipalities_override_their_prefix(self) -> None:
        # Emsland towns carry a 48xxx prefix but sit in Niedersachsen.
        self.assertEqual(subdivision_for_postal_code(48499), "NI")
        self.assertEqual(subdivision_for_postal_code(48488), "NI")
        # Ibbenbueren carries a 49xxx prefix but sits in Nordrhein-Westfalen.
        self.assertEqual(subdivision_for_postal_code(49469), "NW")

    def test_state_calendar_separates_the_diverging_holidays(self) -> None:
        calendar = build_state_calendar(
            pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")
        )
        flags = calendar.set_index(["subdivision", "DATE_D"]).IS_HOLIDAY

        # Fronleichnam: a public holiday in NW only.
        self.assertTrue(flags.loc[("NW", pd.Timestamp("2026-06-04").date())])
        self.assertFalse(flags.loc[("NI", pd.Timestamp("2026-06-04").date())])
        # Reformationstag: a public holiday in NI only.
        self.assertTrue(flags.loc[("NI", pd.Timestamp("2026-10-31").date())])
        self.assertFalse(flags.loc[("NW", pd.Timestamp("2026-10-31").date())])
        # Allerheiligen: NW only.
        self.assertTrue(flags.loc[("NW", pd.Timestamp("2026-11-01").date())])
        self.assertFalse(flags.loc[("NI", pd.Timestamp("2026-11-01").date())])
        # A shared holiday must be flagged for both.
        for subdivision in ("NI", "NW"):
            self.assertTrue(
                flags.loc[(subdivision, pd.Timestamp("2026-12-25").date())]
            )

    def test_every_date_appears_once_per_subdivision(self) -> None:
        calendar = build_state_calendar(
            pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
        )
        self.assertFalse(calendar.duplicated(["subdivision", "DATE_D"]).any())
        self.assertEqual(
            calendar.groupby("subdivision").DATE_D.nunique().nunique(), 1
        )


if __name__ == "__main__":
    unittest.main()


class StateAwareClosureTest(unittest.TestCase):
    """The whole point of the fix: two stores, same date, different closure."""

    def test_fronleichnam_closes_only_the_nw_store(self) -> None:
        import duckdb

        from src.data.preparation.distribute_sales_over_active_days import (
            STATIC_COLS,
            output_select_sql,
        )

        con = duckdb.connect()
        calendar = build_state_calendar(
            pd.Timestamp("2026-06-03"), pd.Timestamp("2026-06-05")
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
                    (1, 10, DATE '2026-06-03', DATE '2026-06-05', 'NI'),
                    (1, 20, DATE '2026-06-03', DATE '2026-06-05', 'NW')
            ) AS v(ARTIKEL_ID, MARKT_ID, START_DATE, END_DATE, subdivision)
            CROSS JOIN (
                SELECT TRUE AS is_fcm, FALSE AS is_pseudo, {static_projection}
            )
            """
        )
        con.execute(
            f"""
            CREATE TEMP TABLE source AS
            SELECT
                1::BIGINT AS ARTIKEL_ID, markt::BIGINT AS MARKT_ID, DATE_D,
                5.0::DOUBLE AS UMS_MENGE,
                5.0::DOUBLE AS ABVERKAUFTE_MENGE_KG,
                5.0::DOUBLE AS UMS_VK_WERT,
                0::TINYINT AS AKTION_KENNZEICHEN, 0::TINYINT AS RABATT,
                0::TINYINT AS ARTIKELRABATT, {static_projection}
            FROM (
                VALUES (10, DATE '2026-06-04'), (20, DATE '2026-06-04')
            ) AS v(markt, DATE_D)
            """
        )
        out = con.execute(output_select_sql(2026)).fetchdf()
        out["DATE"] = pd.to_datetime(out["DATE"])
        fronleichnam = out[out.DATE == pd.Timestamp("2026-06-04")].set_index("MARKT_ID")

        # The Niedersachsen store trades; the Nordrhein-Westfalen store is shut.
        self.assertTrue(bool(fronleichnam.loc[10, "is_active"]))
        self.assertEqual(fronleichnam.loc[10, "ABVERKAUFTE_MENGE_KG"], 5.0)
        self.assertFalse(bool(fronleichnam.loc[20, "is_active"]))
        self.assertEqual(fronleichnam.loc[20, "reason_closed"], "Holiday")
        self.assertEqual(fronleichnam.loc[20, "ABVERKAUFTE_MENGE_KG"], 0.0)
