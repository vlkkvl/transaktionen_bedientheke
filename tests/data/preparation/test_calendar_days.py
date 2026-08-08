from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data.preparation.calendar_days import (
    bridge_day_dates,
    calendar_day_flags,
    school_holiday_dates,
)


class BridgeDayTest(unittest.TestCase):
    def test_friday_after_a_thursday_holiday_bridges(self) -> None:
        # Christi Himmelfahrt 2026 falls on Thursday 14 May in both Bundeslaender.
        for subdivision in ("NI", "NW"):
            bridges = bridge_day_dates(subdivision, [2026])
            self.assertIn(pd.Timestamp("2026-05-15"), bridges)

    def test_monday_before_a_tuesday_holiday_bridges(self) -> None:
        # Tag der Deutschen Einheit 2023 falls on Tuesday 3 October.
        bridges = bridge_day_dates("NI", [2023])
        self.assertIn(pd.Timestamp("2023-10-02"), bridges)

    def test_bridge_days_follow_the_bundesland_calendar(self) -> None:
        # Fronleichnam 2026 is a Thursday holiday in NW only, so only NW bridges
        # the Friday after it.
        self.assertIn(pd.Timestamp("2026-06-05"), bridge_day_dates("NW", [2026]))
        self.assertNotIn(pd.Timestamp("2026-06-05"), bridge_day_dates("NI", [2026]))

    def test_a_public_holiday_is_never_itself_a_bridge_day(self) -> None:
        # Tag der Arbeit 2026 is Friday 1 May; the Thursday before is not a
        # holiday, and the holiday itself must not be flagged.
        bridges = bridge_day_dates("NI", [2026])
        self.assertNotIn(pd.Timestamp("2026-05-01"), bridges)
        for day in bridges:
            self.assertLess(day.dayofweek, 5, f"{day.date()} is not a working day")

    def test_bridge_days_are_found_across_the_year_boundary(self) -> None:
        # Neujahr 2026 is Thursday 1 January, so Friday 2 January bridges even
        # though the holiday driving it belongs to the requested year's start.
        self.assertIn(pd.Timestamp("2026-01-02"), bridge_day_dates("NI", [2026]))

    def test_only_requested_years_are_returned(self) -> None:
        for day in bridge_day_dates("NI", [2025]):
            self.assertEqual(day.year, 2025)


class SchoolHolidayTest(unittest.TestCase):
    def test_summer_holidays_start_on_different_dates_per_bundesland(self) -> None:
        # Niedersachsen starts its 2026 Sommerferien on 2 July, Nordrhein-
        # Westfalen only on 20 July.
        niedersachsen = school_holiday_dates("NI", [2026])
        nordrhein = school_holiday_dates("NW", [2026])
        self.assertIn(pd.Timestamp("2026-07-06"), niedersachsen)
        self.assertNotIn(pd.Timestamp("2026-07-06"), nordrhein)
        self.assertIn(pd.Timestamp("2026-07-21"), nordrhein)

    def test_term_time_is_not_flagged(self) -> None:
        for subdivision in ("NI", "NW"):
            dates = school_holiday_dates(subdivision, [2026])
            self.assertNotIn(pd.Timestamp("2026-06-16"), dates)

    def test_only_requested_years_are_returned(self) -> None:
        for day in school_holiday_dates("NW", [2025]):
            self.assertEqual(day.year, 2025)

    def test_uncovered_years_raise_instead_of_returning_nothing(self) -> None:
        # A silent empty result would produce an all-false feature column, which
        # is exactly the failure mode the Bundesland calendar fix was about.
        with self.assertRaises(ValueError):
            school_holiday_dates("NI", [2035])

    def test_unknown_subdivision_raises(self) -> None:
        with self.assertRaises(ValueError):
            school_holiday_dates("BY", [2026])

    def test_periods_are_read_from_the_committed_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "school_holidays.json"
            path.write_text(
                json.dumps(
                    {
                        "subdivisions": ["NI"],
                        "fully_covered_years": [2026, 2026],
                        "periods": [
                            {
                                "subdivision": "NI",
                                "start": "2026-03-02",
                                "end": "2026-03-04",
                                "name": "Testferien",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dates = school_holiday_dates("NI", [2026], path=path)
        self.assertEqual(
            sorted(dates),
            [
                pd.Timestamp("2026-03-02"),
                pd.Timestamp("2026-03-03"),
                pd.Timestamp("2026-03-04"),
            ],
        )


class CalendarDayFlagsTest(unittest.TestCase):
    def test_flags_cover_every_date_in_the_range(self) -> None:
        flags = calendar_day_flags("2026-05-01", "2026-05-31", "NW")
        self.assertEqual(len(flags), 31)
        self.assertEqual(list(flags.columns), ["period", "is_bridge_day", "is_school_holiday"])
        by_date = flags.set_index("period")
        self.assertTrue(bool(by_date.loc[pd.Timestamp("2026-05-15").date(), "is_bridge_day"]))
        self.assertFalse(bool(by_date.loc[pd.Timestamp("2026-05-14").date(), "is_bridge_day"]))


if __name__ == "__main__":
    unittest.main()
