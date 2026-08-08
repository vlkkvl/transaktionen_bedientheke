"""Brueckentag and school-holiday calendars per Bundesland.

Both calendars are keyed by Bundesland because the store network spans
Niedersachsen and Nordrhein-Westfalen, whose public-holiday and school-holiday
calendars differ. They are *target-date* calendars: whether a date is a bridge
day or a school-holiday day is known years in advance, so neither carries
leakage risk in the way a historical aggregate does.

**Brueckentage** are derived from the public-holiday calendar of the Bundesland:
a working day that sits between a public holiday and the weekend, i.e. a Friday
whose Thursday is a public holiday, or a Monday whose Tuesday is one. The store
is open on such a day; the hypothesis is that customers who bridge the day away
shop differently.

**School holidays** cannot be derived — they are set per Bundesland by decree.
They are read from ``reports/config/school_holidays.json``, a committed snapshot
of the official OpenHolidays API so that feature building needs no network
access. Refresh it with::

    python -m src.data.preparation.calendar_days --refresh
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import ROOT
from src.data.preparation.distribute_sales_over_active_days import (
    HOLIDAY_SUBDIVISIONS,
    create_germany_holidays,
)


SCHOOL_HOLIDAY_PATH = ROOT / "reports" / "config" / "school_holidays.json"
OPEN_HOLIDAYS_URL = "https://openholidaysapi.org/SchoolHolidays"


def _normalized_years(years: Iterable[int]) -> list[int]:
    values = sorted({int(year) for year in years})
    if not values:
        raise ValueError("At least one year is required")
    return values


@lru_cache(maxsize=None)
def _school_holiday_document(path: str) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as file:
        document = json.load(file)
    for key in ("periods", "subdivisions", "fully_covered_years"):
        if key not in document:
            raise ValueError(f"School-holiday reference is missing '{key}': {path}")
    return document


def school_holiday_dates(
    subdivision: str,
    years: Iterable[int],
    path: Path | str = SCHOOL_HOLIDAY_PATH,
) -> set[pd.Timestamp]:
    """Return every school-holiday date of one Bundesland in the given years.

    Raises when the committed reference does not cover a requested year, so a
    missing year fails loudly instead of silently producing an all-false column.
    """
    document = _school_holiday_document(str(path))
    if subdivision not in document["subdivisions"]:
        raise ValueError(
            f"School-holiday reference has no data for subdivision {subdivision!r}; "
            f"it covers {document['subdivisions']}"
        )
    requested = _normalized_years(years)
    first_covered, last_covered = document["fully_covered_years"]
    uncovered = [
        year for year in requested if not first_covered <= year <= last_covered
    ]
    if uncovered:
        raise ValueError(
            f"School-holiday reference covers {first_covered}-{last_covered} but "
            f"years {uncovered} were requested. Refresh "
            f"{Path(path).name} with 'python -m "
            "src.data.preparation.calendar_days --refresh'."
        )
    wanted = set(requested)
    dates: set[pd.Timestamp] = set()
    for entry in document["periods"]:
        if entry["subdivision"] != subdivision:
            continue
        span = pd.date_range(entry["start"], entry["end"], freq="D")
        dates.update(day for day in span if day.year in wanted)
    return dates


def bridge_day_dates(
    subdivision: str,
    years: Iterable[int],
) -> set[pd.Timestamp]:
    """Return the Brueckentage of one Bundesland in the given years.

    A Brueckentag is a Monday-to-Friday working day that is not itself a public
    holiday and that joins a public holiday to the weekend: a Friday whose
    Thursday is a public holiday, or a Monday whose Tuesday is one.
    """
    requested = _normalized_years(years)
    # Neighbouring years are needed because a bridge day in early January can
    # depend on a holiday in the previous year and vice versa.
    span = range(min(requested) - 1, max(requested) + 2)
    holidays = {
        pd.Timestamp(day).normalize()
        for day in create_germany_holidays(subdivision, span)
    }
    wanted = set(requested)
    bridges: set[pd.Timestamp] = set()
    for day in holidays:
        if day.dayofweek == 3:  # Thursday holiday -> the Friday bridges
            candidate = day + pd.Timedelta(days=1)
        elif day.dayofweek == 1:  # Tuesday holiday -> the Monday bridges
            candidate = day - pd.Timedelta(days=1)
        else:
            continue
        if candidate in holidays or candidate.year not in wanted:
            continue
        bridges.add(candidate)
    return bridges


def calendar_day_flags(
    start: object,
    end: object,
    subdivision: str,
    path: Path | str = SCHOOL_HOLIDAY_PATH,
) -> pd.DataFrame:
    """Return per-date bridge-day and school-holiday flags for one Bundesland."""
    dates = pd.date_range(
        pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D"
    )
    if len(dates) == 0:
        raise ValueError("The requested date range is empty")
    years = range(dates.min().year, dates.max().year + 1)
    bridges = bridge_day_dates(subdivision, years)
    school = school_holiday_dates(subdivision, years, path=path)
    return pd.DataFrame(
        {
            "period": dates.date,
            "is_bridge_day": [day in bridges for day in dates],
            "is_school_holiday": [day in school for day in dates],
        }
    )


def refresh_school_holidays(
    subdivisions: Iterable[str] = HOLIDAY_SUBDIVISIONS,
    years: Iterable[int] = range(2021, 2029),
    path: Path | str = SCHOOL_HOLIDAY_PATH,
) -> Path:
    """Re-download the official school-holiday periods and rewrite the snapshot."""
    import urllib.parse
    import urllib.request

    periods: list[dict[str, str]] = []
    for subdivision in subdivisions:
        for year in years:
            query = urllib.parse.urlencode(
                {
                    "countryIsoCode": "DE",
                    "subdivisionCode": f"DE-{subdivision}",
                    "languageIsoCode": "DE",
                    "validFrom": f"{year}-01-01",
                    "validTo": f"{year}-12-31",
                }
            )
            with urllib.request.urlopen(
                f"{OPEN_HOLIDAYS_URL}?{query}", timeout=60
            ) as response:
                payload = json.load(response)
            for entry in payload:
                periods.append(
                    {
                        "subdivision": subdivision,
                        "start": entry["startDate"],
                        "end": entry["endDate"],
                        "name": entry["name"][0]["text"],
                    }
                )
    unique = {
        (entry["subdivision"], entry["start"], entry["end"]): entry
        for entry in periods
    }
    ordered = sorted(
        unique.values(), key=lambda entry: (entry["subdivision"], entry["start"])
    )
    summer_years: dict[int, set[str]] = {}
    for entry in ordered:
        if "sommerferien" in entry["name"].lower():
            summer_years.setdefault(int(entry["start"][:4]), set()).add(
                entry["subdivision"]
            )
    complete = sorted(
        year
        for year, states in summer_years.items()
        if states == set(subdivisions)
    )
    document = {
        "source": f"{OPEN_HOLIDAYS_URL} (countryIsoCode=DE)",
        "source_note": (
            "Official school-holiday periods published by the German Laender via "
            "the OpenHolidays API. Retrieved once and committed so feature "
            "building needs no network access; refresh with "
            "'python -m src.data.preparation.calendar_days --refresh'."
        ),
        "retrieved": pd.Timestamp.today().date().isoformat(),
        "subdivisions": list(subdivisions),
        "fully_covered_years": [complete[0], complete[-1]],
        "periods": ordered,
    }
    target = Path(path)
    target.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _school_holiday_document.cache_clear()
    return target


def main() -> None:
    if "--refresh" in sys.argv[1:]:
        path = refresh_school_holidays()
        print(f"Refreshed {path}")
        return
    for subdivision in HOLIDAY_SUBDIVISIONS:
        years = range(2023, 2027)
        bridges = sorted(bridge_day_dates(subdivision, years))
        school = school_holiday_dates(subdivision, years)
        print(
            f"{subdivision}: {len(bridges)} bridge days, "
            f"{len(school)} school-holiday days in {years.start}-{years.stop - 1}"
        )
        print(f"   {[str(day.date()) for day in bridges]}")


if __name__ == "__main__":
    main()
