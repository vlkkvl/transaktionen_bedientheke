from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb

from src.data.cleaning.remove_stale_series import main


class RemoveStaleSeriesTest(unittest.TestCase):
    def test_removes_stale_sparse_and_low_demand_day_series(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            in_dir = root / "input"
            out_dir = root / "output"
            in_dir.mkdir()
            input_path = in_dir / "transactions_year_2026.parquet"

            con = duckdb.connect()
            value_rows: list[tuple[int, int, str, float, bool]] = []

            def add_row(
                artikel_id: int,
                date: str,
                demand: float,
                *,
                markt_id: int = 10,
                is_active: bool = True,
            ) -> None:
                value_rows.append((artikel_id, markt_id, date, demand, is_active))

            # Retained: 15 total demand days; last sale exactly 365 days before max date.
            for day in range(17, 32):
                add_row(1, f"2024-12-{day:02d}", 1.0)
            add_row(1, "2025-01-01", 1.0)
            add_row(1, "2026-01-01", 0.0)

            # Removed: last sale 366 days before max date.
            for day in range(16, 32):
                add_row(2, f"2024-12-{day:02d}", 1.0)
            add_row(2, "2026-01-01", 0.0)

            # Removed: no positive sale.
            add_row(3, "2025-12-31", 0.0)
            add_row(3, "2026-01-01", 0.0)

            # Retained: recent, dense, and at least 15 demand days.
            for day in range(1, 16):
                add_row(4, f"2025-12-{day:02d}", 1.0)
            add_row(4, "2026-01-01", 0.0)

            # Removed: enough total demand days, but only 1/11 in the last year.
            for day in range(1, 15):
                add_row(5, f"2024-12-{day:02d}", 1.0)
            for day in range(22, 32):
                add_row(5, f"2025-12-{day:02d}", 1.0 if day == 22 else 0.0)
            add_row(5, "2026-01-01", 0.0)

            # Retained: exactly 15/150 demand days in the last year.
            start_date = date(2025, 1, 1)
            for day in range(1, 151):
                add_row(
                    6,
                    (start_date + timedelta(days=day - 1)).isoformat(),
                    1.0 if day <= 15 else 0.0,
                )

            # Removed: 14 recent demand days.
            for day in range(1, 16):
                add_row(8, f"2025-12-{day:02d}", 1.0 if day <= 14 else 0.0)
            add_row(8, "2026-01-01", 0.0)

            values_sql = ",\n".join(
                f"({artikel_id}, {markt_id}, DATE '{date_sql}', "
                f"{demand}, {str(is_active).upper()})"
                for artikel_id, markt_id, date_sql, demand, is_active in value_rows
            )
            con.execute(
                f"""
                COPY (
                    SELECT * FROM (VALUES
                        {values_sql}
                    ) AS t(
                        ARTIKEL_ID,
                        MARKT_ID,
                        DATE,
                        ABVERKAUFTE_MENGE_KG,
                        is_active
                    )
                ) TO '{input_path}' (FORMAT PARQUET)
                """
            )

            main(in_dir, out_dir, threads=1)

            retained_ids = {
                row[0]
                for row in con.execute(
                    f"""
                    SELECT DISTINCT ARTIKEL_ID
                    FROM read_parquet('{out_dir / "*.parquet"}')
                    ORDER BY ARTIKEL_ID
                    """
                ).fetchall()
            }
            self.assertEqual(retained_ids, {1, 4, 6})


if __name__ == "__main__":
    unittest.main()
