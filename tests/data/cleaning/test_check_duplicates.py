from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb

from src.data.cleaning.check_duplicates import main


class CheckDuplicatesTest(unittest.TestCase):
    def test_materializes_one_merged_row_per_duplicate_key(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            in_dir = base / "filtered"
            out_dir = base / "transactions_no_dups"
            in_dir.mkdir()
            input_path = in_dir / "transactions_year_2025.parquet"

            con = duckdb.connect()
            con.execute(
                f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, 10, 100, DATE '2025-01-01', TIME '10:00:00', 2.0,
                         200, 0, 301, 2025, 23, 1, DATE '2025-06-02',
                         DATE '2025-06-07', 0, 0, 'selected'),
                        (1, 10, 100, DATE '2025-01-01', TIME '10:00:00', 2.0,
                         201, 1, 302, 2025, 23, 1, DATE '2025-06-02',
                         DATE '2025-06-07', 1, 1, 'other'),
                        (2, 10, 101, DATE '2025-01-02', TIME '11:00:00', 1.0,
                         300, 0, 303, 2025, 23, 1, DATE '2025-06-02',
                         DATE '2025-06-07', 0, 0, 'unique')
                    ) AS rows(
                        ARTIKEL_ID, MARKT_ID, BON_ID, DATE, TIME, UMS_MENGE,
                        EAN_ID, AKTION_KENNZEICHEN, AKTIONSNUMMER, AKTIONSJAHR,
                        AKTIONSWOCHE, AKTIONSZUSATZ, GUELTIG_VON, GUELTIG_BIS,
                        RABATT, ARTIKELRABATT, note
                    )
                ) TO '{input_path}' (FORMAT PARQUET)
                """
            )

            main(in_dir=in_dir, out_dir=out_dir)

            output_path = out_dir / input_path.name
            cursor = con.execute(
                f"""
                SELECT *
                FROM read_parquet('{output_path}')
                ORDER BY ARTIKEL_ID
                """
            )
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()

            self.assertNotIn("EAN_ID", columns)
            self.assertNotIn("AKTIONSNUMMER", columns)
            self.assertNotIn("AKTIONSJAHR", columns)
            self.assertNotIn("AKTIONSWOCHE", columns)
            self.assertNotIn("AKTIONSZUSATZ", columns)
            self.assertNotIn("GUELTIG_VON", columns)
            self.assertNotIn("GUELTIG_BIS", columns)
            self.assertIn("RABATT", columns)
            self.assertIn("ARTIKELRABATT", columns)
            self.assertEqual(len(rows), 2)
            first = dict(zip(columns, rows[0]))
            self.assertEqual(first["AKTION_KENNZEICHEN"], 1)
            self.assertEqual(first["RABATT"], 1)
            self.assertEqual(first["ARTIKELRABATT"], 1)
            self.assertEqual(first["note"], "selected")


if __name__ == "__main__":
    unittest.main()
