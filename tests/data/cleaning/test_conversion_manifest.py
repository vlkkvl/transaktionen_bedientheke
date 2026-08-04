from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.data.cleaning.convert.common import (
    conversion_is_current,
    write_source_manifest,
)


class ConversionManifestTest(unittest.TestCase):
    def test_detects_added_or_changed_raw_files(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw_dir = base / "raw"
            out_dir = base / "yearly"
            raw_dir.mkdir()
            out_dir.mkdir()
            first = raw_dir / "first.csv.gz"
            first.write_bytes(b"first")
            (out_dir / "transactions_year_2025.parquet").write_bytes(b"parquet")

            write_source_manifest([first], out_dir)
            self.assertTrue(conversion_is_current([first], out_dir))

            second = raw_dir / "second.csv.gz"
            second.write_bytes(b"second")
            self.assertFalse(conversion_is_current([first, second], out_dir))

            write_source_manifest([first, second], out_dir)
            first.write_bytes(b"changed")
            self.assertFalse(conversion_is_current([first, second], out_dir))


if __name__ == "__main__":
    unittest.main()
