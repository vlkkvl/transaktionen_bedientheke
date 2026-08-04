from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings

import duckdb
import numpy as np
import pandas as pd

from src.data.preparation.discover_sparse_regions import (
    EXCLUDED_REGIONS,
    PSEUDO_890_SPARSE_SCALE_OUTLIER_IDS,
    REGION_FLAG_COLUMNS,
    apply_exclusion_flags,
    assign_regions,
    materialize_filtered_transactions,
    warn_for_missing_sparse_scale_outliers,
)


class SparseRegionDiscoveryTest(unittest.TestCase):
    def test_assignments_are_exclusive_and_cover_all_requested_regions(self) -> None:
        rows = []
        product_id = 1
        for segment in ["FCM 890", "FCM 900", "Pseudo 890", "Pseudo 900"]:
            source_class, category_id = segment.split()
            for index in range(12):
                frequency = float(np.geomspace(0.001, 0.8, 12)[index])
                scale = float(np.geomspace(0.05, 200.0, 12)[index])
                active_store_days = 1_000 + index
                demand_store_days = max(1, round(active_store_days * frequency))
                rows.append(
                    {
                        "source_class": source_class,
                        "category_id": int(category_id),
                        "segment": segment,
                        "ARTIKEL_ID": product_id,
                        "ARTIKEL_BEZ": f"Product {product_id}",
                        "VERKAUFSEINHEIT": "kg",
                        "stores": 10 + index,
                        "active_store_days": active_store_days,
                        "demand_store_days": demand_store_days,
                        "total_demand_kg": demand_store_days * scale,
                        "demand_frequency": frequency,
                        "kg_per_demand_day": scale,
                        "kg_per_active_day": frequency * scale,
                        "first_observed_date": pd.Timestamp("2023-01-01"),
                        "last_demand_date": pd.Timestamp("2025-12-31"),
                    }
                )
                product_id += 1

        assignments = assign_regions(pd.DataFrame(rows))

        self.assertEqual(set(assignments["region"]), set(REGION_FLAG_COLUMNS))
        self.assertTrue(assignments["ARTIKEL_ID"].is_unique)
        self.assertTrue(
            assignments[list(REGION_FLAG_COLUMNS.values())].sum(axis=1).eq(1).all()
        )
        self.assertEqual(
            set(assignments.loc[assignments.exclude_from_daily, "region"]),
            EXCLUDED_REGIONS,
        )
        pseudo_low = assignments[assignments.region == "pseudo_890_low_velocity"]
        self.assertTrue(
            pseudo_low["pseudo_890_low_velocity_subcluster"].between(1, 4).all()
        )

    def test_explicit_sparse_scale_products_are_excluded_without_relabeling(self) -> None:
        outlier_id = next(iter(PSEUDO_890_SPARSE_SCALE_OUTLIER_IDS))
        assignments = pd.DataFrame(
            {
                "ARTIKEL_ID": [outlier_id, 1],
                "segment": ["Pseudo 890", "Pseudo 890"],
                "region": ["pseudo_890_high_velocity", "pseudo_890_high_velocity"],
            }
        )

        flagged = apply_exclusion_flags(assignments).set_index("ARTIKEL_ID")

        self.assertTrue(flagged.loc[outlier_id, "exclude_from_daily"])
        self.assertTrue(flagged.loc[outlier_id, "is_pseudo_890_sparse_scale_outlier"])
        self.assertEqual(
            flagged.loc[outlier_id, "exclusion_reason"],
            "pseudo_890_sparse_scale_outlier",
        )
        self.assertEqual(
            flagged.loc[outlier_id, "region"], "pseudo_890_high_velocity"
        )
        self.assertFalse(flagged.loc[1, "exclude_from_daily"])

    def test_missing_configured_outliers_warn_without_failing(self) -> None:
        present_id = next(iter(PSEUDO_890_SPARSE_SCALE_OUTLIER_IDS))
        expected_missing = PSEUDO_890_SPARSE_SCALE_OUTLIER_IDS - {present_id}
        assignments = pd.DataFrame({"ARTIKEL_ID": [present_id, 1]})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            missing = warn_for_missing_sparse_scale_outliers(assignments)

        self.assertEqual(missing, expected_missing)
        self.assertEqual(len(caught), 1)
        self.assertIn(str(sorted(expected_missing)), str(caught[0].message))

    def test_filtered_output_uses_materialized_product_ids(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            in_dir = root / "input"
            out_dir = root / "output"
            in_dir.mkdir()
            flags_path = root / "flags.parquet"
            input_path = in_dir / "transactions_year_2025.parquet"

            con = duckdb.connect()
            con.execute(
                f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, 10, DATE '2025-01-01', 1.0),
                        (2, 10, DATE '2025-01-01', 2.0),
                        (3, 10, DATE '2025-01-01', 3.0),
                        (4, 10, DATE '2025-01-01', 4.0)
                    ) AS t(ARTIKEL_ID, MARKT_ID, DATE, ABVERKAUFTE_MENGE_KG)
                ) TO '{input_path}' (FORMAT PARQUET)
                """
            )
            con.execute(
                f"""
                COPY (
                    SELECT * FROM (VALUES
                        (1, 'fcm_890', NULL, FALSE),
                        (2, 'fcm_900_low_intensity', 'fcm_900_low_intensity', TRUE),
                        (3, 'pseudo_900_high_intensity', 'pseudo_900_high_intensity', TRUE),
                        (4, 'pseudo_890_high_velocity', NULL, FALSE)
                    ) AS f(ARTIKEL_ID, region, exclusion_reason, exclude_from_daily)
                ) TO '{flags_path}' (FORMAT PARQUET)
                """
            )

            materialize_filtered_transactions(
                in_dir,
                out_dir,
                flags_path=flags_path,
                threads=1,
            )

            retained_ids = {
                row[0]
                for row in con.execute(
                    f"SELECT ARTIKEL_ID FROM read_parquet('{out_dir / '*.parquet'}')"
                ).fetchall()
            }
            self.assertEqual(retained_ids, {1, 4})


if __name__ == "__main__":
    unittest.main()
