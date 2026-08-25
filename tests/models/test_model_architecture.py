from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd

from src.models.benchmark.config import BenchmarkDesign
from src.models.benchmark.evaluation import _create_assessed_origins
from src.models.benchmark.models import create_history_features
from src.models.core.config import FitControl, ForecastWindow, RuntimeConfig
from src.models.lightgbm.features.store import LightGBMFeatureStore
from src.models.lightgbm.l2.model import L2Config
from src.models.lightgbm.registry import LIGHTGBM_MODELS
from src.models.lightgbm.tweedie.model import (
    TweedieConfig,
    TweedieHyperparameters,
)
from src.models.lightgbm.two_stage.model import TwoStageConfig
from src.models.lightgbm.two_stage_quantile.model import TwoStageQuantileConfig
from src.models.lightgbm.weekly_total.model import WeeklyTotalConfig
from src.models.lightgbm.runner import run_specs


class ModelConfigurationTest(unittest.TestCase):
    def test_registry_has_independently_configured_model_types(self) -> None:
        configs = [spec.make_config() for spec in LIGHTGBM_MODELS]

        self.assertEqual(len(configs), 5)
        self.assertEqual(len({type(config) for config in configs}), 5)
        self.assertTrue(all(spec.family == "lightgbm" for spec in LIGHTGBM_MODELS))

    def test_tweedie_tuning_does_not_change_l2_parameters(self) -> None:
        l2 = L2Config()
        tweedie = TweedieConfig(
            hyperparameters=replace(
                TweedieHyperparameters(), learning_rate=0.02, num_leaves=31
            )
        )

        self.assertEqual(l2.parameters()["learning_rate"], 0.05)
        self.assertEqual(l2.parameters()["num_leaves"], 63)
        self.assertEqual(tweedie.parameters()["learning_rate"], 0.02)
        self.assertEqual(tweedie.parameters()["num_leaves"], 31)

    def test_two_stage_owns_separate_stage_parameters(self) -> None:
        config = TwoStageConfig()

        self.assertEqual(config.occurrence_parameters()["objective"], "binary")
        self.assertEqual(
            config.occurrence_parameters()["metric"], "binary_logloss"
        )
        self.assertEqual(config.quantity_parameters()["objective"], "gamma")
        self.assertEqual(config.quantity_parameters()["metric"], "gamma")

    def test_two_stage_quantile_owns_per_level_quantity_parameters(self) -> None:
        config = TwoStageQuantileConfig()

        self.assertEqual(config.quantile_levels, (0.1, 0.5, 0.9))
        self.assertEqual(config.occurrence_parameters()["objective"], "binary")
        for level in config.quantile_levels:
            parameters = config.quantity_parameters(level)
            self.assertEqual(parameters["objective"], "quantile")
            self.assertEqual(parameters["metric"], "quantile")
            self.assertEqual(parameters["alpha"], level)
        gamma = TwoStageConfig().quantity_parameters()
        self.assertEqual(gamma["objective"], "gamma")
        self.assertNotIn("alpha", gamma)

    def test_objectives_and_early_stopping_metrics_are_model_specific(self) -> None:
        l2 = L2Config().parameters()
        tweedie = TweedieConfig().parameters()
        weekly = WeeklyTotalConfig().parameters()

        self.assertEqual((l2["objective"], l2["metric"]), ("regression_l2", "rmse"))
        self.assertEqual(
            (tweedie["objective"], tweedie["metric"]),
            ("tweedie", "tweedie"),
        )
        self.assertEqual(
            (weekly["objective"], weekly["metric"]),
            ("tweedie", "tweedie"),
        )


class FeatureStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.con = duckdb.connect()
        self.addCleanup(self.con.close)
        self.con.execute(
            """
            CREATE TEMP TABLE benchmark_daily_rows (
                ARTIKEL_ID BIGINT,
                MARKT_ID BIGINT,
                period DATE,
                demand DOUBLE,
                is_active BOOLEAN,
                reason_closed VARCHAR,
                action_flag TINYINT,
                sourcing_group VARCHAR,
                category_id INTEGER
            )
            """
        )
        dates = pd.date_range("2023-01-02", periods=440, freq="D")
        rows = [
            (
                1,
                10,
                date.date(),
                float(index % 5 == 0),
                True,
                None,
                int(index % 28 == 0),
                "FCM",
                890,
            )
            for index, date in enumerate(dates)
        ]
        self.con.executemany(
            "INSERT INTO benchmark_daily_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.design = BenchmarkDesign(
            28, pd.Timestamp("2024-01-08"), 7, 7, max_origins=2
        )
        self.origins = pd.date_range(self.design.first_origin, periods=2, freq="7D")
        create_history_features(self.con)
        _create_assessed_origins(self.con, self.origins, self.design)
        self.store = LightGBMFeatureStore(Path(self.directory.name))

    def test_cache_hit_does_not_rewrite_available_partition(self) -> None:
        first = self.store.ensure(
            self.con, origins=self.origins[:1], design=self.design
        )
        first_mtime = first.parquet_paths[0].stat().st_mtime_ns

        expanded = self.store.ensure(
            self.con, origins=self.origins, design=self.design
        )

        self.assertEqual(first.fingerprint, expanded.fingerprint)
        self.assertEqual(first_mtime, expanded.parquet_paths[0].stat().st_mtime_ns)
        self.assertEqual(len(expanded.parquet_paths), 2)
        self.assertTrue(all(path.exists() for path in expanded.parquet_paths))

    def test_source_change_selects_a_new_cache_generation(self) -> None:
        first = self.store.ensure(
            self.con, origins=self.origins[:1], design=self.design
        )
        self.con.execute(
            """UPDATE benchmark_daily_rows SET demand = demand + 1
            WHERE period = DATE '2023-01-02'"""
        )

        changed = self.store.ensure(
            self.con, origins=self.origins[:1], design=self.design
        )

        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        self.assertNotEqual(first.directory, changed.directory)

    def test_family_runner_shares_store_and_persists_each_config(self) -> None:
        root = Path(self.directory.name)
        design_path = root / "design.json"
        design_path.write_text(
            json.dumps(
                {
                    "DATA_DIR": "unused",
                    "MIN_ACTIVE_DAYS": self.design.min_active_days,
                    "FIRST_ORIGIN": self.design.first_origin.date().isoformat(),
                    "ORIGIN_SPACING_DAYS": self.design.origin_spacing_days,
                    "FORECAST_HORIZON_DAYS": self.design.forecast_horizon_days,
                    "MAX_ORIGINS": self.design.max_origins,
                }
            ),
            encoding="utf-8",
        )
        window = ForecastWindow(
            training_origins=4,
            validation_origins=1,
            test_origins=2,
            refit_interval_origins=2,
            origin_spacing_days=7,
        )
        runtime = RuntimeConfig(num_threads=2)
        fit_control = FitControl(num_boost_round=5, early_stopping_rounds=2)
        configs = {
            spec.name: replace(
                spec.make_config(),
                window=window,
                runtime=runtime,
                fit_control=fit_control,
            )
            for spec in LIGHTGBM_MODELS
        }

        paths = run_specs(
            LIGHTGBM_MODELS,
            design_path=design_path,
            results_dir=root / "results",
            feature_dataset_path=root / "features",
            configs=configs,
            connection=self.con,
        )

        self.assertTrue(paths)
        self.assertTrue(all(path.exists() for path in paths))
        self.assertEqual(
            sum(path.parent.name == "run_configs" for path in paths),
            len(LIGHTGBM_MODELS),
        )
        validation_paths = [
            path for path in paths if path.parent.name == "validation_predictions"
        ]
        self.assertEqual(len(validation_paths), 2)  # two_stage + two_stage_quantile
        for validation_path in validation_paths:
            validation_predictions = pd.read_csv(validation_path)
            self.assertIn("occurrence_probability", validation_predictions)
            self.assertEqual(
                len(validation_predictions),
                window.validation_origins * 7,
            )
        generations = list(
            (root / "features" / "lightgbm_daily" / "v1").iterdir()
        )
        self.assertEqual(len(generations), 1)


if __name__ == "__main__":
    unittest.main()
