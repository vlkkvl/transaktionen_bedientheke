"""Shared command-line arguments for model entry points."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.models.benchmark.config import DEFAULT_DESIGN_PATH
from src.models.results import DEFAULT_RESULTS_DIR


def model_parser(description: str, *, features: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN_PATH)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    if features:
        parser.add_argument(
            "--feature-dataset",
            type=Path,
            help=(
                "Feature-store directory. A .parquet path selects the legacy "
                "single-file cache."
            ),
        )
        parser.add_argument("--rebuild-features", action="store_true")
        parser.add_argument(
            "--force-feature-recompute",
            action="store_true",
            help=argparse.SUPPRESS,
        )
    return parser


def rebuild_features(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "rebuild_features", False)
        or getattr(args, "force_feature_recompute", False)
    )
