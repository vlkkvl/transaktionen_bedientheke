"""Run the Tweedie feature ablation with the shared design and features."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import duckdb

from src.models.benchmark import load_benchmark_design
from src.models.benchmark.config import BenchmarkDesign, DEFAULT_DESIGN_PATH
from src.models.core.cli import model_parser, rebuild_features
from src.models.lightgbm.features.builder import GlobalLightGBMConfig
from src.models.lightgbm.runner import prepare_run_frames
from src.models.lightgbm.ablation.model import (
    LightGBMAblationResult,
    fit_tweedie_feature_ablation,
)
from src.models.results import DEFAULT_RESULTS_DIR, write_forecasts, write_result


def save_result(
    result: LightGBMAblationResult,
    design: BenchmarkDesign,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> list[Path]:
    paths = list(
        write_forecasts(
            result.forecasts, design, results_dir=results_dir
        ).values()
    )
    for model, summary in result.training_summary.groupby(
        "model", observed=True, sort=False
    ):
        paths.append(
            write_result(
                summary,
                str(model),
                design,
                artifact="training_summaries",
                results_dir=results_dir,
            )
        )
    for model, importance in result.feature_importance.groupby(
        "model", observed=True, sort=False
    ):
        paths.append(
            write_result(
                importance,
                str(model),
                design,
                artifact="feature_importance",
                results_dir=results_dir,
            )
        )
    paths.append(
        write_result(
            result.feature_design,
            "lightgbm_tweedie_ablation_design",
            design,
            artifact="diagnostics",
            results_dir=results_dir,
        )
    )
    return paths


def run(
    *,
    design_path: Path = DEFAULT_DESIGN_PATH,
    data_dir: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    feature_dataset_path: Path | None = None,
    force_feature_recompute: bool = False,
) -> list[Path]:
    design = load_benchmark_design(design_path)
    config = GlobalLightGBMConfig()
    con = duckdb.connect()
    con.execute("SET temp_directory='/tmp/ba_lightgbm_ablation_duckdb'")
    try:
        frames = prepare_run_frames(
            con,
            design,
            config,
            data_dir=data_dir,
            feature_dataset_path=feature_dataset_path,
            rebuild_features=force_feature_recompute,
        )
        return save_result(
            fit_tweedie_feature_ablation(frames, config), design, results_dir
        )
    finally:
        con.close()


def main() -> None:
    args = model_parser(__doc__ or "", features=True).parse_args()
    paths = run(
        design_path=args.design,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        feature_dataset_path=args.feature_dataset,
        force_feature_recompute=rebuild_features(args),
    )
    print(f"Wrote {len(paths)} ablation result files to {args.results_dir}")


if __name__ == "__main__":
    main()
