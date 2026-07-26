"""Run only the direct daily L2 LightGBM model."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.models.benchmark.config import DEFAULT_DESIGN_PATH
from src.models.core.cli import model_parser, rebuild_features
from src.models.lightgbm.l2.spec import SPEC
from src.models.lightgbm.runner import run_specs
from src.models.results import DEFAULT_RESULTS_DIR


def run(
    *,
    design_path: Path = DEFAULT_DESIGN_PATH,
    data_dir: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    feature_dataset_path: Path | None = None,
    force_feature_recompute: bool = False,
) -> list[Path]:
    return run_specs(
        (SPEC,), design_path=design_path, data_dir=data_dir,
        results_dir=results_dir, feature_dataset_path=feature_dataset_path,
        rebuild_features=force_feature_recompute,
    )


def main() -> None:
    args = model_parser(__doc__ or "", features=True).parse_args()
    paths = run(
        design_path=args.design, data_dir=args.data_dir,
        results_dir=args.results_dir, feature_dataset_path=args.feature_dataset,
        force_feature_recompute=rebuild_features(args),
    )
    print(f"Wrote {len(paths)} L2 result files to {args.results_dir}")


if __name__ == "__main__":
    main()
