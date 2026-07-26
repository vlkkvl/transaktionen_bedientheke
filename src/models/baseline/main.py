"""Run all classical baseline models using the shared forecast design."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.benchmark import load_benchmark_design, run_benchmark
from src.models.benchmark.config import DEFAULT_DESIGN_PATH
from src.models.benchmark.main import save_result
from src.models.core.cli import model_parser
from src.models.results import DEFAULT_RESULTS_DIR


def run(
    *,
    design_path: Path = DEFAULT_DESIGN_PATH,
    data_dir: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[Path]:
    design = load_benchmark_design(design_path)
    result = run_benchmark(
        design=design,
        data_dir=data_dir,
        include_extended_baselines=True,
    )
    return save_result(result, results_dir)


def main() -> None:
    args = model_parser(__doc__ or "").parse_args()
    paths = run(
        design_path=args.design,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
    print(f"Wrote {len(paths)} baseline result files to {args.results_dir}")


if __name__ == "__main__":
    main()
