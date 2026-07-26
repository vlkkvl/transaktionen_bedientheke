"""Run the mature-series benchmark models and persist their results."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.benchmark import BenchmarkResult, load_benchmark_design, run_benchmark
from src.models.benchmark.config import DEFAULT_DESIGN_PATH
from src.models.core.cli import model_parser
from src.models.results import DEFAULT_RESULTS_DIR, write_forecasts, write_result


def save_result(
    result: BenchmarkResult,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> list[Path]:
    paths = list(
        write_forecasts(
            result.forecasts, result.design, results_dir=results_dir
        ).values()
    )
    paths.extend(
        [
            write_result(
                result.origin_summary,
                "benchmark_origin_summary",
                result.design,
                artifact="diagnostics",
                results_dir=results_dir,
            ),
            write_result(
                result.data_audit,
                "benchmark_data_audit",
                result.design,
                artifact="diagnostics",
                results_dir=results_dir,
            ),
        ]
    )
    return paths


def run(
    *,
    design_path: Path = DEFAULT_DESIGN_PATH,
    data_dir: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[Path]:
    design = load_benchmark_design(design_path)
    return save_result(
        run_benchmark(design=design, data_dir=data_dir), results_dir
    )


def main() -> None:
    args = model_parser(__doc__ or "").parse_args()
    paths = run(
        design_path=args.design,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
    print(f"Wrote {len(paths)} benchmark result files to {args.results_dir}")


if __name__ == "__main__":
    main()
