"""Maturity-aware benchmark forecasts and evaluation metrics."""

from src.models.benchmark.config import BenchmarkDesign, load_benchmark_design
from src.models.benchmark.evaluation import BenchmarkResult, run_benchmark
from src.models.benchmark.metrics import segment_wape, summarize_models
from src.models.benchmark.models import MODEL_COLUMNS, PRIMARY_MODEL
from src.models.baseline.evaluation import BASELINE_MODEL_LABELS

__all__ = [
    "BenchmarkDesign",
    "BenchmarkResult",
    "BASELINE_MODEL_LABELS",
    "MODEL_COLUMNS",
    "PRIMARY_MODEL",
    "load_benchmark_design",
    "run_benchmark",
    "segment_wape",
    "summarize_models",
]
