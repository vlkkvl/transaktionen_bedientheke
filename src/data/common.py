"""Shared helpers for data cleaning and preparation scripts."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import duckdb


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a script nested below it."""
    start = (start or Path(__file__)).resolve()
    for parent in start.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return start.parents[2]


ROOT = find_project_root()


def sql_literal(value: Path | str) -> str:
    """Escape a value for use as a single-quoted DuckDB SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def ident(name: str) -> str:
    """Escape a DuckDB identifier."""
    return '"' + name.replace('"', '""') + '"'


def read_parquet_expr(path: Path | str, *, filename: bool = False) -> str:
    option = ", filename=true" if filename else ""
    return f"read_parquet({sql_literal(path)}{option})"


def parquet_files(directory: Path, pattern: str = "transactions_year_*.parquet") -> list[Path]:
    return sorted(directory.glob(pattern))


def require_parquet_files(
    directory: Path,
    pattern: str = "transactions_year_*.parquet",
) -> list[Path]:
    files = parquet_files(directory, pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return files


def clear_parquet_outputs(
    directory: Path,
    pattern: str = "transactions_year_*.parquet",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob(pattern):
        path.unlink()


def configure_duckdb(threads: int = 8) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order = false")
    return con


def columns_for_expr(con: duckdb.DuckDBPyConnection, read_expr: str) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
    return [row[0] for row in rows]


def step(message: str, started_at: float) -> float:
    finished_at = perf_counter()
    print(f"{message} ({finished_at - started_at:.1f}s)")
    return finished_at
