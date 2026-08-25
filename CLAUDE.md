# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A bachelor's-thesis codebase for intermittent retail demand forecasting on German grocery
transaction data (fresh meat/sausage, article × store × day). It compares classical
intermittent-demand baselines against global LightGBM models under a fixed expanding-origin
backtest.

## Environment and commands

There is no `pyproject.toml`, `setup.py`, or `Makefile`. Dependencies are pinned in
`requirements.txt`; the interpreter is the in-repo virtualenv (Python 3.12).

```bash
.venv/bin/python -m pytest tests/ -q                       # full suite (92 tests)
.venv/bin/python -m pytest tests/models/machine_learning/test_lightgbm.py -q
.venv/bin/python -m pytest tests/models/machine_learning/test_lightgbm.py -k "position_lift" -x -q
```

Tests are `unittest.TestCase` classes executed via pytest. There is no linter or formatter
configured.

Running models (each writes CSVs under `reports/results/`):

```bash
.venv/bin/python -m src.models.lightgbm                    # all 5 registered LightGBM variants
.venv/bin/python -m src.models.lightgbm.two_stage          # one variant
.venv/bin/python -m src.models.lightgbm.two_stage --rebuild-features
.venv/bin/python -m src.models.benchmark                   # naive/benchmark models
.venv/bin/python -m src.models.baseline                    # Croston, SBA, TSB, SES, agg-then-disagg
.venv/bin/python -m src.data.pipeline                      # raw -> interim -> processed
```

Shared CLI flags come from `src/models/core/cli.py`: `--design`, `--data-dir`,
`--results-dir`, plus `--feature-dataset` and `--rebuild-features` for feature-backed models.

`data/{raw,interim,processed}` and `reports/results/` are gitignored and large (forecast CSVs
run to ~800 MB). Read them with chunked/columnar reads, never whole-file.

## Architecture

### The forecast design is a single source of truth

`reports/config/forecast_design.json` fixes the backtest: first origin, 7-day origin spacing,
7-day horizon, 20 evaluation origins, minimum series maturity. `BenchmarkDesign`
(`src/models/benchmark/config.py`) loads and validates it — `FIRST_ORIGIN` must be a Monday.

Critically, `runner._validate_design` cross-checks the JSON design against each model's own
config and **raises** on mismatch (origin spacing, horizon, test-origin count). Changing the
design without changing model configs is a hard error, not a silent divergence. All model
families read the same file, so benchmark, baseline, and LightGBM results are always
comparable.

### Feature store: content-addressed, origin-partitioned

`src/models/lightgbm/features/store.py` builds features once per origin and shares them across
model processes. The cache generation is fingerprinted from a DuckDB source signature (row
count, period range, hash of the input rows) plus `FEATURE_SET_VERSION` and
`FEATURE_BUILDER_VERSION`.

**Bumping `FEATURE_BUILDER_VERSION` invalidates every partition and forces a full rebuild of
all origins.** That is the intended mechanism when feature logic changes — do it, don't try to
patch partitions in place. Changed source data selects a new generation directory
automatically; unchanged partitions are never rewritten (asserted by tests).

### Feature contract

`src/models/lightgbm/features/builder.py` owns `FEATURE_COLUMNS`, `CATEGORICAL_FEATURES`, and
`FEATURE_DESCRIPTIONS`. A module-level guard raises `RuntimeError` at import if the description
map and the column tuple disagree — **every new feature needs a prose description in the same
commit or nothing imports**. `features/definition.py` re-exports the contract so models never
import the builder's internals.

`REMOVED_FEATURE_COLUMNS` records columns deliberately dropped, so stale caches are detected
rather than silently re-admitted.

**Leakage discipline:** every historical feature is computed strictly before the forecast
origin and restricted to active rows. The descriptions state this per feature; preserve the
phrasing and the invariant when adding features.

### Model registry and config ownership

`src/models/lightgbm/registry.py` lists five `ModelSpec`s (`src/models/core/contracts.py`):

| Model | Objective |
|---|---|
| `l2` | daily direct, `regression_l2` |
| `tweedie` | daily direct, Tweedie |
| `two_stage` | binary occurrence × gamma positive-quantity |
| `two_stage_quantile` | binary occurrence × pinball-loss quantity at P10/P50/P90 (predictive intervals, not the shipped point forecast) |
| `weekly_total` | Tweedie on 7-day sums + weekday allocation |

Each variant owns an independent config dataclass so tuning one cannot leak into another —
`tests/models/test_model_architecture.py` enforces this explicitly. Shared *shapes* live in
`src/models/core/config.py` (`ForecastWindow`, `RuntimeConfig`, `FitControl`) but each model
config holds its own instance.

Adding a variant means: `model.py` (fit function + config) → `spec.py` (a `ModelSpec`) → add to
`LIGHTGBM_MODELS` and `LIGHTGBM_MODEL_LABELS` → add a file name in `MODEL_FILE_NAMES`.

### Two model structures worth knowing before editing

**Two-stage** (`two_stage/model.py`) fits an occurrence classifier on `actual > 0` and a gamma
model on positive rows only, then multiplies clipped probability × quantity. The stages get
*different* feature subsets (stage-specific routing — e.g. pooled occurrence lift goes only to
the occurrence booster), and the quantity label is normalised by the article-store target mean
and rescaled after prediction.

**Weekly total** (`weekly_total/model.py`) collapses daily rows to one row per
article-store-week via `make_weekly_frame`, predicts a single weekly value, then splits it over
days. That allocation is strictly total-preserving (audited to ~1e-13 in
`reports/results/allocation_audits/`), so scoring at weekly grain measures the model's native
output and the allocation step is irrelevant there. Its feature list
(`WEEKLY_FEATURE_COLUMNS`) is a hand-maintained aggregation and currently lags the daily
contract by ~17 columns — including all event-lift features.

### Results

`src/models/results.py` defines stable CSV paths under `reports/results/{forecasts,
feature_importance, training_summaries, run_configs, validation_predictions,
allocation_audits}`. `MODEL_FILE_NAMES` maps internal model names to public file names — the
two differ, so always go through `model_file_name()` / `result_path()` rather than composing
paths by hand. `DATE_COLUMNS` drives date restoration on read.

## Development workflow

Analysis lives in numbered notebooks (`00_data_cleaning` → `09_decision_layer`) and prose
findings in `reports/*.md`. The dominant loop for model work is a **feature batch**, visible in
`notebooks/05_machine_learning/05_06`–`05_08`:

1. State the mechanism and the expected effect before running anything.
2. Add features to `builder.py` with descriptions and unit tests.
3. Bump `FEATURE_BUILDER_VERSION`, rebuild all partitions, refit **only** the affected model.
4. Back up the previous run to `reports/results/backup_<batch_name>/` and compare against it.
5. Report WAPE at row / article-store-week / article-day grains **plus relative bias** — bias
   near zero is a standing requirement, so a WAPE gain bought with bias is not a gain.

Verify the hypothesised mechanism exists in the data before attributing a metric change to it;
several batches found the mechanism real but too small to explain the target error.
