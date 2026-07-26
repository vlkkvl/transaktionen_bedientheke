# Model architecture

Model families expose a `main.py` runner, while independently configurable
models live in their own package and export a `ModelSpec` from `spec.py`.

## Commands

```bash
# One model
python -m src.models.lightgbm.tweedie
python -m src.models.lightgbm.l2
python -m src.models.lightgbm.two_stage
python -m src.models.lightgbm.weekly_total
python -m src.models.lightgbm.ablation

# Entire model families
python -m src.models.lightgbm
python -m src.models.baseline
python -m src.models.benchmark
```

Every LightGBM command accepts `--feature-dataset DIRECTORY`. Directory paths
use the versioned, origin-partitioned feature store. Existing `.parquet` paths
continue to use the legacy single-file cache. Use `--rebuild-features` only
when an intentional rebuild is required.

## Adding a LightGBM model

1. Create `src/models/lightgbm/<model>/` with `model.py`, `spec.py`, `main.py`,
   and `__main__.py`.
2. Keep the model implementation and all its statistical hyperparameters in
   `model.py`. Backtest, runtime, and fit-control settings are composed from
   shared types but remain independently configurable for each model.
3. Implement the fit adapter declared by `ModelSpec`.
4. Add the specification to `src.models.lightgbm.registry.LIGHTGBM_MODELS`.

All registered variants receive the same materialized daily feature frame.
Model-local transformations, such as the weekly-total aggregation, operate on
that frame and do not rebuild historical features.

The shared feature schema is exposed by `lightgbm/features/definition.py`,
feature construction by `lightgbm/features/builder.py`, and cache lifecycle by
`lightgbm/features/store.py`. Model files do not own feature computation.

There are no root-level `run_*.py` scripts. Run a model from its own package;
the family-level `lightgbm/main.py` is the only all-LightGBM entry point.
