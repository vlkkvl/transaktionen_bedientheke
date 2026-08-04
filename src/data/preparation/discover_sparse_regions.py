"""Discover product demand regions and materialize their product-level flags.

The clustering source is the complete expanded daily calendar. No minimum-
demand, history-length, recency, or late-tail rule is applied before profiles
are built. Products are clustered separately within the four business
segments using demand frequency and kilograms per positive store-day.

Outputs:

* ``data/interim/product_sparse_regions/product_sparse_region_flags.parquet``
* ``data/interim/product_sparse_regions/cluster_diagnostics.parquet``
* ``data/interim/transactions_dst_daily_filtered/transactions_year_*.parquet``

The filtered transaction output removes all product-store series belonging to
FCM 900 low intensity or Pseudo 900 high intensity, plus configured Pseudo 890
sparse high-scale products that occur in the current data. The product flags
remain the source of truth for those exclusions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import warnings

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.common import (
    ROOT,
    clear_parquet_outputs,
    configure_duckdb,
    read_parquet_expr,
    require_parquet_files,
    sql_literal,
)


IN_DIR = ROOT / "data" / "interim" / "transactions_dst_over_days"
REGION_DIR = ROOT / "data" / "interim" / "product_sparse_regions"
REGION_FLAGS_PATH = REGION_DIR / "product_sparse_region_flags.parquet"
DIAGNOSTICS_PATH = REGION_DIR / "cluster_diagnostics.parquet"
FILTERED_OUT_DIR = ROOT / "data" / "interim" / "transactions_dst_daily_no_outliers"

DEMAND_COL = "ABVERKAUFTE_MENGE_KG"
RANDOM_STATE = 42
N_INIT = 50
SEGMENT_ORDER = ["FCM 890", "FCM 900", "Pseudo 890", "Pseudo 900"]
REGION_FLAG_COLUMNS = {
    "pseudo_890_high_velocity": "is_pseudo_890_high_velocity",
    "pseudo_890_low_velocity": "is_pseudo_890_low_velocity",
    "fcm_890": "is_fcm_890",
    "fcm_900_low_intensity": "is_fcm_900_low_intensity",
    "fcm_900_high_intensity": "is_fcm_900_high_intensity",
    "pseudo_900_high_intensity": "is_pseudo_900_high_intensity",
    "pseudo_900_low_intensity": "is_pseudo_900_low_intensity",
}
EXCLUDED_REGIONS = {
    "fcm_900_low_intensity",
    "pseudo_900_high_intensity",
}
PSEUDO_890_SPARSE_SCALE_OUTLIER_IDS = {388783, 515828}
EXCLUSION_REASON_BY_REGION = {
    "fcm_900_low_intensity": "fcm_900_low_intensity",
    "pseudo_900_high_intensity": "pseudo_900_high_intensity",
}


def create_product_profile_view(
    con: duckdb.DuckDBPyConnection,
    input_glob: Path,
) -> None:
    """Create one raw behavior profile per product without eligibility rules."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_product_profiles AS
        SELECT
            CASE WHEN is_fcm THEN 'FCM' ELSE 'Pseudo' END AS source_class,
            WGR_ID::INTEGER AS category_id,
            source_class || ' ' || category_id AS segment,
            ARTIKEL_ID,
            ARG_MAX(ARTIKEL_BEZ, CAST(DATE AS DATE)) AS ARTIKEL_BEZ,
            ARG_MAX(VERKAUFSEINHEIT, CAST(DATE AS DATE)) AS VERKAUFSEINHEIT,
            COUNT(DISTINCT MARKT_ID)::BIGINT AS stores,
            COUNT(*)::BIGINT AS active_store_days,
            COUNT_IF(COALESCE({DEMAND_COL}, 0) > 0)::BIGINT AS demand_store_days,
            SUM(COALESCE({DEMAND_COL}, 0))::DOUBLE AS total_demand_kg,
            COUNT_IF(COALESCE({DEMAND_COL}, 0) > 0)::DOUBLE / COUNT(*)
                AS demand_frequency,
            SUM(COALESCE({DEMAND_COL}, 0))::DOUBLE
                / NULLIF(COUNT_IF(COALESCE({DEMAND_COL}, 0) > 0), 0)
                AS kg_per_demand_day,
            SUM(COALESCE({DEMAND_COL}, 0))::DOUBLE / COUNT(*)
                AS kg_per_active_day,
            MIN(CAST(DATE AS DATE)) AS first_observed_date,
            MAX(CAST(DATE AS DATE)) FILTER (
                WHERE COALESCE({DEMAND_COL}, 0) > 0
            ) AS last_demand_date
        FROM {read_parquet_expr(input_glob)}
        WHERE is_active AND (is_fcm OR is_pseudo)
        GROUP BY source_class, category_id, ARTIKEL_ID
        """
    )


def audit_source(con: duckdb.DuckDBPyConnection, input_glob: Path) -> None:
    """Reject ambiguous flags, unexpected categories, and duplicate products."""
    source_audit = con.execute(
        f"""
        SELECT
            COUNT_IF(ARTIKEL_ID IS NULL OR MARKT_ID IS NULL OR DATE IS NULL),
            COUNT_IF(is_fcm IS NULL OR is_pseudo IS NULL),
            COUNT_IF(is_fcm AND is_pseudo),
            COUNT_IF(NOT is_fcm AND NOT is_pseudo),
            COUNT_IF(WGR_ID IS NULL OR WGR_ID::INTEGER NOT IN (890, 900)),
            (SELECT COUNT(*) FROM (
                SELECT ARTIKEL_ID, MARKT_ID, DATE
                FROM {read_parquet_expr(input_glob)}
                GROUP BY ALL
                HAVING COUNT(*) > 1
            ))
        FROM {read_parquet_expr(input_glob)}
        """
    ).fetchone()
    if any(source_audit):
        raise RuntimeError(
            "Sparse-region source audit failed: null keys/flags, overlapping or "
            "unclassified flags, unexpected categories, or duplicate series dates "
            "are present"
        )

    profile_audit = con.execute(
        """
        SELECT
            COUNT(*) - COUNT(DISTINCT ARTIKEL_ID),
            COUNT_IF(demand_store_days = 0),
            COUNT(DISTINCT segment)
        FROM raw_product_profiles
        """
    ).fetchone()
    duplicate_products, products_without_demand, segment_count = profile_audit
    if duplicate_products or products_without_demand or segment_count != len(SEGMENT_ORDER):
        raise RuntimeError(
            "Sparse-region product profiles are incomplete or not unique: "
            f"duplicate products={duplicate_products}, products without demand="
            f"{products_without_demand}, segments={segment_count}"
        )


def behavior_features(frame: pd.DataFrame) -> np.ndarray:
    """Return robustly scaled frequency and positive-day scale features."""
    frequency = frame["demand_frequency"].clip(1e-4, 1 - 1e-4)
    raw_features = np.column_stack(
        [
            np.log(frequency / (1 - frequency)),
            np.log1p(frame["kg_per_demand_day"]),
        ]
    )
    if not np.isfinite(raw_features).all():
        raise ValueError("Non-finite sparse-region features encountered")
    return RobustScaler().fit_transform(raw_features)


def k_diagnostics(
    frame: pd.DataFrame,
    scope: str,
    *,
    max_k: int = 6,
) -> pd.DataFrame:
    """Calculate compactness diagnostics for candidate cluster counts."""
    features = behavior_features(frame)
    rows: list[dict[str, object]] = []
    upper_k = min(max_k, len(frame) - 1)
    for candidate_k in range(2, upper_k + 1):
        labels = KMeans(
            n_clusters=candidate_k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        ).fit_predict(features)
        sizes = np.bincount(labels)
        rows.append(
            {
                "scope": scope,
                "k": candidate_k,
                "silhouette": silhouette_score(features, labels),
                "davies_bouldin": davies_bouldin_score(features, labels),
                "smallest_cluster": int(sizes.min()),
                "cluster_sizes": ", ".join(map(str, sorted(sizes))),
            }
        )
    return pd.DataFrame(rows)


def assign_k2_with_intensity_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit k=2 and name clusters by median kilograms per active store-day."""
    result = frame.copy()
    result["raw_cluster_id"] = KMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    ).fit_predict(behavior_features(result))
    ordered_ids = (
        result.groupby("raw_cluster_id")["kg_per_active_day"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    result["intensity_cluster"] = result["raw_cluster_id"].map(
        {ordered_ids[0]: "low_intensity", ordered_ids[1]: "high_intensity"}
    )
    return result.drop(columns="raw_cluster_id")


def apply_exclusion_flags(assignments: pd.DataFrame) -> pd.DataFrame:
    """Add explicit downstream exclusion flags without changing cluster regions."""
    result = assignments.copy()
    result["is_pseudo_890_sparse_scale_outlier"] = result["ARTIKEL_ID"].isin(
        PSEUDO_890_SPARSE_SCALE_OUTLIER_IDS
    )
    invalid_outliers = result.loc[
        result["is_pseudo_890_sparse_scale_outlier"]
        & ~result["segment"].eq("Pseudo 890"),
        ["ARTIKEL_ID", "segment"],
    ]
    if not invalid_outliers.empty:
        raise RuntimeError(
            "Configured Pseudo 890 sparse-scale outlier IDs belong to another segment: "
            f"{invalid_outliers.to_dict(orient='records')}"
        )

    result["exclusion_reason"] = result["region"].map(EXCLUSION_REASON_BY_REGION)
    result.loc[
        result["is_pseudo_890_sparse_scale_outlier"], "exclusion_reason"
    ] = "pseudo_890_sparse_scale_outlier"
    result["exclude_from_daily"] = result["exclusion_reason"].notna()
    return result


def warn_for_missing_sparse_scale_outliers(
    assignments: pd.DataFrame,
) -> set[int]:
    """Warn when configured outliers are absent without blocking discovery."""
    missing = PSEUDO_890_SPARSE_SCALE_OUTLIER_IDS.difference(
        assignments["ARTIKEL_ID"]
    )
    if missing:
        warnings.warn(
            "Configured Pseudo 890 sparse-scale product IDs are absent from "
            f"the current profiles and cannot be flagged: {sorted(missing)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return missing


def assign_regions(profiles: pd.DataFrame) -> pd.DataFrame:
    """Assign every product to exactly one of the seven requested regions."""
    assignments = pd.concat(
        [
            assign_k2_with_intensity_order(profiles[profiles["segment"] == segment])
            for segment in SEGMENT_ORDER
        ],
        ignore_index=True,
    )

    assignments["region"] = pd.NA
    assignments.loc[assignments["segment"].eq("FCM 890"), "region"] = "fcm_890"
    assignments.loc[
        assignments["segment"].eq("FCM 900"), "region"
    ] = assignments.loc[
        assignments["segment"].eq("FCM 900"), "intensity_cluster"
    ].map(
        {
            "low_intensity": "fcm_900_low_intensity",
            "high_intensity": "fcm_900_high_intensity",
        }
    )
    assignments.loc[
        assignments["segment"].eq("Pseudo 890"), "region"
    ] = assignments.loc[
        assignments["segment"].eq("Pseudo 890"), "intensity_cluster"
    ].map(
        {
            "low_intensity": "pseudo_890_low_velocity",
            "high_intensity": "pseudo_890_high_velocity",
        }
    )
    assignments.loc[
        assignments["segment"].eq("Pseudo 900"), "region"
    ] = assignments.loc[
        assignments["segment"].eq("Pseudo 900"), "intensity_cluster"
    ].map(
        {
            "low_intensity": "pseudo_900_low_intensity",
            "high_intensity": "pseudo_900_high_intensity",
        }
    )

    if assignments["region"].isna().any():
        missing = assignments.loc[assignments["region"].isna(), "segment"].unique()
        raise RuntimeError(f"Products without a sparse-region assignment: {missing}")

    for region, flag_column in REGION_FLAG_COLUMNS.items():
        assignments[flag_column] = assignments["region"].eq(region)
    assignments = apply_exclusion_flags(assignments)

    assignments["pseudo_890_low_velocity_subcluster"] = pd.NA
    low_velocity_mask = assignments["region"].eq("pseudo_890_low_velocity")
    low_velocity = assignments.loc[low_velocity_mask].copy()
    low_velocity["raw_subcluster_id"] = KMeans(
        n_clusters=4,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    ).fit_predict(behavior_features(low_velocity))
    ordered_subclusters = (
        low_velocity.groupby("raw_subcluster_id")["kg_per_active_day"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    subcluster_labels = {
        cluster_id: rank
        for rank, cluster_id in enumerate(ordered_subclusters, start=1)
    }
    assignments.loc[
        low_velocity_mask, "pseudo_890_low_velocity_subcluster"
    ] = low_velocity["raw_subcluster_id"].map(subcluster_labels).to_numpy()
    assignments["pseudo_890_low_velocity_subcluster"] = assignments[
        "pseudo_890_low_velocity_subcluster"
    ].astype("Int64")

    flag_count = assignments[list(REGION_FLAG_COLUMNS.values())].sum(axis=1)
    if not flag_count.eq(1).all() or not assignments["ARTIKEL_ID"].is_unique:
        raise RuntimeError("Sparse-region flags must be exclusive and product IDs unique")
    return assignments.sort_values(["segment", "region", "ARTIKEL_ID"]).reset_index(
        drop=True
    )


def discover_product_regions(
    con: duckdb.DuckDBPyConnection,
    input_glob: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build profiles, fit segment clusters, and return flags plus diagnostics."""
    create_product_profile_view(con, input_glob)
    audit_source(con, input_glob)
    profiles = con.execute(
        "SELECT * FROM raw_product_profiles ORDER BY segment, ARTIKEL_ID"
    ).fetchdf()
    assignments = assign_regions(profiles)
    warn_for_missing_sparse_scale_outliers(assignments)
    diagnostics = pd.concat(
        [
            k_diagnostics(profiles, "All products", max_k=8),
            *[
                k_diagnostics(
                    profiles[profiles["segment"] == segment],
                    segment,
                    max_k=6,
                )
                for segment in SEGMENT_ORDER
            ],
            k_diagnostics(
                assignments[
                    assignments["region"].eq("pseudo_890_low_velocity")
                ],
                "Pseudo 890 low velocity",
                max_k=6,
            ),
        ],
        ignore_index=True,
    )
    return assignments, diagnostics


def _write_frame(
    con: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    path: Path,
    relation_name: str,
) -> None:
    """Atomically write a DataFrame as compressed Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    con.register(relation_name, frame)
    con.execute(
        f"COPY {relation_name} TO {sql_literal(tmp_path)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    tmp_path.replace(path)


def materialize_region_views(
    con: duckdb.DuckDBPyConnection,
    assignments: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    flags_path: Path = REGION_FLAGS_PATH,
    diagnostics_path: Path = DIAGNOSTICS_PATH,
) -> None:
    """Persist product flags and cluster diagnostics for downstream consumers."""
    _write_frame(con, assignments, flags_path, "region_assignments_frame")
    _write_frame(con, diagnostics, diagnostics_path, "region_diagnostics_frame")


def print_region_summary(assignments: pd.DataFrame) -> None:
    """Print product counts and demand volume for each materialized region."""
    summary = (
        assignments.groupby("region", observed=True)
        .agg(
            products=("ARTIKEL_ID", "size"),
            median_frequency=("demand_frequency", "median"),
            median_kg_per_demand_day=("kg_per_demand_day", "median"),
            median_kg_per_active_day=("kg_per_active_day", "median"),
            total_demand_kg=("total_demand_kg", "sum"),
            excluded_products=("exclude_from_daily", "sum"),
        )
        .reset_index()
    )
    print("\nMaterialized sparse product regions")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.region}: {row.products:,} products, median frequency "
            f"{row.median_frequency:.1%}, median kg/positive-day "
            f"{row.median_kg_per_demand_day:.3f}, total "
            f"{row.total_demand_kg:,.1f} kg, excluded products="
            f"{row.excluded_products:,}"
        )


def materialize_sparse_regions(
    in_dir: Path = IN_DIR,
    *,
    flags_path: Path = REGION_FLAGS_PATH,
    diagnostics_path: Path = DIAGNOSTICS_PATH,
    threads: int = 8,
) -> None:
    """Discover and write the product-level sparse-region artifacts."""
    require_parquet_files(in_dir)
    input_glob = in_dir / "transactions_year_*.parquet"
    con = configure_duckdb(threads)
    assignments, diagnostics = discover_product_regions(con, input_glob)
    materialize_region_views(
        con,
        assignments,
        diagnostics,
        flags_path=flags_path,
        diagnostics_path=diagnostics_path,
    )
    print_region_summary(assignments)
    print(f"Wrote product flags to {flags_path}")
    print(f"Wrote clustering diagnostics to {diagnostics_path}")


def filtered_select_sql(path: Path, flags_path: Path = REGION_FLAGS_PATH) -> str:
    """Return the row-preserving selection excluding the two flagged regions."""
    return f"""
        SELECT t.*
        FROM {read_parquet_expr(path)} t
        INNER JOIN {read_parquet_expr(flags_path)} flags USING (ARTIKEL_ID)
        WHERE NOT flags.exclude_from_daily
    """


def materialize_filtered_transactions(
    in_dir: Path = IN_DIR,
    out_dir: Path = FILTERED_OUT_DIR,
    *,
    flags_path: Path = REGION_FLAGS_PATH,
    threads: int = 8,
) -> None:
    """Write daily transactions after product-region exclusions."""
    input_files = require_parquet_files(in_dir)
    if not flags_path.exists():
        raise FileNotFoundError(
            f"Sparse-region flags not found at {flags_path}; run discovery first"
        )
    con = configure_duckdb(threads)
    input_glob = in_dir / "transactions_year_*.parquet"
    missing_products = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT ARTIKEL_ID FROM {read_parquet_expr(input_glob)}
            EXCEPT
            SELECT ARTIKEL_ID FROM {read_parquet_expr(flags_path)}
        )
        """
    ).fetchone()[0]
    if missing_products:
        raise RuntimeError(
            f"Sparse-region flags are missing {missing_products:,} input products"
        )

    duplicate_flags = con.execute(
        f"""
        SELECT COUNT(*) - COUNT(DISTINCT ARTIKEL_ID)
        FROM {read_parquet_expr(flags_path)}
        """
    ).fetchone()[0]
    if duplicate_flags:
        raise RuntimeError("Sparse-region flag artifact contains duplicate product IDs")

    summary = con.execute(
        f"""
        SELECT
            flags.exclusion_reason,
            COUNT(DISTINCT t.ARTIKEL_ID) AS products,
            COUNT(DISTINCT (t.ARTIKEL_ID, t.MARKT_ID)) AS series,
            COUNT(*) AS rows,
            SUM(COALESCE(t.{DEMAND_COL}, 0)) AS demand_kg
        FROM {read_parquet_expr(input_glob)} t
        INNER JOIN {read_parquet_expr(flags_path)} flags USING (ARTIKEL_ID)
        WHERE flags.exclude_from_daily
        GROUP BY flags.exclusion_reason
        ORDER BY flags.exclusion_reason
        """
    ).fetchall()
    print("\nSparse-region transaction exclusions")
    for exclusion_reason, products, series, rows, demand_kg in summary:
        print(
            f"  {exclusion_reason}: {products:,} products, {series:,} series, "
            f"{rows:,} rows, {demand_kg:,.1f} kg"
        )

    clear_parquet_outputs(out_dir)
    total_input_rows = 0
    total_output_rows = 0
    for path in input_files:
        out_path = out_dir / path.name
        tmp_path = out_path.with_name(f".{out_path.stem}.filtered.tmp{out_path.suffix}")
        if tmp_path.exists():
            tmp_path.unlink()
        input_rows = con.execute(
            f"SELECT COUNT(*) FROM {read_parquet_expr(path)}"
        ).fetchone()[0]
        output_rows = con.execute(
            f"SELECT COUNT(*) FROM ({filtered_select_sql(path, flags_path)})"
        ).fetchone()[0]
        con.execute(
            f"COPY ({filtered_select_sql(path, flags_path)}) "
            f"TO {sql_literal(tmp_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        tmp_path.replace(out_path)
        total_input_rows += input_rows
        total_output_rows += output_rows
        print(f"Wrote {out_path.name}: {output_rows:,}/{input_rows:,} rows retained")

    leaked_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {read_parquet_expr(out_dir / 'transactions_year_*.parquet')} t
        INNER JOIN {read_parquet_expr(flags_path)} flags USING (ARTIKEL_ID)
        WHERE flags.exclude_from_daily
        """
    ).fetchone()[0]
    if leaked_rows:
        raise RuntimeError(f"Filtered output still contains {leaked_rows:,} excluded rows")
    print(f"Total rows retained: {total_output_rows:,}/{total_input_rows:,}")


def main() -> None:
    """Run discovery followed by the sparse-region transaction filter."""
    args = parse_args()
    materialize_sparse_regions(
        args.in_dir,
        flags_path=args.flags_path,
        diagnostics_path=args.diagnostics_path,
        threads=args.threads,
    )
    if not args.skip_filtered_transactions:
        materialize_filtered_transactions(
            args.in_dir,
            args.out_dir,
            flags_path=args.flags_path,
            threads=args.threads,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=FILTERED_OUT_DIR)
    parser.add_argument("--flags-path", type=Path, default=REGION_FLAGS_PATH)
    parser.add_argument("--diagnostics-path", type=Path, default=DIAGNOSTICS_PATH)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-filtered-transactions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
