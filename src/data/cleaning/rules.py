"""Shared transaction cleaning rules."""
from __future__ import annotations

from pathlib import Path

from src.data.common import ident, read_parquet_expr, sql_literal

ARTICLE_ID_COL = "ARTIKEL_ID"
MANDANT_ID_COL = "MANDANT_ID"
UMS_MENGE_COL = "UMS_MENGE"
ARTIKEL_INHALT_COL = "ARTIKEL_INHALT"
GEWICHT_FLAG_COL = "GEWICHT_FLAG"

MANDANT_RULE = True
FCM_RULE = True
WEIGHT_RULE = True

MIN_UMS_MENGE = 0.01
WEIGHT_CONTENT_LIKE = "%amm%"

ALLOWED_MANDANT_IDS = {110, 130, 135}
ALLOWED_FCM_ARTICLE_IDS = {
    560039,
    579781,
    1376569,
    1376691,
    1376698,
    1376854,
    1376855,
    1376860,
    1376862,
    1376875,
    1376880,
    1376881,
    1377267,
    1379003,
    1379020,
    1382746,
    1382748,
    1382752,
    1382760,
    1382761,
    1382762,
    1382763,
    1382764,
    1382765,
    1382766,
    1382767,
    1382768,
    1382770,
    1382771,
    1382774,
    1382777,
    1382780,
    1382782,
    1382788,
    1382858,
    1382862,
    1382863,
    1389566,
    1389568,
    1389569,
    1394487,
    1394488,
    1394489,
    1394490,
    1396337,
    1399512,
    1401346,
    1401486,
    1401487,
    1401488,
    1401489,
    1401490,
    1402017,
    1402822,
    1402844,
    1403340,
    1405612,
    1406063,
    1406252,
    1406310,
    1406315,
    1406317,
    1406587,
    1406590,
    1406593,
    1406595,
    1406603,
    1406792,
    1406794,
    1406796,
    1407213,
    1407289,
    1407292,
    1407293,
    1407295,
    1407886,
    1408116,
    1409377,
    1411185,
    1411695,
    1412026,
    1413118,
    1413120,
    1413121,
    1413122,
    1413150,
    1413151,
    1413521,
    1414244,
    1416733,
    1422555,
    1422634,
    1425414,
    1425927,
    1426349,
    1426594,
    1428223,
    1430131,
    1433364,
    1433365,
    1433578,
    1433967,
    1434898,
}

DUPLICATE_KEY_COLS = ["ARTIKEL_ID", "MARKT_ID", "BON_ID", "DATE", "TIME", "UMS_MENGE"]
DROP_COLUMNS = {"EAN_ID"}
BINARY_FLAG_COLUMNS = {
    "GEWICHT_FLAG",
    "GEWICHTSARTIKEL",
    "WAAGENARTIKEL",
    "AKTION_KENNZEICHEN",
    "PREISUEBERSCHREIBUNG",
    "BONABBRUCH",
    "STORNOART",
    "STORNOZEILE",
    "NEGATIVARTIKEL",
}


def allowed_mandant_ids_sql() -> str:
    return ", ".join(str(mandant_id) for mandant_id in sorted(ALLOWED_MANDANT_IDS))


def allowed_fcm_article_ids_sql() -> str:
    return ", ".join(str(article_id) for article_id in sorted(ALLOWED_FCM_ARTICLE_IDS))


def mandant_filter_condition(alias: str | None = None) -> str:
    col = ident(MANDANT_ID_COL)
    if alias:
        col = f"{alias}.{col}"
    return f"{col} IN ({allowed_mandant_ids_sql()})"


def fcm_filter_condition(alias: str | None = None) -> str:
    col = ident(ARTICLE_ID_COL)
    if alias:
        col = f"{alias}.{col}"
    return f"{col} IN ({allowed_fcm_article_ids_sql()})"


def ums_menge_filter_condition(alias: str | None = None) -> str:
    col = ident(UMS_MENGE_COL)
    if alias:
        col = f"{alias}.{col}"
    return f"{col} > {MIN_UMS_MENGE}"


def gewicht_flag_expression_sql(alias: str | None = None) -> str:
    col = ident(ARTIKEL_INHALT_COL)
    if alias:
        col = f"{alias}.{col}"
    content = f"LOWER(COALESCE(CAST({col} AS VARCHAR), ''))"
    return (
        f"CASE WHEN {content} LIKE {sql_literal(WEIGHT_CONTENT_LIKE)} "
        "THEN 1 ELSE 0 END"
    )


def weight_filter_condition(alias: str | None = None) -> str:
    return f"({gewicht_flag_expression_sql(alias)}) = 1"


def active_rule_flags() -> tuple[bool, bool, bool]:
    return bool(MANDANT_RULE), bool(FCM_RULE), bool(WEIGHT_RULE)


def transaction_filter_condition(alias: str | None = None) -> str:
    mandant_rule, fcm_rule, weight_rule = active_rule_flags()
    conditions = [f"({ums_menge_filter_condition(alias)})"]
    if mandant_rule:
        conditions.append(f"({mandant_filter_condition(alias)})")
    if fcm_rule:
        conditions.append(f"({fcm_filter_condition(alias)})")
    if weight_rule:
        conditions.append(f"({weight_filter_condition(alias)})")
    return " AND ".join(conditions)


def duplicate_keys_sql(alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    return ", ".join(f"{prefix}{ident(col)}" for col in DUPLICATE_KEY_COLS)


def filtered_transactions_expr(path: Path | str, *, filename: bool = False) -> str:
    return f"""
        (
            SELECT *
            FROM {read_parquet_expr(path, filename=filename)}
            WHERE {transaction_filter_condition()}
        )
        """
