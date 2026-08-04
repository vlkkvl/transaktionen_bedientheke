from __future__ import annotations

import unittest

import duckdb

from src.data.cleaning.aggregate_daily import (
    FIRST_COLS,
    FLAG_COLS,
    REQUIRED_COLS,
    SUM_COLS,
    aggregate_select_sql,
)
from src.data.cleaning.define_goal_variable import output_select_sql
from src.data.cleaning.rules import (
    ALLOWED_FCM_ARTICLE_IDS,
    PSEUDO_ARTICLE_IDS,
    fcm_or_pseudo_filter_condition,
)


class ProductTypeFlagTest(unittest.TestCase):
    @staticmethod
    def pseudo_only_article_id() -> int:
        return next(iter(PSEUDO_ARTICLE_IDS - ALLOWED_FCM_ARTICLE_IDS))

    def test_goal_variable_stage_preserves_existing_product_flags(self) -> None:
        columns = [
            "ARTIKEL_ID",
            "GEWICHTSARTIKEL",
            "ARTIKEL_INHALT",
            "GRAMM_BON",
            "UMS_MENGE",
            "is_fcm",
            "is_pseudo",
        ]
        article_id = self.pseudo_only_article_id()
        result = duckdb.connect().execute(
            f"""
            SELECT {output_select_sql(columns)}
            FROM (
                SELECT
                    {article_id}::BIGINT AS ARTIKEL_ID,
                    0::TINYINT AS GEWICHTSARTIKEL,
                    '100 gramm'::VARCHAR AS ARTIKEL_INHALT,
                    0.1::DOUBLE AS GRAMM_BON,
                    1.0::DOUBLE AS UMS_MENGE,
                    TRUE::BOOLEAN AS is_fcm,
                    FALSE::BOOLEAN AS is_pseudo
            )
            """
        ).fetchone()

        self.assertIs(result[5], True)
        self.assertIs(result[6], False)

    def test_filter_uses_existing_flags_as_source_of_truth(self) -> None:
        article_id = self.pseudo_only_article_id()
        condition = fcm_or_pseudo_filter_condition(
            columns=["ARTIKEL_ID", "is_fcm", "is_pseudo"]
        )
        kept = duckdb.connect().execute(
            f"""
            SELECT {condition}
            FROM (
                SELECT
                    {article_id}::BIGINT AS ARTIKEL_ID,
                    FALSE::BOOLEAN AS is_fcm,
                    FALSE::BOOLEAN AS is_pseudo
            )
            """
        ).fetchone()[0]

        self.assertIs(kept, False)

    def test_goal_variable_stage_assigns_only_a_missing_flag(self) -> None:
        columns = [
            "ARTIKEL_ID",
            "GEWICHTSARTIKEL",
            "ARTIKEL_INHALT",
            "GRAMM_BON",
            "UMS_MENGE",
            "is_fcm",
        ]
        article_id = self.pseudo_only_article_id()
        cursor = duckdb.connect().execute(
            f"""
            SELECT {output_select_sql(columns)}
            FROM (
                SELECT
                    {article_id}::BIGINT AS ARTIKEL_ID,
                    0::TINYINT AS GEWICHTSARTIKEL,
                    '100 gramm'::VARCHAR AS ARTIKEL_INHALT,
                    0.1::DOUBLE AS GRAMM_BON,
                    1.0::DOUBLE AS UMS_MENGE,
                    TRUE::BOOLEAN AS is_fcm
            )
            """
        )
        row = dict(zip((item[0] for item in cursor.description), cursor.fetchone()))

        self.assertIs(row["is_fcm"], True)
        self.assertIs(row["is_pseudo"], True)

    def test_goal_variable_stage_canonicalizes_uppercase_product_flags(self) -> None:
        columns = [
            "ARTIKEL_ID",
            "GEWICHTSARTIKEL",
            "ARTIKEL_INHALT",
            "GRAMM_BON",
            "UMS_MENGE",
            "IS_FCM",
            "IS_PSEUDO",
        ]
        cursor = duckdb.connect().execute(
            f"""
            SELECT {output_select_sql(columns)}
            FROM (
                SELECT
                    1::BIGINT AS ARTIKEL_ID,
                    0::TINYINT AS GEWICHTSARTIKEL,
                    '100 gramm'::VARCHAR AS ARTIKEL_INHALT,
                    0.1::DOUBLE AS GRAMM_BON,
                    1.0::DOUBLE AS UMS_MENGE,
                    TRUE::BOOLEAN AS IS_FCM,
                    FALSE::BOOLEAN AS IS_PSEUDO
            )
            """
        )
        names = [item[0] for item in cursor.description]
        row = dict(zip(names, cursor.fetchone()))

        self.assertIn("is_fcm", names)
        self.assertIn("is_pseudo", names)
        self.assertNotIn("IS_FCM", names)
        self.assertNotIn("IS_PSEUDO", names)
        self.assertIs(row["is_fcm"], True)
        self.assertIs(row["is_pseudo"], False)

    def test_daily_aggregation_uses_preserved_product_flags(self) -> None:
        article_id = self.pseudo_only_article_id()
        columns = sorted(set(REQUIRED_COLS + ["is_fcm", "is_pseudo"]))
        select_parts = []
        for col in columns:
            if col == "ARTIKEL_ID":
                expression = f"{article_id}::BIGINT"
            elif col == "MARKT_ID":
                expression = "1::BIGINT"
            elif col == "BON_ID":
                expression = "i::BIGINT"
            elif col == "DATE":
                expression = "DATE '2024-01-01'"
            elif col == "TIME":
                expression = "CAST(i AS VARCHAR)"
            elif col in SUM_COLS:
                expression = "1.0::DOUBLE"
            elif col in FLAG_COLS:
                expression = "i::TINYINT"
            elif col in {"GEWICHT_FLAG", "GEWICHTSARTIKEL"}:
                expression = "0::TINYINT"
            elif col in {"MARKT_NR", "MANDANT_ID", "WGR_ID"}:
                expression = "1::BIGINT"
            elif col == "is_fcm":
                expression = "TRUE::BOOLEAN"
            elif col == "is_pseudo":
                expression = "FALSE::BOOLEAN"
            elif col in FIRST_COLS:
                expression = "'value'::VARCHAR"
            else:
                raise AssertionError(f"Test schema does not define {col}")
            select_parts.append(f'{expression} AS "{col}"')

        con = duckdb.connect()
        con.execute(
            f"""
            CREATE TEMP TABLE input_rows AS
            SELECT {", ".join(select_parts)}
            FROM range(2) AS rows(i)
            """
        )
        cursor = con.execute(aggregate_select_sql(columns, "input_rows"))
        output = dict(zip((item[0] for item in cursor.description), cursor.fetchone()))

        self.assertIs(output["is_fcm"], True)
        self.assertIs(output["is_pseudo"], False)
        self.assertEqual(output["AKTION_KENNZEICHEN"], 1)
        self.assertEqual(output["RABATT"], 1)
        self.assertEqual(output["ARTIKELRABATT"], 1)


if __name__ == "__main__":
    unittest.main()
