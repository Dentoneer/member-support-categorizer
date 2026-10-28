"""Gold tables: create, idempotent MERGE, the fixed evaluation sample, accuracy rows."""

from __future__ import annotations

import pandas as pd

from support_categorizer.settings import (
    APP_ADMINS,
    EVAL_N,
    EVAL_SEED,
    METRICS,
    OVERRIDES,
    PREDICTIONS,
    TAXONOMY,
    TAXONOMY_REQUESTS,
    VALIDATIONS,
    Settings,
)

PRED_COLS = [
    "call_id", "model", "model_version", "run_id", "primary_category", "secondary_category",
    "urgency_score", "concise_summary", "valid", "predicted_at",
]  # fmt: skip
VALIDATION_COLS = ["call_id", "model", "primary_ok", "secondary_ok", "summary_faithful",
                   "validated_at"]  # fmt: skip
OVERRIDE_COLS = ["call_id", "model", "secondary_category", "primary_category", "reviewer", "note",
                 "overridden_at"]  # fmt: skip

DDL = {
    PREDICTIONS: (
        "call_id STRING, model STRING, model_version STRING, run_id STRING, "
        "primary_category STRING, secondary_category STRING, urgency_score INT, "
        "concise_summary STRING, valid BOOLEAN, predicted_at TIMESTAMP",
        "Gold: one prediction per (call_id, model).",
    ),
    VALIDATIONS: (
        "call_id STRING, model STRING, primary_ok DOUBLE, secondary_ok DOUBLE, "
        "summary_faithful DOUBLE, validated_at TIMESTAMP",
        "Gold: Jev probabilities that each prediction is right (null = not validated).",
    ),
    METRICS: (
        "model STRING, metric STRING, value DOUBLE, n BIGINT, computed_at TIMESTAMP",
        "Gold: accuracy and other metrics per model on the fixed eval sample.",
    ),
    OVERRIDES: (
        "call_id STRING, model STRING, secondary_category STRING, primary_category STRING, "
        "reviewer STRING, note STRING, overridden_at TIMESTAMP",
        "Gold: supervisor corrections from the review app (latest per call wins).",
    ),
    TAXONOMY: (
        "version INT, primary_category STRING, secondary_category STRING, description STRING, "
        "status STRING, renamed_from STRING, changed_by STRING, changed_at TIMESTAMP",
        "Gold: versioned category list; each approved change writes a full new version.",
    ),
    TAXONOMY_REQUESTS: (
        "request_id STRING, kind STRING, primary_category STRING, secondary_category STRING, "
        "new_name STRING, description STRING, reason STRING, example_call_ids ARRAY<STRING>, "
        "requested_by STRING, requested_at TIMESTAMP, status STRING, decided_by STRING, "
        "decided_at TIMESTAMP, decision_note STRING",
        "Gold: requests to add, rename or retire a category, and their approval.",
    ),
    APP_ADMINS: ("email STRING", "Who may approve category changes in the review app."),
}


def ensure_gold_tables(spark, s: Settings) -> None:
    for name, (cols, comment) in DDL.items():
        spark.sql(f"CREATE TABLE IF NOT EXISTS {s.fqn(name)} ({cols}) COMMENT '{comment}'")


def merge_into(spark, df, table: str, keys: list[str]) -> None:
    """Delta MERGE: update every column on a key match, insert otherwise (rerun-safe)."""
    from delta.tables import DeltaTable

    on = " AND ".join(f"t.`{k}` = s.`{k}`" for k in keys)
    (
        DeltaTable.forName(spark, table)
        .alias("t")
        .merge(df.alias("s"), on)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def make_eval_sample(
    test_pdf: pd.DataFrame, n: int = EVAL_N, seed: int = EVAL_SEED
) -> pd.DataFrame:
    """Stratified (by secondary_category), seeded sample of the test split; >= 1 per class."""
    df = test_pdf[["call_id", "secondary_category"]].sort_values("call_id").reset_index(drop=True)
    if len(df) <= n:
        return df
    counts = df["secondary_category"].value_counts()
    quota = (counts / len(df) * n).round().clip(lower=1).astype(int)
    while quota.sum() > n:  # trim the largest classes first
        quota[quota.idxmax()] -= 1
    while quota.sum() < n:  # top up where the most calls are left unsampled
        quota[(counts - quota).idxmax()] += 1
    parts = [
        g.sample(n=min(len(g), quota[label]), random_state=seed)
        for label, g in df.groupby("secondary_category", sort=True)
    ]
    return pd.concat(parts).sort_values("call_id").reset_index(drop=True)


def accuracy_rows(preds: pd.DataFrame, truth: pd.DataFrame, model: str) -> list[dict]:
    """Secondary/primary accuracy and valid rate over `truth` rows; missing/invalid = wrong."""
    m = truth[["call_id", "primary_category", "secondary_category"]].merge(
        preds[["call_id", "primary_category", "secondary_category", "valid"]],
        on="call_id", how="left", suffixes=("_true", "_pred"),
    )  # fmt: skip
    valid = m["valid"].fillna(False).astype(bool)
    n = len(m)
    sec = (valid & (m["secondary_category_true"] == m["secondary_category_pred"])).sum()
    pri = (valid & (m["primary_category_true"] == m["primary_category_pred"])).sum()
    return [
        {"model": model, "metric": "secondary_accuracy", "value": sec / n if n else 0.0, "n": n},
        {"model": model, "metric": "primary_accuracy", "value": pri / n if n else 0.0, "n": n},
        {"model": model, "metric": "valid_rate", "value": valid.mean() if n else 0.0, "n": n},
    ]


def write_metrics(spark, s: Settings, rows: list[dict]) -> None:
    """Replace this model's metric rows (rerun-safe)."""
    from pyspark.sql import functions as F

    df = spark.createDataFrame(pd.DataFrame(rows)).withColumn("computed_at", F.current_timestamp())
    merge_into(spark, df.select("model", "metric", "value", "n", "computed_at"), s.fqn(METRICS),
               ["model", "metric"])  # fmt: skip


def write_predictions(spark, s: Settings, rows: list[dict]) -> None:
    from pyspark.sql import functions as F

    cols = [c for c in PRED_COLS if c != "predicted_at"]
    data = [tuple(r.get(c) for c in cols) for r in rows]  # tuples keep None ints as null
    df = spark.createDataFrame(data, schema=DDL[PREDICTIONS][0].rsplit(",", 1)[0]).withColumn(
        "predicted_at", F.current_timestamp()
    )
    merge_into(spark, df, s.fqn(PREDICTIONS), ["call_id", "model"])


def write_validations(spark, s: Settings, rows: list[dict]) -> None:
    from pyspark.sql import functions as F

    cols = [c for c in VALIDATION_COLS if c != "validated_at"]
    data = [tuple(r.get(c) for c in cols) for r in rows]
    df = spark.createDataFrame(data, schema=DDL[VALIDATIONS][0].rsplit(",", 1)[0]).withColumn(
        "validated_at", F.current_timestamp()
    )
    merge_into(spark, df, s.fqn(VALIDATIONS), ["call_id", "model"])
