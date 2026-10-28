import pandas as pd
import pytest

from support_categorizer.tables import accuracy_rows, make_eval_sample


def _test_split(n_per=(40, 30, 1)):
    rows = []
    for label, k in zip(["a", "b", "c"], n_per, strict=True):
        rows += [{"call_id": f"{label}{i:03d}", "secondary_category": label} for i in range(k)]
    return pd.DataFrame(rows)


def test_eval_sample_deterministic_and_stratified():
    df = _test_split()
    s1, s2 = make_eval_sample(df, n=20, seed=7), make_eval_sample(df, n=20, seed=7)
    assert list(s1["call_id"]) == list(s2["call_id"]) and len(s1) == 20
    assert set(s1["secondary_category"]) == {"a", "b", "c"}  # rare class kept


def test_eval_sample_small_split_returns_all():
    assert len(make_eval_sample(_test_split((2, 1, 1)), n=300)) == 4


def test_accuracy_counts_invalid_and_missing_as_wrong():
    truth = pd.DataFrame({"call_id": ["1", "2", "3", "4"], "primary_category": ["p"] * 4,
                          "secondary_category": ["s"] * 4})  # fmt: skip
    preds = pd.DataFrame({"call_id": ["1", "2", "3"], "primary_category": ["p", "p", "p"],
                          "secondary_category": ["s", "s", "x"], "valid": [True, False, True]})  # fmt: skip
    m = {r["metric"]: r["value"] for r in accuracy_rows(preds, truth, "m")}
    assert m["secondary_accuracy"] == 0.25 and m["primary_accuracy"] == 0.5
    assert m["valid_rate"] == 0.5


@pytest.mark.spark
def test_merge_into_is_idempotent(spark):
    from support_categorizer.tables import merge_into

    spark.sql("CREATE TABLE IF NOT EXISTS t_merge (call_id STRING, model STRING, v INT) USING DELTA")
    df = spark.createDataFrame([("1", "m", 1), ("2", "m", 2), ("3", "m", 3)], "call_id string, model string, v int")
    merge_into(spark, df, "t_merge", ["call_id", "model"])
    merge_into(spark, df, "t_merge", ["call_id", "model"])
    assert spark.table("t_merge").count() == 3
    merge_into(spark, spark.createDataFrame([("2", "m", 20)], df.schema), "t_merge", ["call_id", "model"])
    assert spark.sql("SELECT v FROM t_merge WHERE call_id = '2'").first().v == 20


@pytest.mark.spark
def test_write_predictions_accepts_null_urgency(spark):
    from support_categorizer.settings import Settings
    from support_categorizer.tables import DDL, write_predictions

    spark.sql("CREATE DATABASE IF NOT EXISTS sup")
    s = Settings(catalog="spark_catalog", schema="sup")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {s.fqn('predictions')} ({DDL['predictions'][0]}) USING DELTA")
    row = {"call_id": "1", "model": "baseline", "model_version": "v", "run_id": "r",
           "primary_category": "p", "secondary_category": "s", "urgency_score": None,
           "concise_summary": None, "valid": True}  # fmt: skip
    write_predictions(spark, s, [row])
    write_predictions(spark, s, [row])
    assert spark.table(s.fqn("predictions")).count() == 1


@pytest.mark.spark
def test_write_validations_nulls_and_rerun(spark):
    from support_categorizer.settings import Settings
    from support_categorizer.tables import DDL, write_validations

    spark.sql("CREATE DATABASE IF NOT EXISTS sup")
    s = Settings(catalog="spark_catalog", schema="sup")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {s.fqn('validations')} ({DDL['validations'][0]}) USING DELTA")
    rows = [{"call_id": "1", "model": "m", "primary_ok": 0.9, "secondary_ok": None, "summary_faithful": 1.0}]
    write_validations(spark, s, rows)
    write_validations(spark, s, rows)
    assert spark.table(s.fqn("validations")).count() == 1
