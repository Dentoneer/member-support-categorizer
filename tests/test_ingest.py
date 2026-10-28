import gzip
import json

import pytest

from support_categorizer.ingest import read_abcd_rows, subflow_map
from support_categorizer.transcript import build_transcript

LONG = [["agent", "Hi!"], ["action", "Account has been pulled up"], ["customer", "my  jeans"],
        ["agent", "ok"], ["customer", "thanks"]]  # fmt: skip


def _fixture(tmp_path):
    data = {
        "train": [
            {"convo_id": 1, "original": LONG,
             "scenario": {"flow": "single_item_query", "subflow": "jeans_how_4",
                          "personal": {"member_level": "gold"}}},
            {"convo_id": 2, "original": [["agent", "Hi"], ["customer", "x"]],
             "scenario": {"flow": "shipping_issue", "subflow": "cost",
                          "personal": {"member_level": "bronze"}}},
        ],
        "dev": [
            {"convo_id": 3, "original": LONG,
             "scenario": {"flow": "shipping_issue", "subflow": "cost", "personal": {}}},
            {"convo_id": 4, "original": LONG,
             "scenario": {"flow": "shipping_issue", "subflow": "weird_unknown",
                          "personal": {"member_level": "gold"}}},
        ],
        "test": [],
    }  # fmt: skip
    p = tmp_path / "abcd.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return str(p)


def test_read_rows_handles_missing_member_level(tmp_path):
    rows = {r[0]: r for r in read_abcd_rows(_fixture(tmp_path))}
    assert rows["3"][5] is None and rows["1"][5] == "gold" and rows["1"][1] == "train"
    assert rows["1"][2][1] == ("action", "Account has been pulled up")


def test_subflow_map():
    pairs = [("single_item_query", "jeans_how_4"), ("shipping_issue", "weird"), (None, "x")]
    m = subflow_map(pairs)
    assert m == {"single_item_query|jeans_how_4": "jeans"}


@pytest.mark.spark
def test_bronze_schema_and_counts(spark, tmp_path):
    from support_categorizer.ingest import bronze_from_json

    b = bronze_from_json(spark, _fixture(tmp_path))
    assert b.count() == 4
    assert set(b.columns) == {"call_id", "split", "turns", "flow", "subflow", "member_level",
                              "ingested_at"}  # fmt: skip


@pytest.mark.spark
def test_silver_drops_actions_short_calls_and_unknown_labels(spark, tmp_path):
    from support_categorizer.ingest import bronze_from_json, silver_from_bronze

    s = {r.call_id: r for r in silver_from_bronze(bronze_from_json(spark, _fixture(tmp_path)))
         .collect()}  # fmt: skip
    assert set(s) == {"1", "3"}  # 2 is too short, 4 has an unknown subflow
    assert s["1"].secondary_category == "jeans" and s["1"].primary_category == "single_item_query"
    assert s["3"].customer_tier == "unknown"
    assert "Account has been pulled up" not in s["1"].call_transcript


@pytest.mark.spark
def test_silver_matches_python_twin(spark, tmp_path):
    from support_categorizer.ingest import bronze_from_json, silver_from_bronze

    row = silver_from_bronze(bronze_from_json(spark, _fixture(tmp_path))).where("call_id = '1'")
    assert row.first().call_transcript == build_transcript([tuple(t) for t in LONG])
