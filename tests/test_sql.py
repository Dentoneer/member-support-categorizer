import json

import pytest

from support_categorizer.settings import Settings
from support_categorizer.sql import ai_query_sql, comparison_sql, response_schema, sql_str

FLOWS = ["account_access", "shipping_issue"]
SUBS = {"reset_2fa": "account_access", "cost": "shipping_issue"}


def test_sql_str_escapes_quotes_and_backslashes():
    assert sql_str("it's a \\ path") == "'it\\'s a \\\\ path'"


def test_schema_has_all_labels_and_required():
    sch = json.loads(response_schema(FLOWS, sorted(SUBS)))
    props = sch["json_schema"]["schema"]["properties"]
    assert props["secondary_category"]["enum"] == ["cost", "reset_2fa"]
    assert props["primary_category"]["enum"] == FLOWS
    assert set(sch["json_schema"]["schema"]["required"]) == set(props)


def test_ai_query_sql_structured():
    q = ai_query_sql(Settings(), "databricks-meta-llama-3-3-70b-instruct", FLOWS, SUBS)
    assert "ai_query('databricks-meta-llama-3-3-70b-instruct'" in q
    assert "responseFormat =>" in q and "failOnError => false" in q
    assert "workspace.support.calls_clean" in q and "workspace.support.eval_sample" in q
    assert "'reset_2fa', 'account_access'" in q  # sub -> flow map literal


def test_unstructured_variant_has_no_responseFormat():
    q = ai_query_sql(Settings(), "ep", FLOWS, SUBS, structured=False)
    assert "responseFormat" not in q and "JSON only" in q and "regexp_extract" in q


def test_comparison_sql_pivots_metrics():
    q = comparison_sql(Settings())
    assert "workspace.support.metrics" in q and "secondary_accuracy" in q


@pytest.mark.spark
def test_ai_query_sql_parses_in_spark(spark):
    parser = spark._jsparkSession.sessionState().sqlParser()
    for structured in (True, False):
        q = ai_query_sql(Settings(), "ep", FLOWS, {**SUBS, "it's": "shipping_issue"}, structured=structured)
        parser.parsePlan(q)  # raises ParseException on bad quoting
    parser.parsePlan(comparison_sql(Settings()))
