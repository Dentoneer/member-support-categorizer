import pandas as pd
import pytest

from support_categorizer.review import primary_volumes, volumes
from support_categorizer.settings import Settings
from support_categorizer.sql import (
    is_admin_sql,
    override_insert_sql,
    predicted_counts_sql,
    request_insert_sql,
    review_query_sql,
    taxonomy_insert_sql,
)
from support_categorizer.taxonomy import apply_request, seed_rows


def test_override_insert_uses_named_params():
    q = override_insert_sql(Settings())
    for p in (":call_id", ":model", ":secondary_category", ":primary_category", ":reviewer", ":note"):
        assert p in q
    assert "workspace.support.overrides" in q


def test_review_query_joins_latest_override_and_jev():
    q = review_query_sql(Settings())
    assert "row_number() OVER (PARTITION BY call_id, model ORDER BY overridden_at DESC) = 1" in q
    assert "workspace.support.validations" in q and "WHERE p.model = :model" in q


def test_predicted_counts_prefers_override():
    assert "coalesce(o.secondary_category, p.secondary_category)" in predicted_counts_sql(Settings())


def test_request_insert_and_admin_sql():
    assert "split(:example_call_ids, ',')" in request_insert_sql(Settings())
    assert "workspace.support.app_admins" in is_admin_sql(Settings())


def test_taxonomy_insert_params_and_nan_to_null():
    rows = seed_rows().head(2).to_dict("records")
    rows[0]["renamed_from"] = float("nan")
    q, params = taxonomy_insert_sql(Settings(), rows)
    assert q.count("current_timestamp()") == 2 and params["r0_renamed_from"] is None
    assert params["r1_secondary_category"] == rows[1]["secondary_category"]
    assert "CAST(:r0_version AS INT)" in q


def test_volumes_maps_renames_and_keeps_retired():
    tax = seed_rows()
    v2 = apply_request({"kind": "rename", "secondary_category": "cost", "new_name": "shipping_cost",
                        "reason": "r"}, tax, "m")
    tax = pd.concat([tax, v2])
    v3 = apply_request({"kind": "retire", "secondary_category": "missing", "reason": "r"}, tax, "m")
    tax = pd.concat([tax, v3])
    true_counts = pd.DataFrame({"secondary_category": ["cost", "missing", "reset_2fa"], "n": [10, 4, 6]})
    pred_counts = pd.DataFrame({"secondary_category": ["cost", "shipping_cost", "reset_2fa"], "n": [3, 2, 5]})
    vol = volumes(true_counts, pred_counts, tax).set_index("secondary_category")
    assert vol.loc["shipping_cost", "calls_all"] == 10 and vol.loc["shipping_cost", "calls_predicted"] == 5
    assert vol.loc["missing", "status"] == "retired" and vol.loc["missing", "calls_all"] == 4
    assert vol["share_predicted"].sum() == pytest.approx(1.0, abs=0.01)
    prim = primary_volumes(vol.reset_index()).set_index("primary_category")
    assert prim.loc["shipping_issue", "calls_all"] == 14
