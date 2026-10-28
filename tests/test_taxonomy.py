import pandas as pd
import pytest

from support_categorizer.taxonomy import (
    active_labels,
    apply_request,
    canonical,
    seed_rows,
    validate_request,
)


def test_seed_has_55_active():
    flows, subs, m = active_labels(seed_rows())
    assert len(subs) == 55 and len(flows) == 10 and m["reset_2fa"] == "account_access"


def test_validate_rejects_duplicate_add_and_unknown_retire():
    t = seed_rows()
    dup = {"kind": "add", "primary_category": "shipping_issue", "secondary_category": "cost",
           "description": "d", "reason": "r"}  # fmt: skip
    assert any("already exists" in p for p in validate_request(dup, t))
    bad = {"kind": "retire", "secondary_category": "nope", "reason": "r"}
    assert any("not an active" in p for p in validate_request(bad, t))
    assert any("reason" in p for p in validate_request({**bad, "reason": ""}, t))


def test_add_needs_existing_primary_unless_flagged():
    t = seed_rows()
    req = {"kind": "add", "primary_category": "billing", "secondary_category": "late_fee",
           "description": "late fees", "reason": "new policy"}  # fmt: skip
    assert validate_request(req, t)
    assert validate_request({**req, "new_primary": True}, t) == []


def test_apply_rename_keeps_alias_and_new_version():
    t = seed_rows()
    v2 = apply_request({"kind": "rename", "secondary_category": "cost", "new_name": "shipping_cost",
                        "reason": "clearer"}, t, "marco")  # fmt: skip
    assert (v2["version"] == 2).all() and len(v2) == 55
    row = v2[v2["secondary_category"] == "shipping_cost"].iloc[0]
    assert row["renamed_from"] == "cost" and row["changed_by"] == "marco"


def test_apply_retire_is_soft():
    t = seed_rows()
    v2 = apply_request({"kind": "retire", "secondary_category": "cost", "reason": "unused"}, t, "m")
    assert v2.loc[v2["secondary_category"] == "cost", "status"].item() == "retired"
    assert "cost" not in active_labels(v2)[1]


def test_apply_rejects_invalid():
    with pytest.raises(ValueError, match="not an active"):
        apply_request({"kind": "retire", "secondary_category": "x", "reason": "r"}, seed_rows(), "m")


def test_canonical_follows_chain():
    t = seed_rows()
    v2 = apply_request({"kind": "rename", "secondary_category": "cost", "new_name": "b",
                        "reason": "r"}, t, "m")  # fmt: skip
    all_v = pd.concat([t, v2])
    v3 = apply_request({"kind": "rename", "secondary_category": "b", "new_name": "c",
                        "reason": "r"}, all_v, "m")  # fmt: skip
    all_v = pd.concat([all_v, v3])
    assert canonical("cost", all_v) == "c" and canonical("reset_2fa", all_v) == "reset_2fa"
    assert canonical(None, all_v) is None


def test_canonical_truth_maps_renames_and_drops_retired():
    from support_categorizer.taxonomy import canonical_truth

    t = seed_rows()
    v2 = apply_request({"kind": "rename", "secondary_category": "cost", "new_name": "shipping_cost",
                        "reason": "r"}, t, "m")
    t = pd.concat([t, v2])
    t = pd.concat([t, apply_request({"kind": "retire", "secondary_category": "missing", "reason": "r"}, t, "m")])
    truth = pd.DataFrame({"call_id": ["1", "2", "3"], "primary_category": ["shipping_issue"] * 3,
                          "secondary_category": ["cost", "missing", "status"]})
    out = canonical_truth(truth, t)
    assert list(out["call_id"]) == ["1", "3"]
    assert list(out["secondary_category"]) == ["shipping_cost", "status"]
    assert list(out["primary_category"]) == ["shipping_issue", "shipping_issue"]
