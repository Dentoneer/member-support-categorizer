from pathlib import Path

import pandas as pd

from support_categorizer.guidelines import guideline_rows, load_guidelines_file
from support_categorizer.labels import SUBFLOWS
from support_categorizer.taxonomy import apply_request, seed_rows

FIX = str(Path(__file__).parent / "fixtures" / "abcd_guidelines_min.json")  # intent sentences from ABCD (MIT)


def test_every_ontology_label_has_a_guideline():
    g = load_guidelines_file(FIX)
    assert sorted(g) == SUBFLOWS
    assert "shipment" in g["status"].lower() and g["status"].startswith("Shipping Issue > Shipping Status")
    assert "international" in g["status_shipping_question"].lower()


def test_rows_follow_renames_use_descriptions_for_new_and_skip_retired():
    g = load_guidelines_file(FIX)
    tax = seed_rows()
    tax = pd.concat([tax, apply_request({"kind": "rename", "secondary_category": "cost",
                                         "new_name": "shipping_cost", "reason": "r"}, tax, "m")])
    tax = pd.concat([tax, apply_request({"kind": "retire", "secondary_category": "missing", "reason": "r"}, tax, "m")])
    tax = pd.concat([tax, apply_request({"kind": "add", "primary_category": "shipping_issue",
                                         "secondary_category": "late_fee", "description": "Customer disputes a late fee.",
                                         "reason": "r"}, tax, "m")])
    rows = guideline_rows(tax, g).set_index("secondary_category")
    assert len(rows) == 55  # 55 - retired + added
    assert rows.loc["shipping_cost", "guideline_text"] == g["cost"] and rows.loc["shipping_cost", "source"] == "abcd"
    assert rows.loc["late_fee", "guideline_text"] == "Customer disputes a late fee."
    assert rows.loc["late_fee", "source"] == "taxonomy"
    assert "missing" not in rows.index
