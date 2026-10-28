import pytest

from support_categorizer.labels import FLOWS, ONTOLOGY, SUBFLOWS, flow_of, normalize_subflow


def test_counts():
    assert len(SUBFLOWS) == 55 and len(FLOWS) == 10
    assert SUBFLOWS == sorted(SUBFLOWS) and FLOWS == sorted(ONTOLOGY)


@pytest.mark.parametrize(
    "flow, raw, want",
    [
        ("single_item_query", "jeans_how_4", "jeans"),
        ("single_item_query", "boots_other_1", "boots"),
        ("storewide_query", "pricing_3", "pricing"),
        ("order_issue", "status_delivery_date", "status_delivery_time"),
        ("subscription_inquiry", "status_questions", "status_active"),
        ("shipping_issue", "cost", "cost"),
        ("shipping_issue", "nonsense_label", None),
        ("unknown_flow", "cost", None),
    ],
)
def test_normalize_subflow(flow, raw, want):
    assert normalize_subflow(flow, raw) == want


def test_flow_of():
    assert flow_of("reset_2fa") == "account_access"
    with pytest.raises(KeyError):
        flow_of("nope")
