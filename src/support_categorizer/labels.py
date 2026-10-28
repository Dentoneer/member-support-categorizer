"""ABCD label ontology (seed of taxonomy version 1) and raw-subflow normalization."""

from __future__ import annotations

import re

# Verbatim from ABCD data/ontology.json intents.subflows (10 flows, 55 subflows).
ONTOLOGY: dict[str, list[str]] = {
    "account_access": ["recover_username", "recover_password", "reset_2fa"],
    "manage_account": [
        "status_service_added", "status_service_removed", "status_shipping_question",
        "status_credit_missing", "manage_change_address", "manage_change_name",
        "manage_change_phone", "manage_payment_method",
    ],
    "order_issue": [
        "status_mystery_fee", "status_delivery_time", "status_payment_method", "status_quantity",
        "manage_upgrade", "manage_downgrade", "manage_create", "manage_cancel",
    ],
    "product_defect": [
        "refund_initiate", "refund_update", "refund_status", "return_stain", "return_color",
        "return_size",
    ],
    "purchase_dispute": [
        "bad_price_competitor", "bad_price_yesterday", "out_of_stock_general",
        "out_of_stock_one_item", "promo_code_invalid", "promo_code_out_of_date",
        "mistimed_billing_already_returned", "mistimed_billing_never_bought",
    ],
    "shipping_issue": ["status", "manage", "missing", "cost"],
    "single_item_query": ["boots", "shirt", "jeans", "jacket"],
    "storewide_query": ["pricing", "membership", "timing", "policy"],
    "subscription_inquiry": [
        "status_active", "status_due_amount", "status_due_date", "manage_pay_bill",
        "manage_extension", "manage_dispute_bill",
    ],
    "troubleshoot_site": ["credit_card", "shopping_cart", "search_results", "slow_speed"],
}  # fmt: skip

FLOWS: list[str] = sorted(ONTOLOGY)
SUBFLOWS: list[str] = sorted(s for subs in ONTOLOGY.values() for s in subs)
_FLOW_OF = {s: f for f, subs in ONTOLOGY.items() for s in subs}

# Raw data names that differ from the ontology (found by inspecting abcd_v1.1).
ALIASES = {"status_delivery_date": "status_delivery_time", "status_questions": "status_active"}
_NUMBERED = re.compile(r"_\d+$")


def flow_of(subflow: str) -> str:
    return _FLOW_OF[subflow]


def normalize_subflow(flow: str, subflow: str) -> str | None:
    """Map a raw ABCD subflow to one of the 55 ontology subflows of `flow`, or None.

    Order: exact; alias; strip a trailing `_<n>` (pricing_3 -> pricing); the part before the
    first `_` (jeans_how_4 -> jeans). FAQ variants collapse onto their topic.
    """
    allowed = ONTOLOGY.get(flow)
    if not allowed:
        return None
    for cand in (
        subflow,
        ALIASES.get(subflow),
        _NUMBERED.sub("", subflow),
        subflow.split("_", 1)[0],
    ):
        if cand in allowed:
            return cand
    return None
