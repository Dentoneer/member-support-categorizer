"""Category guidelines: ABCD's per-subflow intent descriptions, the knowledge RAG retrieves."""

from __future__ import annotations

import json

import pandas as pd

from support_categorizer.taxonomy import latest

ABCD_GUIDELINES_URL = "https://raw.githubusercontent.com/asappresearch/abcd/master/data/guidelines.json"
GUIDELINES_FILE = "abcd_guidelines.json"

# ABCD guidelines.json display names -> ontology subflow names (checked against the file).
DISPLAY_TO_ONTOLOGY: dict[str, dict[str, str]] = {
    "Account Access": {"Recover Username": "recover_username", "Recover Password": "recover_password",
                       "Reset Two-Factor Auth": "reset_2fa"},
    "Manage Account": {"Status Service Added": "status_service_added",
                       "Status Service Removed": "status_service_removed",
                       "Status Shipping Question": "status_shipping_question",
                       "Status Credit Missing": "status_credit_missing",
                       "Manage Change Address": "manage_change_address",
                       "Manage Change Name": "manage_change_name",
                       "Manage Change Phone": "manage_change_phone",
                       "Manage Payment Method": "manage_payment_method"},
    "Order Issue": {"Status Mystery Fee": "status_mystery_fee",
                    "Status Delivery Time": "status_delivery_time",
                    "Status Payment Method": "status_payment_method", "Status Quantity": "status_quantity",
                    "Manage Upgrade": "manage_upgrade", "Manage Downgrade": "manage_downgrade",
                    "Manage Create": "manage_create", "Manage Cancel": "manage_cancel"},
    "Product Defect": {"Initiate Refund": "refund_initiate", "Update Refund": "refund_update",
                       "Refund Status": "refund_status", "Return Due to Stain": "return_stain",
                       "Return Due to Color": "return_color", "Return Due to Size": "return_size"},
    "Purchase Dispute": {"Bad Price Competitor": "bad_price_competitor",
                         "Bad Price Yesterday": "bad_price_yesterday",
                         "Out-of-Stock General": "out_of_stock_general",
                         "Out-of-Stock One Item": "out_of_stock_one_item",
                         "Promo Code Invalid": "promo_code_invalid",
                         "Promo Code Out of Date": "promo_code_out_of_date",
                         "Mistimed Billing Already Returned": "mistimed_billing_already_returned",
                         "Mistimed Billing Never Bought": "mistimed_billing_never_bought"},
    "Shipping Issue": {"Shipping Status": "status", "Manage Shipping": "manage", "Missing Item": "missing",
                       "Shipping Cost": "cost"},
    "Single-Item Query": {"Boots FAQ": "boots", "Shirt FAQ": "shirt", "Jeans FAQ": "jeans",
                          "Jacket FAQ": "jacket"},
    "Storewide Query": {"Pricing FAQ": "pricing", "Membership FAQ": "membership", "Timing FAQ": "timing",
                        "Policy FAQ": "policy"},
    "Subscription Inquiry": {"Status Active": "status_active", "Status Due Amount": "status_due_amount",
                             "Status Due Date": "status_due_date", "Manage Pay Bill": "manage_pay_bill",
                             "Manage Extension": "manage_extension",
                             "Manage Dispute Bill": "manage_dispute_bill"},
    "Troubleshoot Site": {"Invalid Credit Card": "credit_card", "Cart Not Updating": "shopping_cart",
                          "Search Not Working": "search_results", "Website Too Slow": "slow_speed"},
}  # fmt: skip


def parse_guidelines(raw: dict) -> dict[str, str]:
    """ontology subflow -> one-paragraph guideline: what the category covers (+ contrast hints)."""
    out: dict[str, str] = {}
    for flow_display, flow in raw.items():
        names = DISPLAY_TO_ONTOLOGY.get(flow_display, {})
        for sub_display, sub in flow.get("subflows", {}).items():
            key = names.get(sub_display)
            if not key:
                continue
            intent = " ".join((sub.get("instructions") or [""])[0].split())
            out[key] = f"{flow_display} > {sub_display} ({flow.get('description', '')}). {intent}".strip()
    return out


def load_guidelines_file(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return parse_guidelines(json.load(f))


def _origin(label: str, tax: pd.DataFrame) -> str:
    """The first name of `label` before any renames (renamed_from chain backwards)."""
    back = dict(zip(tax["secondary_category"], tax["renamed_from"], strict=True))
    seen = set()
    while isinstance(back.get(label), str) and back[label] and label not in seen:
        seen.add(label)
        label = back[label]
    return label


def guideline_rows(tax: pd.DataFrame, abcd: dict[str, str]) -> pd.DataFrame:
    """One row per active category of the latest taxonomy version, with the text RAG retrieves.

    ABCD's guideline when the category (or what it was renamed from) has one, else the taxonomy
    description (categories added in the review app).
    """
    cur = latest(tax)
    cur = cur[cur["status"] == "active"]
    hist = tax.dropna(subset=["renamed_from"])
    rows = []
    for r in cur.itertuples(index=False):
        origin = _origin(r.secondary_category, hist) if len(hist) else r.secondary_category
        text = abcd.get(origin)
        rows.append({"secondary_category": r.secondary_category, "primary_category": r.primary_category,
                     "guideline_text": text or str(r.description), "source": "abcd" if text else "taxonomy"})
    return pd.DataFrame(rows, columns=["secondary_category", "primary_category", "guideline_text", "source"])
