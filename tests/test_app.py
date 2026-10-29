"""Smoke test: the review app renders all tabs and saves an override against a fake warehouse."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from support_categorizer.taxonomy import seed_rows

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")


class _Cursor:
    def __init__(self, log):
        self.log, self.q = log, ""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, q, params=None):
        self.q = q
        self.log.append((q, params))

    def fetchall_arrow(self):
        q = self.q
        if "ORDER BY m.value DESC LIMIT 1" in q:
            df = pd.DataFrame({"model": ["dspy_optimized"]})
        elif "DISTINCT model" in q:
            df = pd.DataFrame({"model": ["baseline", "dspy_optimized"]})
        elif "app_admins" in q:
            df = pd.DataFrame({"ok": [True]})
        elif "taxonomy_requests" in q:
            df = pd.DataFrame([{"request_id": "r1", "kind": "retire", "primary_category": "shipping_issue",
                                "secondary_category": "cost", "new_name": None, "description": "",
                                "reason": "unused", "example_call_ids": None, "requested_by": "a@b.c",
                                "requested_at": None, "status": "pending", "decided_by": None,
                                "decided_at": None, "decision_note": None}])
        elif "taxonomy" in q:
            df = seed_rows()
        elif "p.model = :model" in q and "call_transcript" in q:
            df = pd.DataFrame([{"call_id": "1", "customer_tier": "gold", "true_primary": "account_access",
                                "true_secondary": "reset_2fa", "primary_category": "account_access",
                                "secondary_category": "reset_2fa", "urgency_score": 3,
                                "concise_summary": "2FA broken", "valid": True, "jev_confidence": 0.6,
                                "override_secondary": None, "override_by": None,
                                "call_transcript": "Customer: my 2fa"}])
        else:  # true / predicted counts
            df = pd.DataFrame({"secondary_category": ["reset_2fa", "cost"], "n": [5, 2]})
        return pa.Table.from_pandas(df)


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    log = []

    class _Conn:
        def cursor(self):
            return _Cursor(log)

    class _Cfg:
        host = "https://example.cloud.databricks.com"

        def authenticate(self):
            return {}

    import databricks.sdk.core as core
    import databricks.sql as dbsql

    monkeypatch.setattr(dbsql, "connect", lambda **kw: _Conn())
    monkeypatch.setattr(core, "Config", _Cfg)
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh")
    import streamlit as st

    st.cache_resource.clear()  # the app caches its connection across runs in this process
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at, log


def test_app_renders_all_tabs(app):
    at, _ = app
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Categories & volumes", "Review & override", "Category requests"]


def test_app_saves_override(app):
    at, log = app
    at.button[0].click()  # first form submit button = "Save override" (tab 2 renders before tab 3)
    at.run()
    assert not at.exception, at.exception
    inserts = [p for q, p in log if "INSERT INTO workspace.support.overrides" in q]
    assert inserts and inserts[-1]["call_id"] == "1" and inserts[-1]["primary_category"]


def test_admin_can_approve_retire(app):
    at, log = app
    approve = next(b for b in at.button if b.label == "Approve")
    approve.click()
    at.run()
    assert not at.exception, at.exception
    tax_insert = [p for q, p in log if q.startswith("INSERT INTO workspace.support.taxonomy ")]
    assert tax_insert and tax_insert[-1]["r0_version"] == 2
    assert any("UPDATE workspace.support.taxonomy_requests" in q for q, _ in log)
