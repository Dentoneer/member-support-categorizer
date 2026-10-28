import pandas as pd
import pytest

from support_categorizer.errors import UserFacingError
from support_categorizer.jev import (
    FakeJevClient,
    JevError,
    JevResponse,
    build_questions,
    get_jev_key,
    jev_report,
    jev_scorer,
    parse_jev_response,
    validate_rows,
)

IDS = ["primary_ok", "secondary_ok", "summary_faithful"]


def test_build_questions_ids_and_text():
    q = build_questions("account_access", "reset_2fa", "Customer cannot log in.")
    assert list(q) == IDS and all(v["type"] == "noul" for v in q.values())
    assert '"account access"' in q["primary_ok"]["instructions"]
    assert '"reset 2fa"' in q["secondary_ok"]["instructions"]
    assert "Customer cannot log in." in q["summary_faithful"]["instructions"]


@pytest.mark.parametrize("body", [
    {"answers": {"primary_ok": {"type": "noul", "noul": 0.9}, "secondary_ok": 0.2, "summary_faithful": 1}},
    {"primary_ok": 0.9, "secondary_ok": 0.2, "summary_faithful": 1.0, "model": "jev"},
])
def test_parse_response_shapes(body):
    r = parse_jev_response(body, IDS)
    assert r.nouls == {"primary_ok": 0.9, "secondary_ok": 0.2, "summary_faithful": 1.0}


def test_parse_response_missing_raises():
    with pytest.raises(JevError, match="keys found"):
        parse_jev_response({"answers": {"primary_ok": 0.5}}, IDS)


def _row(cid):
    return {"call_id": cid, "model": "dspy_optimized", "call_transcript": f"t{cid}",
            "primary_category": "account_access", "secondary_category": "reset_2fa",
            "concise_summary": "s"}


def test_validate_rows_survives_one_error():
    def responder(model, state, questions):
        if state == "t2":
            raise JevError("Jev HTTP 400: bad")
        return JevResponse({k: 0.7 for k in questions}, 10, 0, model)

    out = {r["call_id"]: r for r in validate_rows(FakeJevClient(responder), [_row("1"), _row("2")],
                                                  "jev-1.14.0", sleep=lambda s: None)}
    assert out["1"]["secondary_ok"] == 0.7 and out["2"]["secondary_ok"] is None
    assert out["2"]["model"] == "dspy_optimized"


def test_validate_rows_skips_invalid_predictions():
    client = FakeJevClient(lambda m, s, q: JevResponse({k: 1.0 for k in q}, 1, 0, m))
    row = {**_row("3"), "secondary_category": None}
    assert validate_rows(client, [row], "jev", sleep=lambda s: None)[0]["secondary_ok"] is None
    assert client.calls == []


class _NoSecret:
    class secrets:  # noqa: N801
        @staticmethod
        def get(scope, key):
            raise RuntimeError("missing")


def test_missing_secret_message():
    from support_categorizer.settings import Settings

    with pytest.raises(UserFacingError, match="notebook 00"):
        get_jev_key(_NoSecret, Settings())


def test_jev_report_auroc_and_split():
    truth = pd.DataFrame({"call_id": list("abcd"), "secondary_category": ["x", "x", "y", "y"]})
    preds = pd.DataFrame({"call_id": list("abcd"), "secondary_category": ["x", "y", "y", "x"]})
    val = pd.DataFrame({"call_id": list("abcd"), "secondary_ok": [0.95, 0.1, 0.9, 0.2]})
    rep = jev_report(val, preds, truth)
    assert rep["auroc"] == 1.0
    assert rep["auto_accept_rate"] == 0.5 and rep["accuracy_auto_accepted"] == 1.0
    assert rep["accuracy_sent_to_human"] == 0.0


def test_jev_scorer_returns_probability():
    client = FakeJevClient(lambda m, s, q: JevResponse({k: 0.8 for k in q}, 1, 0, m))
    sc = jev_scorer(client, "jev")
    fb = sc(inputs={"call_transcript": "t"},
            outputs={"primary_category": "account_access", "secondary_category": "reset_2fa",
                     "concise_summary": "s"})
    assert fb.value == 0.8
