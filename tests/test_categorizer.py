import typing

import dspy
import pytest
from dspy.utils import DummyLM

from support_categorizer.categorizer import (
    CallCategorizerProgram,
    examples_from,
    make_signature,
    metric,
    predict_rows,
)

FLOWS = ["account_access", "shipping_issue"]
SUBS = {"reset_2fa": "account_access", "recover_password": "account_access", "cost": "shipping_issue"}
ROW = {"call_id": "1", "call_transcript": "Customer: my 2fa broke", "customer_tier": "gold"}


def _answer(primary="shipping_issue", secondary="reset_2fa", urgency="3"):
    return {"reasoning": "r", "primary_category": primary, "secondary_category": secondary,
            "urgency_score": urgency, "concise_summary": "Customer cannot pass 2FA."}


def _program():
    return CallCategorizerProgram(FLOWS, SUBS)


def test_signature_fields_and_literals():
    sig = make_signature(FLOWS, sorted(SUBS))
    assert sig.__name__ == "CallCategorizer"
    assert list(sig.input_fields) == ["call_transcript", "customer_tier", "guidelines"]
    assert list(sig.output_fields) == ["primary_category", "secondary_category", "urgency_score",
                                       "concise_summary"]
    assert set(typing.get_args(sig.output_fields["secondary_category"].annotation)) == set(SUBS)


def test_program_rederives_primary():
    with dspy.context(lm=DummyLM([_answer()])):
        pred = _program()(call_transcript="x", customer_tier="gold")
    assert pred.secondary_category == "reset_2fa" and pred.primary_category == "account_access"


def test_predict_rows_marks_out_of_set_invalid():
    lm = DummyLM([_answer(secondary="not_a_label")] * 10)
    out = predict_rows(_program(), [ROW], "dspy_zero_shot", "v1", "run", lm=lm, sleep=lambda s: None)
    assert out[0]["valid"] is False and out[0]["secondary_category"] is None
    assert out[0]["call_id"] == "1" and out[0]["model"] == "dspy_zero_shot"


def test_predict_rows_valid_row_and_urgency_clamped():
    lm = DummyLM([_answer(urgency="9")])
    out = predict_rows(_program(), [ROW], "m", "v1", "run", lm=lm, sleep=lambda s: None)
    r = out[0]
    assert r["valid"] is True and r["secondary_category"] == "reset_2fa"
    assert r["primary_category"] == "account_access" and r["urgency_score"] == 5


class _Flaky:
    def __init__(self, fails):
        self.fails = fails

    def __call__(self, **kw):
        if self.fails.get(kw["call_transcript"], 0) > 0:
            self.fails[kw["call_transcript"]] -= 1
            raise RuntimeError("429 rate limited")
        return dspy.Prediction(primary_category="account_access", secondary_category="reset_2fa",
                               urgency_score=2, concise_summary="s")


def test_predict_rows_retries_then_continues():
    rows = [{**ROW, "call_id": "a", "call_transcript": "a"},
            {**ROW, "call_id": "b", "call_transcript": "b"},
            {**ROW, "call_id": "c", "call_transcript": "c"}]
    prog = _Flaky({"a": 2, "b": 99})
    prog.sub_to_flow = SUBS
    out = {r["call_id"]: r for r in predict_rows(prog, rows, "m", "v", "r", sleep=lambda s: None)}
    assert out["a"]["valid"] is True  # recovered after 2 failures
    assert out["b"]["valid"] is False  # always failing -> invalid, no raise
    assert out["c"]["valid"] is True


@pytest.mark.parametrize("pred_sec, want", [("reset_2fa", 1.0), ("recover_password", 0.5),
                                            ("cost", 0.0), (None, 0.0)])
def test_metric_partial_credit(pred_sec, want):
    ex = dspy.Example(primary_category="account_access", secondary_category="reset_2fa")
    pred = dspy.Prediction(secondary_category=pred_sec,
                           primary_category=SUBS.get(pred_sec) if pred_sec else None)
    assert metric(ex, pred) == want


def test_examples_from_has_inputs():
    import pandas as pd

    df = pd.DataFrame([{**ROW, "primary_category": "account_access", "secondary_category": "reset_2fa"}])
    ex = examples_from(df)[0]
    assert set(ex.inputs().keys()) == {"call_transcript", "customer_tier", "guidelines"}
    assert ex.secondary_category == "reset_2fa"
