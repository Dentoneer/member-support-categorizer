"""Jev (TypeSafe AI) validator: calibrated yes/no checks of each prediction.

Jev answers typed questions about a "state" (here the call transcript) with a probability ("noul").
One request per call, three questions. Ported from the CFPB project's client, without its
budget ledger (Jev costs cents here).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd

from support_categorizer.errors import UserFacingError
from support_categorizer.settings import SECRET_KEY, SECRET_SCOPE, Settings

QUESTION_IDS = ["primary_ok", "secondary_ok", "summary_faithful"]
AUTO_ACCEPT = 0.8  # Jev probability above which a call would skip human review


@dataclass
class JevResponse:
    nouls: dict[str, float]
    input_tokens: int | None
    output_tokens: int | None
    model: str


class JevError(Exception):
    """HTTP error or unreadable answer from Jev (that call stays unvalidated)."""


class JevRetryable(JevError):
    """429 / 5xx / timeout: worth retrying."""


def _noul_value(node: object) -> float | None:
    if isinstance(node, dict):
        node = node.get("noul")
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    f = float(node)
    return f if 0.0 <= f <= 1.0 else None


def parse_jev_response(body: object, question_ids: list[str], model: str = "") -> JevResponse:
    """{id: noul} from a Jev body: under "answers", "nouls", or at the top level."""
    if not isinstance(body, dict):
        raise JevError(f"Jev response is not a JSON object: {str(body)[:200]!r}")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    for cont in (body.get("answers"), body.get("nouls"), body):
        if not isinstance(cont, dict):
            continue
        vals = {q: _noul_value(cont.get(q)) for q in question_ids}
        if all(v is not None for v in vals.values()):
            return JevResponse({q: float(v) for q, v in vals.items()}, usage.get("input_tokens"),
                               usage.get("output_tokens"), str(body.get("model") or model))  # fmt: skip
    keys = {"top-level": sorted(body)}
    for k in ("answers", "nouls"):
        if isinstance(body.get(k), dict):
            keys[k] = sorted(body[k])
    raise JevError(f"no noul in [0, 1] for each of {question_ids}; keys found: {keys}")


class HttpJevClient:
    """POST {base_url}/v1/systemone with a bearer key."""

    def __init__(self, api_key: str, base_url: str = "https://api.typesafe.ai", timeout: float = 60):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def system_one(self, model: str, state: str, questions: dict[str, dict]) -> JevResponse:
        import httpx

        try:
            r = httpx.post(
                f"{self.base_url}/v1/systemone",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": model, "state": state, "questions": questions},
                timeout=self.timeout,
            )
        except httpx.TimeoutException as e:
            raise JevRetryable(f"Jev request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise JevError(f"Jev request failed: {e}") from e
        if r.status_code == 429 or r.status_code >= 500:
            raise JevRetryable(f"Jev HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code in (401, 403):
            raise UserFacingError("Jev rejected the key. Store it again with notebook 00's "
                                  "'Store the Jev key' cells.")  # fmt: skip
        if r.status_code >= 400:
            raise JevError(f"Jev HTTP {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
        except ValueError as e:
            raise JevError(f"Jev returned non-JSON: {r.text[:200]!r}") from e
        return parse_jev_response(body, list(questions), model)


class FakeJevClient:
    """Test double: `responder(model, state, questions) -> JevResponse`; records every call."""

    def __init__(self, responder: Callable[[str, str, dict], JevResponse]):
        self.responder = responder
        self.calls: list[tuple] = []

    def system_one(self, model: str, state: str, questions: dict[str, dict]) -> JevResponse:
        self.calls.append((model, state, questions))
        return self.responder(model, state, questions)


def _human(label: str) -> str:
    return label.replace("_", " ")


def build_questions(primary: str, secondary: str, summary: str) -> dict[str, dict]:
    """Three testable claims about the call, answered with a probability each."""
    claims = {
        "primary_ok": f'The customer contacted support about: "{_human(primary)}".',
        "secondary_ok": f'Specifically, the reason for this call is: "{_human(secondary)}" '
                        f'(within "{_human(primary)}").',
        "summary_faithful": f'This summary describes the conversation accurately and adds nothing '
                            f'that was not said: "{summary}".',
    }  # fmt: skip
    return {k: {"type": "noul", "instructions": v} for k, v in claims.items()}


def get_jev_key(dbutils, s: Settings) -> str:
    try:
        return dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)
    except Exception:
        raise UserFacingError(
            "The Jev key is not stored. Run the 'Store the Jev key' cells at the end of notebook 00 "
            "once, then rerun this cell."
        ) from None


def validate_rows(client, rows: list[dict], jev_model: str, *, threads: int = 4, retries: int = 3,
                  sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """One validation row per prediction row; a failed call leaves that row's probabilities null."""

    def one(row: dict) -> dict:
        base = {"call_id": row["call_id"], "model": row["model"], **dict.fromkeys(QUESTION_IDS)}
        if not row.get("secondary_category"):
            return base  # invalid prediction: nothing to validate
        q = build_questions(row["primary_category"], row["secondary_category"],
                            row.get("concise_summary") or "")  # fmt: skip
        for attempt in range(retries + 1):
            try:
                return {**base, **client.system_one(jev_model, row["call_transcript"], q).nouls}
            except JevRetryable as e:
                if attempt == retries:
                    print(f"call {row['call_id']}: Jev unavailable ({str(e)[:100]})")
                    return base
                sleep(2**attempt)
            except JevError as e:
                print(f"call {row['call_id']}: Jev error ({str(e)[:100]})")
                return base
        return base

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        return list(pool.map(one, rows))


def jev_report(validations: pd.DataFrame, preds: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Does Jev's confidence separate right from wrong predictions?"""
    from sklearn.metrics import roc_auc_score

    m = truth[["call_id", "secondary_category"]].merge(
        preds[["call_id", "secondary_category"]], on="call_id", suffixes=("_true", "_pred")
    ).merge(validations[["call_id", "secondary_ok"]], on="call_id").dropna(subset=["secondary_ok"])
    right = m["secondary_category_true"] == m["secondary_category_pred"]
    high = m["secondary_ok"] > AUTO_ACCEPT
    auroc = float(roc_auc_score(right, m["secondary_ok"])) if right.nunique() == 2 else None
    return {
        "validated_calls": int(len(m)),
        "auroc": auroc,
        "auto_accept_rate": float(high.mean()) if len(m) else 0.0,
        "accuracy_auto_accepted": float(right[high].mean()) if high.any() else None,
        "accuracy_sent_to_human": float(right[~high].mean()) if (~high).any() else None,
    }


def jev_scorer(client, jev_model: str):
    """An MLflow GenAI scorer: Jev's probability that the sub-category is right."""
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    @scorer(name="jev_secondary_ok")
    def jev_secondary_ok(inputs, outputs) -> Feedback:
        q = build_questions(outputs["primary_category"], outputs["secondary_category"],
                            outputs.get("concise_summary") or "")  # fmt: skip
        try:
            r = client.system_one(jev_model, inputs["call_transcript"], q)
        except JevError as e:
            return Feedback(error=str(e))
        return Feedback(value=r.nouls["secondary_ok"],
                        rationale=f"summary faithful p={r.nouls['summary_faithful']:.2f}")  # fmt: skip

    return jev_secondary_ok
