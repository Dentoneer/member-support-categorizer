"""DSPy CallCategorizer: signature, ChainOfThought program, metric, robust batch prediction."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import dspy
import pandas as pd

INSTRUCTION = (
    "Categorize a member-support conversation between a customer and an agent. Pick the "
    "secondary_category (the specific reason for contacting support) from the allowed list, and the "
    "primary_category it belongs to. Rate urgency from 1 (routine question) to 5 (urgent: money "
    "lost, account locked, order stuck). Write a factual summary of at most 40 words."
)


def make_signature(flows: list[str], subflows: list[str]) -> type[dspy.Signature]:
    """The `CallCategorizer` signature with categories constrained to the active label set."""
    fields = {
        "call_transcript": (str, dspy.InputField(desc="Customer/Agent lines of one support call")),
        "customer_tier": (str, dspy.InputField(desc="membership level: guest/bronze/silver/gold")),
        "guidelines": (str, dspy.InputField(desc="optional category guidelines; may be empty")),
        "primary_category": (Literal[tuple(flows)], dspy.OutputField()),
        "secondary_category": (Literal[tuple(subflows)], dspy.OutputField()),
        "urgency_score": (int, dspy.OutputField(desc="1 = routine ... 5 = urgent")),
        "concise_summary": (str, dspy.OutputField(desc="at most 40 words")),
    }
    return dspy.make_signature(fields, INSTRUCTION, signature_name="CallCategorizer")


class CallCategorizerProgram(dspy.Module):
    """ChainOfThought over CallCategorizer; primary is re-derived from the secondary category."""

    def __init__(self, flows: list[str], sub_to_flow: dict[str, str]):
        super().__init__()
        self.sub_to_flow = dict(sub_to_flow)
        self.categorize = dspy.ChainOfThought(make_signature(flows, sorted(sub_to_flow)))

    def forward(self, call_transcript: str, customer_tier: str, guidelines: str = ""):
        pred = self.categorize(
            call_transcript=call_transcript, customer_tier=customer_tier, guidelines=guidelines
        )
        # The label is one path: a sub-category implies its category.
        if pred.secondary_category in self.sub_to_flow:
            pred.primary_category = self.sub_to_flow[pred.secondary_category]
        return pred


def metric(example, pred, trace=None) -> float:
    """1.0 sub-category right; 0.5 only the category right; else 0."""
    if getattr(pred, "secondary_category", None) == example.secondary_category:
        return 1.0
    if getattr(pred, "primary_category", None) == example.primary_category:
        return 0.5
    return 0.0


def examples_from(pdf: pd.DataFrame) -> list[dspy.Example]:
    return [
        dspy.Example(
            call_transcript=r.call_transcript,
            customer_tier=r.customer_tier,
            guidelines=getattr(r, "guidelines", "") or "",
            primary_category=r.primary_category,
            secondary_category=r.secondary_category,
        ).with_inputs("call_transcript", "customer_tier", "guidelines")
        for r in pdf.itertuples(index=False)
    ]


def _urgency(value) -> int | None:
    try:
        return min(5, max(1, int(value)))
    except (TypeError, ValueError):
        return None


def _invalid(base: dict) -> dict:
    return {**base, "primary_category": None, "secondary_category": None, "urgency_score": None,
            "concise_summary": None, "valid": False}  # fmt: skip


def predict_rows(
    program,
    rows: list[dict],
    model: str,
    model_version: str,
    run_id: str,
    *,
    lm=None,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    threads: int = 4,
) -> list[dict]:
    """One prediction row per input row. A row that keeps failing (rate limit, unparseable or
    out-of-set answer) is retried with backoff, then marked `valid=False`; never raises."""
    lm = lm or dspy.settings.lm
    allowed = program.sub_to_flow

    def one(row: dict) -> dict:
        base = {"call_id": row["call_id"], "model": model, "model_version": model_version,
                "run_id": run_id}  # fmt: skip
        for attempt in range(retries + 1):
            try:
                with dspy.context(lm=lm):
                    pred = program(call_transcript=row["call_transcript"],
                                   customer_tier=row["customer_tier"],
                                   guidelines=row.get("guidelines", "") or "")  # fmt: skip
                sec = pred.secondary_category
                if sec not in allowed:
                    raise ValueError(f"answer {sec!r} is not an allowed sub-category")
                return {**base, "primary_category": allowed[sec], "secondary_category": sec,
                        "urgency_score": _urgency(pred.urgency_score),
                        "concise_summary": str(pred.concise_summary or "").strip(),
                        "valid": True}  # fmt: skip
            except Exception as e:  # one bad row must never stop the batch
                if attempt == retries:
                    print(f"call {row['call_id']}: gave up after {retries + 1} tries ({str(e)[:120]})")
                    return _invalid(base)
                sleep(2**attempt)
        return _invalid(base)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        return list(pool.map(one, rows))


def configure_lm(endpoint: str) -> dspy.LM:
    """Databricks Foundation Model API endpoint as the DSPy LM.

    Inside Databricks, LiteLLM signs in through the Databricks SDK on every call, so no token is
    copied (a copied notebook token can expire during a long optimization run).
    """
    lm = dspy.LM(f"databricks/{endpoint}", temperature=0, max_tokens=1000, cache=False)
    dspy.configure(lm=lm)
    return lm


def bake_off(endpoints: list[str], program_factory: Callable[[], CallCategorizerProgram],
             dev: pd.DataFrame, n: int = 30, threads: int = 4) -> pd.DataFrame:
    """Same `n` dev calls through each endpoint; accuracy, valid rate, seconds."""
    rows = dev.head(n).to_dict("records")
    truth = {r["call_id"]: r["secondary_category"] for r in rows}
    out = []
    for ep in endpoints:
        start = time.time()
        try:
            lm = configure_lm(ep)
            preds = predict_rows(program_factory(), rows, "bake_off", ep, "", lm=lm, retries=1,
                                 threads=threads)  # fmt: skip
            valid = [p for p in preds if p["valid"]]
            right = sum(truth[p["call_id"]] == p["secondary_category"] for p in valid)
            out.append({"endpoint": ep, "secondary_accuracy": right / len(rows),
                        "valid_rate": len(valid) / len(rows), "seconds": round(time.time() - start),
                        "error": ""})  # fmt: skip
        except Exception as e:  # noqa: BLE001 - an endpoint that fails outright is reported
            out.append({"endpoint": ep, "secondary_accuracy": 0.0, "valid_rate": 0.0,
                        "seconds": round(time.time() - start), "error": str(e)[:200]})  # fmt: skip
    return pd.DataFrame(out).sort_values("secondary_accuracy", ascending=False)


def pick_endpoint(bake: pd.DataFrame, default: str, min_valid: float = 0.9) -> str:
    ok = bake[(bake["valid_rate"] >= min_valid) & (bake["error"] == "")]
    return str(ok.iloc[0]["endpoint"]) if len(ok) else default


def optimize(program, trainset: list, valset: list, mode: Literal["bootstrap", "mipro"] = "bootstrap"):
    if mode == "mipro":
        opt = dspy.MIPROv2(metric=metric, auto="light")
        return opt.compile(program, trainset=trainset, valset=valset)
    opt = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4)
    return opt.compile(program, trainset=trainset)
