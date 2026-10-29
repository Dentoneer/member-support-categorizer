"""Pandas logic behind the review app (kept here so it is unit-tested)."""

from __future__ import annotations

import pandas as pd

from support_categorizer.taxonomy import canonical, latest


def volumes(true_counts: pd.DataFrame, pred_counts: pd.DataFrame, tax: pd.DataFrame,
            scored_true: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calls per category: by true label (all calls), and on the model's scored calls by true label
    (`calls_true_scored`) and by the model's label (`calls_predicted`).

    Old labels are mapped through renames; retired categories stay listed with their status.
    """
    cur = latest(tax)[["primary_category", "secondary_category", "status"]]
    out = cur.copy()
    if scored_true is None:
        scored_true = pd.DataFrame({"secondary_category": [], "n": []})
    cols = ["calls_all", "calls_true_scored", "calls_predicted"]
    for col, counts in (("calls_all", true_counts), ("calls_true_scored", scored_true),
                        ("calls_predicted", pred_counts)):
        c = counts.copy()
        c["secondary_category"] = [canonical(x, tax) for x in c["secondary_category"]]
        c = c.groupby("secondary_category", as_index=False)["n"].sum().rename(columns={"n": col})
        out = out.merge(c, on="secondary_category", how="left")
    out[cols] = out[cols].fillna(0).astype(int)
    total = out["calls_predicted"].sum()
    out["share_predicted"] = (out["calls_predicted"] / total).round(3) if total else 0.0
    return out.sort_values(["primary_category", "calls_predicted"], ascending=[True, False]).reset_index(
        drop=True
    )


def primary_volumes(vol: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("calls_all", "calls_true_scored", "calls_predicted") if c in vol]
    return (vol.groupby("primary_category", as_index=False)[cols].sum()
            .sort_values("calls_predicted", ascending=False).reset_index(drop=True))  # fmt: skip
