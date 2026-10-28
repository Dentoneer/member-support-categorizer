"""Versioned category list (taxonomy as data) and approved change requests.

Nothing is hard-deleted: "delete" retires a category, "change" renames it and keeps the old name
in `renamed_from`, "new" adds one. Each approved request writes a full new version.
"""

from __future__ import annotations

import pandas as pd

from support_categorizer.labels import ONTOLOGY

TAX_COLS = ["version", "primary_category", "secondary_category", "description", "status",
            "renamed_from", "changed_by"]  # fmt: skip
KINDS = ("add", "rename", "retire")


def _human(name: str) -> str:
    return name.replace("_", " ")


def seed_rows(changed_by: str = "seed") -> pd.DataFrame:
    """Version 1 from the ABCD ontology (descriptions are filled from guidelines in stage 2)."""
    rows = [
        {"version": 1, "primary_category": f, "secondary_category": s,
         "description": f"{_human(f)}: {_human(s)}", "status": "active", "renamed_from": None,
         "changed_by": changed_by}
        for f, subs in ONTOLOGY.items() for s in subs
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=TAX_COLS)


def latest(tax: pd.DataFrame) -> pd.DataFrame:
    return tax[tax["version"] == tax["version"].max()]


def active_labels(tax: pd.DataFrame) -> tuple[list[str], list[str], dict[str, str]]:
    """(flows, subflows, subflow -> flow) of the latest version's active categories."""
    act = latest(tax)
    act = act[act["status"] == "active"]
    sub_to_flow = dict(zip(act["secondary_category"], act["primary_category"], strict=True))
    return sorted(set(sub_to_flow.values())), sorted(sub_to_flow), sub_to_flow


def validate_request(req: dict, tax: pd.DataFrame) -> list[str]:
    """Plain-English problems with a request (empty list = OK)."""
    cur = latest(tax)
    active = set(cur.loc[cur["status"] == "active", "secondary_category"])
    known = set(cur["secondary_category"])
    flows = set(cur.loc[cur["status"] == "active", "primary_category"])
    kind, name = req.get("kind"), (req.get("secondary_category") or "").strip()
    out = []
    if kind not in KINDS:
        out.append(f"Request type must be one of {', '.join(KINDS)}.")
    if not (req.get("reason") or "").strip():
        out.append("Please give a reason.")
    if not name:
        out.append("Please name the sub-category.")
    if kind == "add":
        if name in known:
            out.append(f"'{name}' already exists.")
        if not (req.get("description") or "").strip():
            out.append("A new category needs a description.")
        if req.get("primary_category") not in flows and not req.get("new_primary"):
            out.append(f"Primary category '{req.get('primary_category')}' does not exist; tick "
                       "'new primary category' to create it.")  # fmt: skip
    elif kind in ("rename", "retire") and name not in active:
        out.append(f"'{name}' is not an active category.")
    if kind == "rename":
        new = (req.get("new_name") or "").strip()
        if not new:
            out.append("Please give the new name.")
        elif new in known:
            out.append(f"'{new}' already exists.")
    return out


def apply_request(req: dict, tax: pd.DataFrame, decided_by: str) -> pd.DataFrame:
    """The next version's full rows after applying an approved request."""
    problems = validate_request(req, tax)
    if problems:
        raise ValueError(" ".join(problems))
    nxt = latest(tax).copy()
    nxt["version"] = int(tax["version"].max()) + 1
    nxt["changed_by"] = decided_by
    nxt["renamed_from"] = None
    name = req["secondary_category"].strip()
    hit = nxt["secondary_category"] == name
    if req["kind"] == "add":
        row = {"version": int(nxt["version"].iloc[0]), "primary_category": req["primary_category"],
               "secondary_category": name, "description": req["description"].strip(),
               "status": "active", "renamed_from": None, "changed_by": decided_by}  # fmt: skip
        nxt = pd.concat([nxt, pd.DataFrame([row])], ignore_index=True)
    elif req["kind"] == "retire":
        nxt.loc[hit, "status"] = "retired"
    else:
        nxt.loc[hit, "renamed_from"] = name
        nxt.loc[hit, "secondary_category"] = req["new_name"].strip()
        if (req.get("description") or "").strip():
            nxt.loc[hit, "description"] = req["description"].strip()
    return nxt[TAX_COLS].reset_index(drop=True)


def canonical(label: str | None, tax: pd.DataFrame) -> str | None:
    """Current name of `label`, following renames across all versions (a -> b -> c)."""
    if label is None:
        return None
    renames = tax.dropna(subset=["renamed_from"]).sort_values("version")
    step = dict(zip(renames["renamed_from"], renames["secondary_category"], strict=True))
    seen = set()
    while label in step and label not in seen:
        seen.add(label)
        label = step[label]
    return label


def read_taxonomy(spark, s) -> pd.DataFrame:
    """All versions of the category list (notebooks and the job read the latest active set)."""
    from support_categorizer.settings import TAXONOMY

    return spark.table(s.fqn(TAXONOMY)).toPandas()


def tax_version(tax: pd.DataFrame) -> int:
    return int(tax["version"].max())
