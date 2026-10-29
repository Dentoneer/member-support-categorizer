"""Supervisor review app (Databricks App, Streamlit).

Tabs: categories & volumes; review & override predictions; request / approve category changes.
Reads and writes Unity Catalog tables through the SQL warehouse attached as an app resource.
"""

from __future__ import annotations

import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from databricks import sql as dbsql  # noqa: E402
from databricks.sdk.core import Config  # noqa: E402

from support_categorizer.review import primary_volumes, volumes  # noqa: E402
from support_categorizer.settings import Settings  # noqa: E402
from support_categorizer.sql import (  # noqa: E402
    is_admin_sql,
    models_sql,
    override_insert_sql,
    predicted_counts_sql,
    request_decide_sql,
    request_insert_sql,
    requests_sql,
    review_query_sql,
    scored_true_counts_sql,
    taxonomy_insert_sql,
    taxonomy_sql,
    true_counts_sql,
)
from support_categorizer.taxonomy import active_labels, apply_request, canonical, validate_request  # noqa: E402

st.set_page_config(page_title="Support call review", layout="wide")
S = Settings(catalog=os.getenv("CATALOG", "workspace"), schema=os.getenv("SCHEMA", "support"))
CFG = Config()
USER = st.context.headers.get("X-Forwarded-Email") or "unknown"


@st.cache_resource
def _conn():
    host = CFG.host.removeprefix("https://").rstrip("/")
    return dbsql.connect(server_hostname=host,
                         http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
                         credentials_provider=lambda: CFG.authenticate)  # fmt: skip


@st.cache_data(ttl=30, show_spinner=False)
def query(q: str, params: tuple = ()) -> pd.DataFrame:
    with _conn().cursor() as c:
        c.execute(q, dict(params))
        return c.fetchall_arrow().to_pandas()


def run(q: str, params: dict | None = None) -> None:
    with _conn().cursor() as c:
        c.execute(q, params or {})
    query.clear()


try:
    tax = query(taxonomy_sql(S))
    models = query(models_sql(S))["model"].tolist()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the tables in {S.catalog}.{S.schema}. Run notebooks 01-05 and the app grants "
             f"cell in notebook 07, then reload. ({str(e)[:200]})")
    st.stop()
flows, subflows, sub_to_flow = active_labels(tax)
is_admin = bool(query(is_admin_sql(S), (("email", USER),))["ok"].iloc[0])

st.title("Member-support call review")
role = " (admin)" if is_admin else ""
st.caption(f"Signed in as {USER}{role} · category list v{int(tax['version'].max())}")
model = st.sidebar.selectbox("Model", models, index=models.index("dspy_optimized")
                             if "dspy_optimized" in models else 0)  # fmt: skip
tab_vol, tab_review, tab_req = st.tabs(["Categories & volumes", "Review & override", "Category requests"])

# ---- 1. categories and their volumes ------------------------------------------------------
with tab_vol:
    import altair as alt

    m = (("model", model),)
    vol = volumes(query(true_counts_sql(S)), query(predicted_counts_sql(S), m), tax,
                  scored_true=query(scored_true_counts_sql(S), m))  # fmt: skip
    prim = primary_volumes(vol)

    def bars(df: pd.DataFrame, key: str):
        """True vs predicted calls side by side, y-axis from zero."""
        long = df.melt(id_vars=key, value_vars=["calls_true_scored", "calls_predicted"],
                       var_name="series", value_name="calls")  # fmt: skip
        long["series"] = long["series"].map({"calls_true_scored": "true label",
                                             "calls_predicted": "model's label"})  # fmt: skip
        return (alt.Chart(long).mark_bar()
                .encode(x=alt.X(f"{key}:N", title=None, axis=alt.Axis(labelAngle=-40)),
                        xOffset="series:N",
                        y=alt.Y("calls:Q", scale=alt.Scale(zero=True), title="calls"),
                        color=alt.Color("series:N", title=None), tooltip=[key, "series", "calls"])
                .properties(height=340))  # fmt: skip

    st.subheader("Calls per category")
    st.caption("On the calls this model scored: how many truly belong to each category vs how many the "
               "model put there (supervisor overrides win). calls_all in the table = every call.")
    st.altair_chart(bars(prim, "primary_category"), use_container_width=True)
    pick = st.selectbox("Drill into a category", prim["primary_category"])
    st.altair_chart(bars(vol[vol["primary_category"] == pick], "secondary_category"),
                    use_container_width=True)  # fmt: skip
    st.dataframe(vol, use_container_width=True, hide_index=True)

# ---- 2. review and override --------------------------------------------------------------
with tab_review:
    df = query(review_query_sql(S), (("model", model),))
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", ["All", *flows])
    low = c2.checkbox("Only Jev confidence below 0.8 (the human-review queue)")
    view = df if cat == "All" else df[df["primary_category"] == cat]
    if low:
        view = view[view["jev_confidence"].isna() | (view["jev_confidence"] < 0.8)]
    true_now = view["true_secondary"].map(lambda x: canonical(x, tax))  # today's name after renames
    view = view.assign(correct=view["secondary_category"] == true_now)
    st.dataframe(view.drop(columns=["call_transcript"]), use_container_width=True, hide_index=True)
    if len(view):
        cid = st.selectbox("Open a call", view["call_id"])
        row = view[view["call_id"] == cid].iloc[0]
        st.text_area("Transcript", row["call_transcript"], height=260, disabled=True)
        st.write(f"**Model:** {row['secondary_category']} · **True label:** {row['true_secondary']} · "
                 f"**Jev:** {row['jev_confidence']} · **Summary:** {row['concise_summary']}")
        st.session_state["selected_call"] = cid
        with st.form("override"):
            new = st.selectbox("Correct sub-category", subflows,
                               index=subflows.index(row["secondary_category"])
                               if row["secondary_category"] in subflows else 0)  # fmt: skip
            note = st.text_input("Note (optional)")
            if st.form_submit_button("Save override"):
                run(override_insert_sql(S), {"call_id": cid, "model": model, "secondary_category": new,
                                             "primary_category": sub_to_flow[new], "reviewer": USER,
                                             "note": note})  # fmt: skip
                st.success(f"Saved: call {cid} → {new}")

# ---- 3. category change requests ---------------------------------------------------------
with tab_req:
    st.subheader("Request a change to the category list")
    st.caption("New, change (rename) or delete (retire) a category. An admin approves it; then rerun "
               "notebook 03 to retrain on the new list.")
    # Choices that change the form's fields live outside the form (forms only rerun on submit).
    kind = st.radio("Request", ["add", "rename", "retire"], horizontal=True,
                    format_func={"add": "New category", "rename": "Change (rename)",
                                 "retire": "Delete (retire)"}.get)  # fmt: skip
    new_primary = kind == "add" and st.checkbox("It belongs to a new top-level category")
    with st.form("request"):
        if kind == "add":
            if new_primary:
                primary = st.text_input("Top-level category")
            else:
                primary = st.selectbox("Top-level category", flows)
            name = st.text_input("New sub-category name (snake_case)")
        else:
            primary = None
            name = st.selectbox("Sub-category", subflows)
        new_name = st.text_input("New name") if kind == "rename" else None
        description = st.text_area("Description (what calls belong here)")
        reason = st.text_area("Why is this needed?")
        selected = st.session_state.get("selected_call", "")
        examples = st.text_input("Example call ids (comma separated)", selected)
        if st.form_submit_button("Submit request"):
            req = {"kind": kind, "primary_category": primary or (sub_to_flow.get(name) if name else None),
                   "secondary_category": (name or "").strip(), "new_name": new_name,
                   "description": description, "reason": reason, "new_primary": new_primary}
            problems = validate_request(req, tax)
            if problems:
                st.error(" ".join(problems))
            else:
                run(request_insert_sql(S), {"request_id": str(uuid.uuid4()), "kind": kind,
                                            "primary_category": req["primary_category"],
                                            "secondary_category": req["secondary_category"],
                                            "new_name": new_name, "description": description,
                                            "reason": reason, "example_call_ids": examples,
                                            "requested_by": USER})  # fmt: skip
                st.success("Request submitted. An admin will review it.")

    reqs = query(requests_sql(S))
    st.subheader("Requests")
    st.dataframe(reqs, use_container_width=True, hide_index=True)
    pending = reqs[reqs["status"] == "pending"]
    if is_admin and len(pending):
        st.subheader("Approve or reject (admins)")
        rid = st.selectbox("Pending request", pending["request_id"],
                           format_func=lambda r: "{kind}: {secondary_category}".format(
                               **pending.set_index("request_id").loc[r].to_dict()))  # fmt: skip
        r = pending.set_index("request_id").loc[rid].to_dict()
        note = st.text_input("Decision note")
        a, b = st.columns(2)
        if a.button("Approve"):
            req = {**r, "new_primary": r["primary_category"] not in flows}
            try:
                q, params = taxonomy_insert_sql(S, apply_request(req, tax, USER).to_dict("records"))
            except ValueError as e:
                st.error(f"Cannot apply: {e}")
            else:
                run(q, params)
                run(request_decide_sql(S), {"status": "approved", "decided_by": USER, "note": note,
                                            "request_id": rid})  # fmt: skip
                st.success("Approved: the category list has a new version. Rerun notebook 03 to retrain "
                           "and register a new model version; until then predictions use the old list.")
        if b.button("Reject"):
            run(request_decide_sql(S), {"status": "rejected", "decided_by": USER, "note": note,
                                        "request_id": rid})  # fmt: skip
            st.info("Rejected.")
    elif not is_admin:
        st.caption("Only admins (listed in the app_admins table) can approve requests.")
