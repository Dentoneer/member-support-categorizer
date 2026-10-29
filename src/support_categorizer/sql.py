"""SQL builders: ai_query batch inference, model comparison, governance, review app queries."""

from __future__ import annotations

import json

from support_categorizer.pii import SUMMARY_PII_RULE
from support_categorizer.settings import (
    APP_ADMINS,
    CALLS_CLEAN,
    EVAL_SAMPLE,
    METRICS,
    OVERRIDES,
    PREDICTIONS,
    TAXONOMY,
    TAXONOMY_REQUESTS,
    VALIDATIONS,
    Settings,
)

AI_QUERY_INSTRUCTION = (
    "You categorize member-support calls. Read the call and answer with a JSON object with keys "
    "primary_category, secondary_category (choose from the allowed values), urgency_score (integer "
    "1 = routine to 5 = urgent) and concise_summary (at most 40 words). " + SUMMARY_PII_RULE
)
OUTPUT_DDL = "primary_category STRING, secondary_category STRING, urgency_score INT, concise_summary STRING"


def sql_str(text: str) -> str:
    """A Spark SQL string literal (backslash escaping, as Spark SQL uses)."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def response_schema(flows: list[str], subflows: list[str]) -> str:
    """JSON schema for ai_query's responseFormat: categories constrained to the label set."""
    schema = {
        "type": "object",
        "properties": {
            "primary_category": {"type": "string", "enum": list(flows)},
            "secondary_category": {"type": "string", "enum": list(subflows)},
            "urgency_score": {"type": "integer"},  # clamped to 1-5 in SQL
            "concise_summary": {"type": "string"},
        },
        "required": ["primary_category", "secondary_category", "urgency_score", "concise_summary"],
        "additionalProperties": False,
    }
    return json.dumps({"type": "json_schema",
                       "json_schema": {"name": "call_category", "schema": schema, "strict": True}})


def _map_literal(d: dict[str, str]) -> str:
    return "map(" + ", ".join(f"{sql_str(k)}, {sql_str(v)}" for k, v in sorted(d.items())) + ")"


def ai_query_sql(s: Settings, endpoint: str, flows: list[str], sub_to_flow: dict[str, str],
                 structured: bool = True) -> str:
    """Batch-categorize the eval sample with ai_query; one parsed row per call (never fails the query)."""
    subs = sorted(sub_to_flow)
    allowed = ", ".join(subs)
    instr = AI_QUERY_INSTRUCTION + f" Allowed secondary_category values: {allowed}."
    if not structured:
        instr += " Answer with JSON only, no other text."
    fmt = f",\n        responseFormat => {sql_str(response_schema(flows, subs))}" if structured else ""
    body = "r.result" if structured else r"regexp_extract(r.result, '(?s)\\{.*\\}', 0)"
    return f"""WITH src AS (
  SELECT c.call_id, c.customer_tier, c.call_transcript
  FROM {s.fqn(CALLS_CLEAN)} c JOIN {s.fqn(EVAL_SAMPLE)} e USING (call_id)
), raw AS (
  SELECT call_id, ai_query({sql_str(endpoint)},
        concat({sql_str(instr)}, '\\nCustomer tier: ', customer_tier,
               '\\nTranscript:\\n', call_transcript){fmt},
        failOnError => false) AS r
  FROM src
), parsed AS (
  SELECT call_id, from_json({body}, '{OUTPUT_DDL}') AS j, r.errorMessage AS error FROM raw
)
SELECT call_id,
       element_at({_map_literal(sub_to_flow)}, j.secondary_category) AS primary_category,
       j.secondary_category,
       least(5, greatest(1, j.urgency_score)) AS urgency_score,
       j.concise_summary,
       coalesce(j.secondary_category IN ({", ".join(sql_str(x) for x in subs)}), false) AS valid,
       error
FROM parsed"""


def comparison_sql(s: Settings) -> str:
    """One row per model: secondary/primary accuracy and valid rate on the eval sample."""
    return f"""SELECT model,
       round(max(CASE WHEN metric = 'secondary_accuracy' THEN value END), 3) AS secondary_accuracy,
       round(max(CASE WHEN metric = 'primary_accuracy' THEN value END), 3) AS primary_accuracy,
       round(max(CASE WHEN metric = 'valid_rate' THEN value END), 3) AS valid_rate,
       max(n) AS calls
FROM {s.fqn(METRICS)}
WHERE metric IN ('secondary_accuracy', 'primary_accuracy', 'valid_rate')
GROUP BY model
ORDER BY secondary_accuracy DESC"""


MASK_FUNCTION = "mask_tier"


def governance_sql(s: Settings, reader_group: str = "account users") -> list[tuple[str, str]]:
    """(what it does, statement) in order: grants, column mask, tags."""
    grp = f"`{reader_group}`"
    mask = s.fqn(MASK_FUNCTION)
    return [
        ("Let readers see the catalog", f"GRANT USE CATALOG ON CATALOG {s.catalog} TO {grp}"),
        ("Let readers query every table in the schema",
         f"GRANT USE SCHEMA, SELECT ON SCHEMA {s.catalog}.{s.schema} TO {grp}"),
        ("Masking rule: only admins see the real customer tier",
         f"CREATE OR REPLACE FUNCTION {mask}(tier STRING) RETURNS STRING "
         "RETURN CASE WHEN is_account_group_member('admins') OR is_member('admins') "
         "THEN tier ELSE 'restricted' END"),
        ("Apply the mask to customer_tier",
         f"ALTER TABLE {s.fqn(CALLS_CLEAN)} ALTER COLUMN customer_tier SET MASK {mask}"),
        ("Tag the silver table (data classification)",
         f"ALTER TABLE {s.fqn(CALLS_CLEAN)} SET TAGS ('layer' = 'silver', 'domain' = 'member_support', "
         "'contains' = 'customer_conversations')"),
    ]  # fmt: skip


def undo_mask_sql(s: Settings) -> str:
    return f"ALTER TABLE {s.fqn(CALLS_CLEAN)} ALTER COLUMN customer_tier DROP MASK"


def app_grants_sql(s: Settings, principal: str) -> list[str]:
    """What the review app's service principal needs: read everything, write its own tables."""
    who = f"`{principal}`"
    return [
        f"GRANT USE CATALOG ON CATALOG {s.catalog} TO {who}",
        f"GRANT USE SCHEMA, SELECT ON SCHEMA {s.catalog}.{s.schema} TO {who}",
        *[f"GRANT MODIFY ON TABLE {s.fqn(t)} TO {who}" for t in (OVERRIDES, TAXONOMY, TAXONOMY_REQUESTS)],
    ]


# ---- review app (all parameterized with :named markers; run through the SQL warehouse) ----


def _latest_overrides(s: Settings) -> str:
    return (f"(SELECT * FROM {s.fqn(OVERRIDES)} QUALIFY row_number() OVER "
            "(PARTITION BY call_id, model ORDER BY overridden_at DESC) = 1)")  # fmt: skip


def models_sql(s: Settings) -> str:
    return f"SELECT DISTINCT model FROM {s.fqn(PREDICTIONS)} ORDER BY model"


def review_query_sql(s: Settings) -> str:
    """One row per scored call of :model: truth, prediction, Jev confidence, latest override."""
    return f"""SELECT p.call_id, c.customer_tier,
       c.primary_category AS true_primary, c.secondary_category AS true_secondary,
       p.primary_category, p.secondary_category, p.urgency_score, p.concise_summary, p.valid,
       v.secondary_ok AS jev_confidence,
       o.secondary_category AS override_secondary, o.reviewer AS override_by,
       c.call_transcript
FROM {s.fqn(PREDICTIONS)} p
JOIN {s.fqn(CALLS_CLEAN)} c ON c.call_id = p.call_id
LEFT JOIN {s.fqn(VALIDATIONS)} v ON v.call_id = p.call_id AND v.model = p.model
LEFT JOIN {_latest_overrides(s)} o ON o.call_id = p.call_id AND o.model = p.model
WHERE p.model = :model
ORDER BY p.call_id"""


def true_counts_sql(s: Settings) -> str:
    return (f"SELECT secondary_category, count(*) AS n FROM {s.fqn(CALLS_CLEAN)} "
            "GROUP BY secondary_category")  # fmt: skip


def scored_true_counts_sql(s: Settings) -> str:
    """True sub-category counts over the calls :model scored (same calls as predicted counts)."""
    return f"""SELECT c.secondary_category, count(*) AS n
FROM {s.fqn(PREDICTIONS)} p JOIN {s.fqn(CALLS_CLEAN)} c ON c.call_id = p.call_id
WHERE p.model = :model
GROUP BY 1"""


def predicted_counts_sql(s: Settings) -> str:
    """Predicted sub-category counts for :model; a supervisor override replaces the prediction."""
    return f"""SELECT coalesce(o.secondary_category, p.secondary_category) AS secondary_category,
       count(*) AS n
FROM {s.fqn(PREDICTIONS)} p
LEFT JOIN {_latest_overrides(s)} o ON o.call_id = p.call_id AND o.model = p.model
WHERE p.model = :model AND (p.valid OR o.secondary_category IS NOT NULL)
GROUP BY 1"""


def override_insert_sql(s: Settings) -> str:
    # INSERT ... SELECT (not VALUES): Spark only allows constant expressions in an inline VALUES table.
    return (f"INSERT INTO {s.fqn(OVERRIDES)} (call_id, model, secondary_category, primary_category, "
            "reviewer, note, overridden_at) SELECT :call_id, :model, :secondary_category, "
            ":primary_category, :reviewer, :note, current_timestamp()")  # fmt: skip


def taxonomy_sql(s: Settings) -> str:
    return f"SELECT * FROM {s.fqn(TAXONOMY)}"


def requests_sql(s: Settings) -> str:
    return f"SELECT * FROM {s.fqn(TAXONOMY_REQUESTS)} ORDER BY requested_at DESC"


def request_insert_sql(s: Settings) -> str:
    """:example_call_ids is a comma-separated string (may be empty)."""
    return (f"INSERT INTO {s.fqn(TAXONOMY_REQUESTS)} (request_id, kind, primary_category, "
            "secondary_category, new_name, description, reason, example_call_ids, requested_by, "
            "requested_at, status) SELECT :request_id, :kind, :primary_category, :secondary_category, "
            ":new_name, :description, :reason, "
            "filter(transform(split(:example_call_ids, ','), x -> trim(x)), x -> x != ''), "
            ":requested_by, current_timestamp(), 'pending'")  # fmt: skip


def request_decide_sql(s: Settings) -> str:
    return (f"UPDATE {s.fqn(TAXONOMY_REQUESTS)} SET status = :status, decided_by = :decided_by, "
            "decided_at = current_timestamp(), decision_note = :note "
            "WHERE request_id = :request_id AND status = 'pending'")  # fmt: skip


TAX_INSERT_COLS = ["version", "primary_category", "secondary_category", "description", "status",
                   "renamed_from", "changed_by"]  # fmt: skip


def taxonomy_insert_sql(s: Settings, rows: list[dict]) -> tuple[str, dict]:
    """Multi-row INSERT of a full taxonomy version, with one named parameter per value."""
    values, params = [], {}
    for i, r in enumerate(rows):
        names = []
        for c in TAX_INSERT_COLS:
            key = f"r{i}_{c}"
            params[key] = None if r.get(c) is None or r.get(c) != r.get(c) else r[c]  # NaN -> NULL
            names.append(f"CAST(:{key} AS INT)" if c == "version" else f":{key}")
        values.append("SELECT " + ", ".join(names) + ", current_timestamp()")
    cols = ", ".join([*TAX_INSERT_COLS, "changed_at"])
    return f"INSERT INTO {s.fqn(TAXONOMY)} ({cols})\n" + "\nUNION ALL ".join(values), params


def is_admin_sql(s: Settings) -> str:
    return f"SELECT count(*) > 0 AS ok FROM {s.fqn(APP_ADMINS)} WHERE lower(email) = lower(:email)"


def needs_fallback(result) -> bool:
    """True if the structured ai_query run mostly failed (e.g. the endpoint rejects responseFormat).

    With failOnError => false such failures come back per row instead of failing the query.
    """
    n = len(result)
    if n == 0:
        return True
    return int(result["valid"].sum()) == 0 or result["error"].notna().sum() / n > 0.5


def best_model_sql(s: Settings) -> str:
    """The scored model with the highest sub-category accuracy (the app's default view)."""
    return f"""SELECT m.model FROM {s.fqn(METRICS)} m
WHERE m.metric = 'secondary_accuracy' AND m.model IN (SELECT DISTINCT model FROM {s.fqn(PREDICTIONS)})
ORDER BY m.value DESC LIMIT 1"""
