"""SQL builders: ai_query batch inference, model comparison, governance, review app queries."""

from __future__ import annotations

import json

from support_categorizer.settings import (
    CALLS_CLEAN,
    EVAL_SAMPLE,
    METRICS,
    OVERRIDES,
    TAXONOMY,
    TAXONOMY_REQUESTS,
    Settings,
)

AI_QUERY_INSTRUCTION = (
    "You categorize member-support calls. Read the call and answer with a JSON object with keys "
    "primary_category, secondary_category (choose from the allowed values), urgency_score (integer "
    "1 = routine to 5 = urgent) and concise_summary (at most 40 words)."
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
            "urgency_score": {"type": "integer", "minimum": 1, "maximum": 5},
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
