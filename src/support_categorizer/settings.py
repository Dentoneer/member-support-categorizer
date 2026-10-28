"""Every name the pipeline uses (catalog, schema, tables, model, secret) lives here."""

from __future__ import annotations

from dataclasses import dataclass, replace

RAW_CALLS = "raw_calls"
CALLS_CLEAN = "calls_clean"
EVAL_SAMPLE = "eval_sample"
PREDICTIONS = "predictions"
VALIDATIONS = "validations"
METRICS = "metrics"
OVERRIDES = "overrides"
TAXONOMY = "taxonomy"
TAXONOMY_REQUESTS = "taxonomy_requests"
VOLUME = "landing"
MODEL_NAME = "call_categorizer_model"
MODEL_ALIAS = "champion"
SECRET_SCOPE = "support"
SECRET_KEY = "typesafe_api_key"

DEFAULT_LLM = "databricks-meta-llama-3-3-70b-instruct"
CANDIDATE_LLMS = [
    "databricks-qwen35-122b-a10b",
    "databricks-llama-4-maverick",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-oss-120b",
]
EVAL_SEED = 20261028
EVAL_N = 300


@dataclass(frozen=True)
class Settings:
    catalog: str = "workspace"
    schema: str = "support"
    llm_endpoint: str = DEFAULT_LLM
    jev_model: str = "jev-1.14.0"

    def fqn(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    def volume_path(self, file: str) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{VOLUME}/{file}"

    @classmethod
    def from_widgets(cls, dbutils) -> Settings:
        """Read the `catalog`, `schema`, `llm_endpoint` widgets when defined, else defaults."""
        s = cls()
        for name in ("catalog", "schema", "llm_endpoint"):
            try:
                value = dbutils.widgets.get(name)
            except Exception:  # widget not defined in this notebook
                continue
            if value and value.strip():
                s = replace(s, **{name: value.strip()})
        return s
