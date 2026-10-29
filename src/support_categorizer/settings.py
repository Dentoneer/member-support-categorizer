"""Every name the pipeline uses (catalog, schema, tables, model, secret) lives here."""

from __future__ import annotations

import os
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
APP_ADMINS = "app_admins"
GUIDELINES = "category_guidelines"
GUIDELINES_INDEX = "category_guidelines_index"
CHALLENGER_ALIAS = "challenger"
JOB_NAME = "support-categorizer-refresh"
SERVING_ENDPOINT = "support-categorizer"
VOLUME = "landing"
MODEL_NAME = "call_categorizer_model"
MODEL_ALIAS = "champion"
SECRET_SCOPE = "support"
SECRET_KEY = "typesafe_api_key"
CHOSEN_LLM_FILE = "chosen_llm.txt"

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


def notebook_settings(dbutils, chosen_path: str | None = None) -> Settings:
    """Create the standard widgets (if missing) and return the Settings they hold.

    If the llm_endpoint box is left at its default and notebook 03's bake-off saved a winner
    (`chosen_llm.txt` in the volume), that winner is used.
    """
    d = Settings()
    for name, default in (
        ("catalog", d.catalog),
        ("schema", d.schema),
        ("llm_endpoint", d.llm_endpoint),
    ):
        dbutils.widgets.text(name, default)
    s = Settings.from_widgets(dbutils)
    path = chosen_path or s.volume_path(CHOSEN_LLM_FILE)
    if s.llm_endpoint == DEFAULT_LLM and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            chosen = f.read().strip()
        if chosen:
            s = replace(s, llm_endpoint=chosen)
    return s


def save_chosen_llm(s: Settings, endpoint: str, path: str | None = None) -> None:
    with open(path or s.volume_path(CHOSEN_LLM_FILE), "w", encoding="utf-8") as f:
        f.write(endpoint)


def experiment_name(user: str) -> str:
    """MLflow experiment in the user's workspace folder (always writable)."""
    return f"/Users/{user}/support-categorizer"
