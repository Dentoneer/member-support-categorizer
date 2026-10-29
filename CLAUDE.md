# member-support-categorizer

Databricks GenAI pipeline that categorizes member-support calls (ABCD dataset) into category /
sub-category + urgency + summary, validated by Jev, reviewed in a Databricks App. Built 2026-10-28 for
Marco's Databricks GenAI Engineer exam (2026-11-01) and a job interview (2026-11-02) on that stack.
Phase 2 of `../cfpb-nlp-classifier` (that project's lessons: noisy labels; baseline beat the LLM).

## Layout
- `src/support_categorizer/`: all logic (unit-tested). `settings.py` holds every name
  (catalog `workspace`, schema `support`, tables, model, secret scope).
- `notebooks/*.py`: Databricks source-format notebooks, thin: widgets → "What you'll learn" box →
  calls into the package. Each starts by putting `../src` on `sys.path` and pip-installing the pins
  in `requirements-databricks.txt`.
- `app/app.py` + root `app.yaml` + root `requirements.txt`: the Databricks App, deployed from the Git
  folder ROOT (Apps only ship the source folder; the app imports `src/`).
- Spec: `docs/superpowers/specs/2026-10-28-databricks-call-categorizer-design.md`; plan:
  `docs/superpowers/plans/2026-10-28-stage1-call-categorizer.md`.

## Key facts
- Marco's workspace: Databricks **Free Edition** (serverless only, FMAPI endpoints listed in the spec,
  GitHub/PyPI/api.typesafe.ai reachable, Apps allowed).
- Labels: ABCD ontology (10 flows / 55 subflows) is only the seed; the live label set is the
  versioned `taxonomy` table (approve changes in the app → rerun notebook 03).
- Every model is scored on `eval_sample` (300 test calls, seed 20261028). Predictions are keyed
  `(call_id, model)`; all writes are MERGE/overwrite (rerun-safe).
- Jev key only via `dbutils.secrets.get("support", "typesafe_api_key")`. Never in code or chat.
- LLM auth: LiteLLM signs in through the Databricks SDK (don't copy notebook tokens).
- Public repo: no company names, no data, no emails (app admins live in UC table `app_admins`).

- Stage 2: RAG over `category_guidelines` (Vector Search endpoint `support-vs`, index
  `category_guidelines_index`); the DSPy program can carry a picklable `GuidelineRetriever`.
  Registry: new versions register as `challenger` with a `secondary_accuracy` tag and take
  `champion` only if better (`registry.should_promote`). Job `support-categorizer-refresh`
  (notebook 06), serving endpoint `support-categorizer` (notebook 09).

## Conventions
- Marco is non-technical: every failure path raises `UserFacingError` with a plain-English fix.
- Tests: `pytest -q` (Spark tests need Java; locally run them in Docker image `msc-spark`, in CI they
  run on ubuntu). `ruff check .` (line length 110). `python -m support_categorizer.tools.check_notebooks notebooks app`.
