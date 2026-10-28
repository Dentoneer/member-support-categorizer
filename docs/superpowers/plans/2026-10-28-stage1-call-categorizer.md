# Member-Support Call Categorizer — Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repo of Databricks notebooks, a small Python package and a Databricks App. Marco runs
them in his Free Edition workspace to categorize ABCD support conversations end to end (ingest,
baseline, DSPy, Jev validation, `ai_query`, UC governance, supervisor app), learning the stack as
he goes.

**Architecture:** The pure logic lives in `src/support_categorizer/`: labels, transcript building,
the DSPy program, the metric, the Jev client, SQL builders and settings. It is unit-tested locally
and in CI. The Spark parts are tested in CI with local PySpark. Notebooks in Databricks `.py` source
format are thin: widgets, a "What you'll learn" box, then calls into the package. Delta tables live
in `workspace.support` (medallion). The DSPy program is registered in Unity Catalog. The App is
Streamlit, reading gold tables through the SQL warehouse.

**Tech Stack:** Python 3.12 (Databricks serverless), PySpark/Delta, DSPy (Databricks provider via
LiteLLM), MLflow (tracing, `mlflow.dspy`, UC registry, `mlflow.genai` evaluation), scikit-learn,
httpx (Jev), Streamlit + `databricks-sql-connector` + `databricks-sdk` (App), pytest, ruff, GitHub
Actions.

**Spec:** `docs/superpowers/specs/2026-10-28-databricks-call-categorizer-design.md`, Stage 1 only.
Stage 2 (RAG, Job, Model Serving) is out of scope, except the hooks the spec requires stage 1 to
include.

## Global Constraints

- Catalog `workspace`, schema `support`, volume `workspace.support.landing`; every name comes from
  `settings.py`, never hard-coded in notebooks.
- Tables: `raw_calls` (bronze), `calls_clean` (silver), `eval_sample`, `predictions`,
  `validations`, `metrics`, `overrides`, `taxonomy`, `taxonomy_requests` (gold); model `workspace.support.call_categorizer_model`,
  alias `champion`.
- `predictions` is keyed `(call_id, model)`. Model names: `baseline`, `dspy_zero_shot`,
  `dspy_optimized`, `ai_query`. Columns: `call_id, model, model_version, run_id,
  primary_category, secondary_category, urgency_score, concise_summary, valid, predicted_at`.
- All writes are idempotent (MERGE on the key, or overwrite of the rows the notebook owns). Rerunning
  any notebook never duplicates rows.
- DSPy Signature `CallCategorizer`. Inputs: `call_transcript`, `customer_tier`, `guidelines`
  (optional, empty string in stage 1). Outputs: `primary_category`, `secondary_category`,
  `urgency_score` (int 1-5), `concise_summary` (≤ 40 words).
- Headline metric: secondary-category accuracy. Primary-category accuracy is reported as partial
  credit. Optimizer metric: 1.0 secondary right, 0.5 only primary right, else 0.
- Every model is scored on the same `eval_sample`: 300 test calls, stratified by
  `secondary_category`, seed `20261028`.
- Jev key only via `dbutils.secrets.get("support", "typesafe_api_key")`. Never in code, notebooks,
  widgets left filled, logs or the repo.
- LLM: Foundation Model API chat endpoints only. Default `databricks-meta-llama-3-3-70b-instruct`.
  Notebook 03 bake-off candidates: `databricks-qwen35-122b-a10b`, `databricks-llama-4-maverick`,
  `databricks-meta-llama-3-3-70b-instruct`, `databricks-gpt-oss-120b`.
- Each notebook opens with a markdown "What you'll learn" box: 2-4 plain sentences, the exam domain
  with its weight, and one interview line. Code comments are succinct.
- Every failure Marco can hit prints a plain-English fix (raise `UserFacingError(message)`), never
  only a raw traceback.
- No company names or branding anywhere. A generic "member support" demo. The raw data is never
  committed.
- **Taxonomy is data, not code:** the active label set is read from table `taxonomy` (latest
  version, `status='active'`). `labels.ONTOLOGY` is only the seed for version 1. Categories are never
  hard-deleted: "delete" = retire (`status='retired'`), "change" = rename with an alias kept in
  `taxonomy.renamed_from`, "new" = add. Changes happen only through an approved request in the App.
- Python ≥ 3.12. `%pip install` in notebooks uses the pins in `requirements-databricks.txt`, then
  `dbutils.library.restartPython()`.

## Review Focus

1. **Rerunning a notebook** (Marco will, after any stop) must leave the table counts unchanged:
   test MERGE idempotence (Task 4).
2. **LLM output outside the label set or unparseable** must become a row with `valid=False` and
   null labels, counted as wrong, never a crash (Task 6).
3. **Rate limit / transient endpoint error mid-evaluation** (Free Edition FMAPI limits): a row is
   retried 3× with backoff, then marked invalid, and the run continues (Task 6).
4. **Jev secret missing or Jev HTTP error:** a plain-English `UserFacingError` naming the one-time
   fix; a single failed call marks that row unvalidated and does not stop the run (Task 8).
5. **Conversation with no member level, only "action" turns, or a very short transcript:** tier
   `"unknown"`; action turns excluded; calls under 3 customer+agent turns dropped in silver (Tasks
   2-3).

6. **Taxonomy change while old predictions exist:** a renamed or retired category must not break
   the App, the metrics or the comparison: old rows map via `renamed_from`, retired labels still
   display (Task 11b).

## File Structure

```
pyproject.toml, requirements-databricks.txt, .gitignore, README.md, STUDY.md, CLAUDE.md, STATE.md
.github/workflows/ci.yml
src/support_categorizer/
  __init__.py
  settings.py      # names of catalog/schema/tables/model/secret; Settings dataclass; fqn()
  errors.py        # UserFacingError
  labels.py        # ONTOLOGY (10 flows -> 55 subflows), normalize_subflow, flow_of
  transcript.py    # build_transcript(turns) pure-python twin of the Spark logic
  ingest.py        # Spark: bronze_from_json, silver_from_bronze
  tables.py        # Spark: merge_into, eval_sample, metrics rows
  baseline.py      # TF-IDF + LR train/predict
  categorizer.py   # DSPy signature/module, metric, predict_rows, bake_off
  registry.py      # log/register/load the DSPy program (UC, fallback to Volume)
  jev.py           # Jev client, questions, validate_rows, mlflow scorer
  sql.py           # ai_query + comparison + governance SQL builders
  taxonomy.py      # versioned taxonomy table, change requests, apply approved changes
  checks.py        # notebook 00 checks (connectivity, endpoints, secret)
  tools/check_notebooks.py
notebooks/  00_setup_and_checks.py 01_ingest.py 02_baseline.py 03_dspy_categorizer.py
            04_jev_validation.py 05_batch_ai_query.py 07_governance.py
app/  app.py  app.yaml  requirements.txt
tests/  one test file per module
```

---

### Task 1: Scaffold, settings, CI, notebook checker

**Files:** Create `pyproject.toml`, `requirements-databricks.txt`, `.gitignore`,
`.github/workflows/ci.yml`, `src/support_categorizer/{__init__,settings,errors}.py`,
`src/support_categorizer/tools/check_notebooks.py`, `tests/test_settings.py`,
`tests/test_check_notebooks.py`.

**Interfaces:**
- Produces: `Settings(catalog="workspace", schema="support", llm_endpoint=DEFAULT_LLM,
  jev_model="jev-1.14.0")` (frozen dataclass); `Settings.fqn(name: str) -> str` returns
  `` `catalog`.`schema`.`name` `` unquoted form `catalog.schema.name`; constants `RAW_CALLS,
  CALLS_CLEAN, EVAL_SAMPLE, PREDICTIONS, VALIDATIONS, METRICS, OVERRIDES, VOLUME="landing",
  MODEL_NAME="call_categorizer_model", MODEL_ALIAS="champion", SECRET_SCOPE="support",
  SECRET_KEY="typesafe_api_key", DEFAULT_LLM, CANDIDATE_LLMS, EVAL_SEED=20261028, EVAL_N=300`;
  `Settings.from_widgets(dbutils) -> Settings` (reads widgets `catalog`, `schema`, `llm_endpoint`
  when present, else defaults); `Settings.volume_path(file: str) -> str` =
  `/Volumes/{catalog}/{schema}/landing/{file}`; `UserFacingError(Exception)`.
- `check_notebooks.main(dir)`: every `from support_categorizer.X import a, b` in `notebooks/*.py` and
  `app/app.py` resolves to a real attribute; every notebook starts with
  `# Databricks notebook source` and its first MAGIC cell contains "What you'll learn". Exit 1 with
  the offending names.

- [ ] **Step 1: Failing tests:** `test_fqn` (`Settings().fqn("raw_calls") ==
  "workspace.support.raw_calls"`); `test_volume_path`; `test_from_widgets_defaults` (a fake dbutils
  whose `widgets.get` raises → defaults); `test_check_notebooks_flags_missing_name` (tmp notebook
  importing `support_categorizer.settings.NOPE` → exit 1); `test_check_notebooks_requires_learn_box`.
- [ ] **Step 2:** Run `pytest -q`, which should FAIL (module missing).
- [ ] **Step 3:** Implement.
  - pyproject: package dir `src`; deps pandas, scikit-learn, httpx, dspy, mlflow. Extras: `dev`
    (pytest, ruff, pyspark, delta-spark) and `app` (streamlit, databricks-sql-connector,
    databricks-sdk).
  - ruff line-length 100, excluding `notebooks` (magics).
  - pytest marker `spark` auto-skipped when `java` isn't on PATH (conftest).
  - CI: Python 3.12, `actions/setup-java` 17, `pip install -e .[dev]`, `ruff check .`, `pytest -q`,
    `python -m support_categorizer.tools.check_notebooks notebooks`. One job named `lint-and-test`.
  - `requirements-databricks.txt`: current PyPI versions of dspy, mlflow, httpx, pinned exactly.
- [ ] **Step 4:** `pytest -q` passes.
- [ ] **Step 5:** Commit `chore: scaffold package, settings, CI, notebook checker`.

### Task 2: Labels and transcript (pure Python)

**Files:** Create `src/support_categorizer/{labels,transcript}.py`, `tests/test_labels.py`,
`tests/test_transcript.py`.

**Interfaces:**
- `ONTOLOGY: dict[str, list[str]]`: the 10 flows → 55 subflows exactly as in ABCD
  `data/ontology.json` `intents.subflows` (copy the lists verbatim).
  - single_item_query: `boots, shirt, jeans, jacket`.
  - storewide_query: `pricing, membership, timing, policy`.
- `SUBFLOWS: list[str]` (sorted, 55); `FLOWS: list[str]` (sorted, 10).
- `flow_of(subflow: str) -> str` (KeyError on unknown).
- `normalize_subflow(flow: str, subflow: str) -> str | None`, in order:
  1. Exact match in ONTOLOGY[flow].
  2. Strip a trailing `_<digits>`.
  3. Then if still unknown, take the part before the first `_` (`jeans_how_4` → `jeans`).
  4. Alias `status_delivery_date` → `status_delivery_time`.
  5. Anything else → `None` (dropped; e.g. `status_questions`, 30 rows).
- `build_transcript(turns: list[tuple[str, str]], max_chars: int = 4000) -> str`.
  - Keep only `agent`/`customer` turns, rendered `Agent: …` / `Customer: …`, one per line.
  - Whitespace collapsed; truncate at `max_chars` on a line boundary.
- `real_turn_count(turns) -> int` (agent + customer turns).

- [ ] **Step 1: Failing tests:**
  - `len(SUBFLOWS) == 55 and len(FLOWS) == 10`.
  - `normalize_subflow("single_item_query","jeans_how_4") == "jeans"`.
  - `normalize_subflow("storewide_query","pricing_3") == "pricing"`.
  - `normalize_subflow("order_issue","status_delivery_date") == "status_delivery_time"`.
  - `normalize_subflow("subscription_inquiry","status_questions") is None`.
  - `normalize_subflow("shipping_issue","cost") == "cost"`.
  - `flow_of("reset_2fa") == "account_access"`.
  - `build_transcript([("agent","Hi!"),("action","Account pulled"),("customer","  need   help ")])
    == "Agent: Hi!\nCustomer: need help"`.
  - The truncation test asserts the result is ≤ max_chars and ends at a full line.
- [ ] **Step 2:** The tests FAIL. **Step 3:** Implement. **Step 4:** The tests PASS.
- [ ] **Step 5:** Commit `feat: ABCD label ontology, normalization, transcript builder`.

### Task 3: Spark ingestion (bronze, silver) + notebooks 00 and 01

**Files:**
- Create `src/support_categorizer/{ingest,checks}.py`, `tests/test_ingest.py` (marker `spark`),
  `tests/test_checks.py`.
- Create `notebooks/00_setup_and_checks.py` and `notebooks/01_ingest.py`.

**Interfaces:**
- `ABCD_URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"`.
- `download_abcd(dest_path: str, url=ABCD_URL) -> str`: writes into the Volume path. On a network
  error, raises `UserFacingError` with the manual-upload click-path (Catalog → workspace → support
  → landing → Upload to volume).
- `bronze_from_json(spark, path: str) -> DataFrame`.
  - The file is `{"train":[...],"dev":[...],"test":[...]}`, and each conversation has `convo_id`,
    `scenario.flow`, `scenario.subflow`, `scenario.personal.member_level`, and `original`
    (`[[speaker, text], ...]`).
  - Read it with Python `json`+`gzip` on the driver: 37 MB, no Spark JSON schema inference.
  - Produce columns `call_id` (string of convo_id), `split`, `turns`
    (`array<struct<speaker:string,text:string>>`), `flow`, `subflow`, `member_level` (null if
    missing), `ingested_at`.
- `silver_from_bronze(bronze: DataFrame) -> DataFrame`, using Spark built-ins only (no Python UDF):
  - `filter` out action turns;
  - `concat_ws("\n", transform(turns, t -> concat(initcap(t.speaker), ': ', t.text)))`;
  - `regexp_replace` to collapse spaces;
  - `substring` to 4000 chars;
  - `customer_tier = coalesce(member_level, 'unknown')`;
  - `secondary_category` via a `create_map` literal of every raw subflow seen → normalized (built
    from `normalize_subflow` on the distinct `(flow, subflow)` pairs collected from bronze);
  - drop rows with null secondary or fewer than 3 real turns;
  - `primary_category = flow`.
  - Columns: `call_id, split, call_transcript, customer_tier, primary_category,
    secondary_category`.
- `checks.run_all(spark, w, dbutils, settings) -> list[tuple[str, bool, str]]`. Each tuple is
  (check name, ok, plain-English message). It covers:
  - schema and volume (created if missing);
  - each `CANDIDATE_LLMS` endpoint exists;
  - GitHub, PyPI and `https://api.typesafe.ai` are reachable (any HTTP status counts as
    reachable);
  - the secret exists. If not, the message gives the one-time setup: run the "Store the Jev key"
    cell, which reads the key from a temporary widget, writes it with
    `w.secrets.put_secret(scope, key, string_value=…)` (creating the scope if needed), then removes
    the widget.

- [ ] **Step 1: Failing tests:**
  - `test_bronze_schema_and_counts`: a tiny fixture JSON with 3 conversations, one missing
    member_level and one using `jeans_how_4`.
  - `test_silver_drops_actions_and_short_calls`:
    - the action text is absent from the transcript;
    - a 2-turn call is dropped;
    - tier `"unknown"`;
    - `jeans_how_4` → `jeans`;
    - `status_questions` dropped.
  - `test_silver_matches_python_twin`: Spark transcript == `build_transcript` for the fixture.
  - `test_checks_report_missing_secret` (fake w/dbutils → ok False, message mentions "Store the Jev
    key").
- [ ] **Step 2:** FAIL. **Step 3:** Implement. Then write the notebooks:
  - **00 (Unity Catalog, secrets and Foundation Model APIs; Governance 8%, Deployment 22%):**
    1. Widgets.
    2. `run_all`, printing a ✓/✗ table.
    3. The optional "Store the Jev key" cell.
  - **01 (Delta, medallion and PySpark; Data Prep 14%):**
    1. Download, unless the file already exists.
    2. Write bronze and silver with `mode("overwrite")` and `overwriteSchema`.
    3. Table and column comments.
    4. `DESCRIBE HISTORY`, a `VERSION AS OF 0` count, and `OPTIMIZE`.
    5. Counts per split and per primary category.
- [ ] **Step 4:** `pytest -q` PASS (spark tests run in CI; locally skipped without Java).
  `check_notebooks` OK.
- [ ] **Step 5:** Commit `feat: bronze/silver ingestion, setup checks, notebooks 00-01`.

### Task 4: Gold tables, eval sample, metrics

**Files:** Create `src/support_categorizer/tables.py`, `tests/test_tables.py` (spark).

**Interfaces:**
- `PRED_COLS` (the Global Constraints list); `VALIDATION_COLS = ["call_id","model",
  "primary_ok","secondary_ok","summary_faithful","validated_at"]` (Jev probabilities, null if
  unvalidated); `OVERRIDE_COLS = ["call_id","model","secondary_category","primary_category",
  "reviewer","note","overridden_at"]`.
- `ensure_gold_tables(spark, settings) -> None`: `CREATE TABLE IF NOT EXISTS` for predictions,
  validations, metrics (`model, metric, value, n, computed_at`) and overrides, with comments.
- `merge_into(spark, df, table: str, keys: list[str]) -> None`: Delta MERGE, update all on match,
  insert otherwise.
- `make_eval_sample(silver_df, n=EVAL_N, seed=EVAL_SEED) -> DataFrame` (`call_id,
  secondary_category`).
  - Test split only, stratified by secondary_category, deterministic for a seed.
  - Proportional allocation, at least 1 per class present.
  - Implemented in pandas on the collected test split (1,004 rows).
- `accuracy_rows(preds_pdf, truth_pdf, model: str) -> list[dict]`.
  - Metrics `secondary_accuracy`, `primary_accuracy`, `valid_rate` over the eval sample.
  - Invalid rows count as wrong.

- [ ] **Step 1: Failing tests:**
  - `test_merge_into_is_idempotent`: merge the same 3 rows twice → count 3; changed value updated.
  - `test_eval_sample_deterministic_and_stratified`: same seed → same ids; every class with ≥1 test
    row is present.
  - `test_accuracy_counts_invalid_as_wrong`.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: gold tables, eval sample, accuracy rows`.

### Task 5: TF-IDF baseline + notebook 02

**Files:** Create `src/support_categorizer/baseline.py`, `tests/test_baseline.py`,
`notebooks/02_baseline.py`.

**Interfaces:**
- `train_baseline(texts: list[str], labels: list[str]) -> Pipeline`
  (TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True) + LogisticRegression(max_iter=1000,
  C=4.0)).
- `predict_baseline(model, texts) -> list[str]` (secondary).

The notebook does the following:
1. Trains on `train` and logs to MLflow (params; metrics on dev and on eval_sample; model via
   `mlflow.sklearn.log_model`).
2. Writes `predictions` rows for the eval sample with model `baseline`, `model_version="tfidf-1"`,
   primary from `flow_of`, urgency/summary null, `valid=True`. Also writes metrics.
3. Also prints full-test-split accuracy.

The learn box covers MLflow tracking and why a baseline matters (Evaluation 12%).

- [ ] **Step 1:** Failing test `test_baseline_learns_toy_labels` (6 toy texts, 2 labels → predicts
  the training labels back).
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: TF-IDF baseline and notebook 02`.

### Task 6: DSPy categorizer, metric, robust prediction, bake-off

**Files:** Create `src/support_categorizer/categorizer.py`, `tests/test_categorizer.py`.

**Interfaces:**
- `make_signature(flows: list[str], subflows: list[str]) -> type[dspy.Signature]`: class named
  `CallCategorizer`.
  - The docstring is the task instruction.
  - Inputs: `call_transcript: str`, `customer_tier: str`, `guidelines: str` (desc: "optional
    category guidelines; may be empty").
  - Outputs: `primary_category: Literal[*flows]`, `secondary_category: Literal[*subflows]`,
    `urgency_score: int` (desc "1 = routine … 5 = urgent"), `concise_summary: str` (desc "≤ 40
    words").
- `CallCategorizerProgram(dspy.Module)` wraps `dspy.ChainOfThought(signature)`.
  - `forward(call_transcript, customer_tier, guidelines="")` returns the prediction.
  - Primary is re-derived with `flow_of(secondary)` when the two disagree (the label is one path).
- `metric(example, pred, trace=None) -> float` (1.0 / 0.5 / 0).
- `configure_lm(endpoint: str) -> dspy.LM`: `dspy.LM(f"databricks/{endpoint}", temperature=0,
  max_tokens=1000)`.
  - In Databricks it authenticates from the notebook context; if LiteLLM needs it, set
    `DATABRICKS_HOST`/`DATABRICKS_TOKEN` from `WorkspaceClient().config`.
  - Verify on the first live run.
- `predict_rows(program, rows: list[dict], model: str, model_version: str, run_id: str,
  retries=3, sleep=time.sleep, threads=4) -> list[dict]`: one PRED_COLS dict per row.
  - Any exception or unparseable or out-of-set output → retried with exponential backoff (1, 2,
    4 s).
  - After the last retry: `valid=False`, labels null.
  - Never raises for a single row.
- `examples_from(pdf) -> list[dspy.Example]` with inputs `call_transcript, customer_tier,
  guidelines`.
- `bake_off(endpoints, dev_rows, n=30) -> pandas.DataFrame` (`endpoint, secondary_accuracy,
  valid_rate, seconds`); an endpoint that errors outright gets a row with `valid_rate=0` and the
  error text.
- `optimize(program, trainset, valset, mode: Literal["bootstrap","mipro"]) -> program`:
  - `BootstrapFewShot(metric, max_bootstrapped_demos=4, max_labeled_demos=4)`;
  - mipro = `MIPROv2(metric, auto="light")`.

- [ ] **Step 1: Failing tests** (all with `dspy.utils.DummyLM`, no network):
  - `test_signature_fields_and_literals`;
  - `test_program_rederives_primary` (DummyLM answers secondary `reset_2fa`, primary
    `shipping_issue` → primary `account_access`);
  - `test_predict_rows_marks_out_of_set_invalid` (DummyLM answers `not_a_label` → `valid False`, no
    raise);
  - `test_predict_rows_retries_then_continues` (a fake program raising twice then answering → valid;
    always raising → invalid, the others still predicted; `sleep` is a no-op);
  - `test_metric_partial_credit`.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: DSPy CallCategorizer program, metric, robust prediction, bake-off`.

### Task 7: Registry + notebook 03

**Files:** Create `src/support_categorizer/registry.py`, `tests/test_registry.py`,
`notebooks/03_dspy_categorizer.py`.

**Interfaces:**
- `register_program(program, settings, input_example: dict) -> dict`:
  - `mlflow.set_registry_uri("databricks-uc")`;
  - `mlflow.dspy.log_model(program, name="model", input_example=input_example,
    registered_model_name=settings.fqn(MODEL_NAME))`;
  - set alias `champion` with `MlflowClient().set_registered_model_alias`;
  - returns `{"path": "uc", "version": v}`.
  - On any registry failure: `program.save(volume_path("program.json"))`, log it as an artifact,
    and return `{"path": "volume", "file": …}`, printing which path was taken.
- `load_program(settings) -> program`: from `models:/{fqn}@champion` via `mlflow.dspy.load_model`,
  else rebuild `CallCategorizerProgram` and `.load(volume file)`.

Notebook 03 (DSPy signatures, modules and optimizers; MLflow tracing; UC model registry; App Dev
30% + Deployment 22%). Cells in order:
1. `%pip install -r ../requirements-databricks.txt`, then restart.
2. `mlflow.dspy.autolog()`; experiment `/Shared/support-categorizer` (or under the user's home
   folder if `/Shared` is not writable).
3. **Bake-off (optional, ~5 min):** 30 dev calls per candidate; picks the best by
   secondary_accuracy among those with `valid_rate ≥ 0.9` and stores it in the widget/setting
   printout (Marco copies the name into the `llm_endpoint` widget; the default stays if skipped).
4. **Zero-shot** on eval_sample → predictions `dspy_zero_shot`.
5. **BootstrapFewShot** (train 200 sampled, val = 100 dev) → eval → `dspy_optimized`.
6. Optional MIPROv2 cell (off by default; flag `RUN_MIPRO = False`).
7. Register and reload, then predict one example.
8. Metrics rows. Print a comparison table of the baseline, zero-shot and optimized models.

- [ ] **Step 1: Failing tests:** `test_register_falls_back_to_volume` (monkeypatch
  `mlflow.dspy.log_model` to raise → returns path "volume", file saved in a tmp dir via patched
  `volume_path`); `test_load_program_from_volume_roundtrip`.
- [ ] **Step 2-4:** FAIL → implement → PASS. `check_notebooks` OK.
- [ ] **Step 5:** Commit `feat: UC registration with volume fallback, notebook 03`.

### Task 8: Jev validation + MLflow Evaluation + notebook 04

**Files:** Create `src/support_categorizer/jev.py`, `tests/test_jev.py`,
`notebooks/04_jev_validation.py`.

**Interfaces:**
- Port from `../cfpb-nlp-classifier/nlp/llm/jev.py`: `JevResponse`, `JevError`,
  `parse_jev_response`, the HTTP client (`POST https://api.typesafe.ai/v1/systemone`, json
  `{"model","state","questions"}`, bearer auth as there), and `FakeJevClient`. Drop the Phase 1
  ledger, budget and cache; keep 429 backoff (3 tries).
- `build_questions(primary: str, secondary: str, summary: str) -> dict[str, dict]`. Three `noul`
  questions with ids `primary_ok`, `secondary_ok`, `summary_faithful`:
  - is this customer's issue in the "<primary>" category;
  - specifically "<secondary>";
  - does the summary "<summary>" faithfully describe the conversation.
  - Labels are rendered human-readable (`_` → space).
- `get_jev_key(dbutils, settings) -> str`: on a missing secret, raise `UserFacingError` pointing to
  notebook 00's "Store the Jev key" cell.
- `validate_rows(client, rows: list[dict], jev_model: str, threads=4) -> list[dict]`
  (VALIDATION_COLS): one call per row, state = transcript. On a per-row `JevError`, the
  probabilities are null and the run continues.
- `jev_report(validations_pdf, preds_pdf, truth_pdf) -> dict`:
  - AUROC of `secondary_ok` vs actual secondary correctness (sklearn; None if one class only);
  - accuracy where `secondary_ok > 0.8` vs ≤ 0.8;
  - share of calls above 0.8 (the "auto-accept rate").
- `jev_scorer(client, jev_model)` returns an `mlflow.genai.scorers.scorer`-decorated function
  `(inputs, outputs) -> Feedback` with value = secondary_ok probability, so `mlflow.genai.evaluate`
  shows Jev next to the built-in `Correctness`/`Guidelines` judges.

Notebook 04 (LLM evaluation, judges, calibration, human-in-the-loop; Evaluation 12%):
1. Validate the `dspy_optimized` predictions on eval_sample → `validations` (MERGE).
2. `jev_report`, printed in plain words.
3. `mlflow.genai.evaluate` on 50 rows with `jev_scorer` plus a built-in judge for summary
   faithfulness, using the chosen FMAPI endpoint as the judge model.
4. Metrics rows.

- [ ] **Step 1: Failing tests:** `test_build_questions_ids_and_text`;
  `test_parse_response_shapes` (port 2 Phase 1 cases); `test_validate_rows_survives_one_error`
  (FakeJevClient raising for one row → that row null, others filled); `test_missing_secret_message`;
  `test_jev_report_auroc_and_split`.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: Jev validator, MLflow evaluation scorer, notebook 04`.

### Task 9: ai_query batch inference + notebook 05

**Files:** Create `src/support_categorizer/sql.py` (ai_query part), `tests/test_sql.py`,
`notebooks/05_batch_ai_query.py`.

**Interfaces:**
- `response_schema(flows, subflows) -> str`: a JSON schema string for `responseFormat`. Object
  with `primary_category` enum, `secondary_category` enum, `urgency_score` integer 1-5,
  `concise_summary` string; all required.
- `ai_query_sql(settings, endpoint: str, structured: bool = True) -> str`. Builds `SELECT call_id,
  ai_query('<endpoint>', concat(<instruction>, '\nTier: ', customer_tier, '\nTranscript:\n',
  call_transcript), responseFormat => '<schema>') AS raw FROM <calls_clean> JOIN <eval_sample>
  USING (call_id)`.
  - When `structured=False`, drop `responseFormat` and add "Answer with JSON only" to the
    instruction.
  - Wrapped in a CTE that `from_json`s `raw` into the 4 fields and sets `valid` = secondary in the
    label set.
  - Single quotes in the instruction and schema are escaped.
- `comparison_sql(settings) -> str`: per model, secondary/primary accuracy and valid rate from
  `predictions` JOIN `calls_clean` on the eval sample.

Notebook 05 (batch inference in SQL, FMAPI vs Model Serving; Deployment 22%):
1. A `%sql` cell shows the query text (rendered by Python and displayed).
2. Python runs it with `spark.sql`. If the endpoint rejects `responseFormat`, it reruns with
   `structured=False` and says so.
3. MERGE into `predictions` as model `ai_query`.
4. A `%sql` comparison table.
5. A markdown cell explaining Model Serving + `ai_query` on a served model (stage 2).

- [ ] **Step 1: Failing tests:**
  - `test_schema_has_all_labels_and_required`;
  - `test_ai_query_sql_escapes_quotes`: an instruction containing `'` appears as `''`;
  - `test_unstructured_variant_has_no_responseFormat`.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: ai_query batch inference SQL, notebook 05`.

### Task 10: Unity Catalog governance + notebook 07

**Files:** Modify `src/support_categorizer/sql.py` (governance builders), `tests/test_sql.py`;
create `notebooks/07_governance.py`.

**Interfaces:**
- `governance_sql(settings, reader_group: str = "account users") -> list[str]`, in order:
  - `GRANT USE CATALOG`;
  - `GRANT USE SCHEMA, SELECT ON SCHEMA … TO \`<group>\``;
  - `CREATE OR REPLACE FUNCTION <fqn mask_tier>(tier STRING) RETURN CASE WHEN
    is_account_group_member('admins') OR is_member('admins') THEN tier ELSE 'restricted' END`;
  - `ALTER TABLE <calls_clean> ALTER COLUMN customer_tier SET MASK <fqn mask_tier>`;
  - `COMMENT ON TABLE` for each table.
- `undo_mask_sql(settings) -> str`.

Notebook 07 (UC governance: RBAC, masking, lineage, comments; Governance 8%):
1. Runs each statement, printing ✓, or the plain error with "Free Edition may not allow X; here is
   what it would do".
2. `SELECT customer_tier` before and after.
3. A query on `system.access.table_lineage` for our tables (skipped if not permitted), plus the
   click-path to the Lineage tab.
4. A last cell with `undo_mask_sql`, commented "run if you want the mask off".

- [ ] **Step 1: Failing tests:** `test_governance_sql_order_and_names` (mask function fqn, group
  quoted with backticks).
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: UC governance statements, notebook 07`.

### Task 11: Supervisor Databricks App

**Files:** Create `app/app.py`, `app/app.yaml`, `app/requirements.txt`; modify
`src/support_categorizer/sql.py` (app queries), `tests/test_sql.py`.

**Interfaces:**
- `review_query_sql(settings, model: str) -> str`: one row per eval-sample call with `call_id`,
  `customer_tier`, true labels, predicted labels, `secondary_ok` (Jev), the latest override if any,
  and transcript.
- `volume_sql(settings, model: str) -> str` (Marco, 2026-10-28: the app shows categories and their
  volumes): per `primary_category`, `secondary_category`, the number of calls by true label across
  all of `calls_clean` (column `calls_all`) and by the selected model's predicted label on its
  scored rows (`calls_predicted`). Latest override wins, labels go through `canonical`, and
  retired categories are flagged. In the App this is the first tab, **"Categories & volumes"**: a
  bar chart of primary categories (drill-down select → bar chart of their secondary categories),
  plus a table with counts, % share and status (active/retired).
- `override_insert_sql(settings) -> str`: a parameterized `INSERT INTO overrides` with named
  markers `:call_id, :model, :secondary_category, :primary_category, :reviewer, :note`;
  `overridden_at = current_timestamp()`.
- App:
  - `app.yaml` runs `streamlit run app.py`; env `DATABRICKS_WAREHOUSE_ID` comes from an app
    resource `sql-warehouse` (`valueFrom`).
  - Connects with `databricks.sql.connect(server_hostname=Config().host,
    http_path=f"/sql/1.0/warehouses/{id}", credentials_provider=lambda: Config().authenticate)`.
  - UI:
    - model selector (default `dspy_optimized`);
    - filters for category and "Jev confidence below 0.8";
    - a table, a detail pane with the transcript;
    - an override form (secondary dropdown from `SUBFLOWS`; primary is derived with `flow_of`; the
      reviewer is the signed-in user from `st.context.headers["X-Forwarded-Email"]`, else
      "unknown").
  - `sys.path` includes `../src`.
  - README section: deploy via Compute → Apps → Create app → custom → source = the Git folder's
    `app/`, add the SQL warehouse resource, Deploy. Grant the app's service principal `SELECT` on
    the schema and `MODIFY` on `overrides` (SQL printed by notebook 07, extra function
    `app_grants_sql(settings, principal)`).

- [ ] **Step 1: Failing tests:** `test_override_insert_uses_named_params`;
  `test_review_query_joins_latest_override`.
- [ ] **Step 2-4:** FAIL → implement → PASS. `check_notebooks` covers `app/app.py`.
- [ ] **Step 5:** Commit `feat: supervisor review Databricks App`.

### Task 11b: Category change requests (Marco, 2026-10-28)

**Files:** Create `src/support_categorizer/taxonomy.py`, `tests/test_taxonomy.py`; modify
`tables.py` (`ensure_gold_tables` creates both tables), `notebooks/01_ingest.py` (seeds version 1),
`categorizer.py` (`make_signature` gets labels from `active_labels`), `sql.py` (`ai_query_sql` and
`response_schema` take the label lists from the caller), `app/app.py` (second tab).

**Interfaces:**
- Table `taxonomy`: `version int, primary_category, secondary_category, description,
  status ('active'|'retired'), renamed_from (nullable), changed_by, changed_at`. Each change writes
  a full new version, so any version can be reproduced.
- Table `taxonomy_requests`: `request_id (uuid), kind ('add'|'rename'|'retire'),
  primary_category, secondary_category, new_name (rename only), description, reason,
  example_call_ids (array<string>), requested_by, requested_at, status
  ('pending'|'approved'|'rejected'), decided_by, decided_at, decision_note`.
- `seed_rows(changed_by) -> list[dict]`: version 1 from ONTOLOGY; the description is
  human-readable from the name until stage 2 fills it from ABCD guidelines.
- `active_labels(tax_pdf) -> tuple[list[str], list[str], dict[str, str]]`: flows, subflows and the
  subflow→flow map of the latest version.
- `validate_request(req: dict, tax_pdf) -> list[str]`: plain-English problems. Covers:
  - add an existing name;
  - rename or retire an unknown or already-retired name;
  - rename to an existing name;
  - empty reason;
  - an add without a description;
  - a primary category not in the active flows (new primary categories are allowed only with
    `kind='add'` and an explicit flag).
- `apply_request(req: dict, tax_pdf, decided_by) -> pandas.DataFrame`: the next version's full
  rows (pure pandas; the caller writes them).
- `canonical(label, tax_pdf) -> str`: follows `renamed_from` chains to the current name, so old
  predictions and overrides display and score under the current name.
- App tab **"Category requests"**:
  - anyone can submit add / change (rename) / retire, with a reason and optional example call ids
    (prefilled from the selected call in the review tab);
  - admins (members of `admins`, checked with `SELECT is_member('admins')` via the warehouse) see
    pending requests and Approve / Reject with a note;
  - Approve writes the new `taxonomy` version and marks the request;
  - after an approval the App shows: "The label set changed. Rerun notebook 03 to retrain and
    register a new model version; until then predictions use the previous version."
- Notebook 03 records the taxonomy version it trained on in `model_version`
  (`"<uc version>/tax<v>"`) and in the MLflow run params.

- [ ] **Step 1: Failing tests:**
  - `test_seed_has_55_active`;
  - `test_validate_rejects_duplicate_add_and_unknown_retire`;
  - `test_apply_rename_keeps_alias_and_new_version`;
  - `test_apply_retire_is_soft`: the row is present with `status='retired'` and absent from
    `active_labels`;
  - `test_canonical_follows_chain`: a → b → c;
  - `test_signature_uses_active_labels`: an added label appears in the Literal; a retired one
    doesn't.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: versioned taxonomy with approved change requests in the App`.

### Task 12: Docs, primer files, GitHub repo

**Files:** Create `README.md`, `STUDY.md`, `CLAUDE.md`, `STATE.md`.

- README: what it is; run guide in Marco's plain style.
  1. Create a Git folder from `https://github.com/Dentoneer/member-support-categorizer`.
  2. Run 00, then store the Jev key, then 01 → 05 and 07.
  3. Deploy the App.
  Include "what to do if a notebook stops" (each `UserFacingError`).
- STUDY.md: per component, the Databricks feature, exam domain and weight, and a 2-3 sentence
  interview answer. Talk-only items: S3 + UC external locations, MWAA/Airflow vs Workflows,
  dbt-databricks vs PySpark, OCR/pytesseract, MCP bridge, and the CFPB Phase 1 story (noisy labels,
  baseline beat the LLM on sub-issue, so the hybrid/RAG next step).
- CLAUDE.md (stable primer) and STATE.md (`_Last updated: 2026-10-28_`, status, open threads =
  stage 2 list).
- [ ] Create the public GitHub repo `Dentoneer/member-support-categorizer`. Enable secret scanning
  and push protection, add the protect-main ruleset matching Phase 1 (PR + `lint-and-test` check;
  admin bypass), and push.
- [ ] Commit `docs: README run guide, STUDY cheat sheet, primers`.
