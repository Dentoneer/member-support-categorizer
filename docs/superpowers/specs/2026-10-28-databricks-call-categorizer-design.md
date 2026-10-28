# Member-support call categorizer on Databricks — design

_2026-10-28. Phase 2 of the CFPB classifier work (repo `cfpb-nlp-classifier`), rebuilt on the
Databricks stack._

## Why

Marco has a job interview on **Monday 2026-11-02** with a company whose data stack is Databricks
on AWS (Delta Lake, Unity Catalog, PySpark, DSPy, MLflow, Model Serving, `ai_query`), and is sitting
the **Databricks Generative AI Engineer Associate** exam on **Sunday 2026-11-01**. He wants to be
fluent in that stack by building one real, working pipeline end to end, and to **learn while
building**: every piece should teach him something he can explain in the interview.

The pipeline categorizes member-support conversations, following the brief Marco pasted:

1. PySpark Delta ingestion of raw call transcripts (`<catalog>.support.raw_calls`) and cleaning.
2. A DSPy Signature `CallCategorizer`: inputs `call_transcript`, `customer_tier`; outputs
   `primary_category`, `secondary_category`, `urgency_score`, `concise_summary`.
3. A DSPy `ChainOfThought` module compiled with BootstrapFewShot / MIPROv2 against an accuracy metric.
4. MLflow tracing with `mlflow.dspy.autolog()`; the program logged to Unity Catalog as
   `<catalog>.support.call_categorizer_model`.
5. A Databricks SQL batch-inference example with `ai_query()` over the Delta table.

Marco added: **Jev (TypeSafe AI) as a validator**, as in Phase 1.

### Lessons carried over from Phase 1 (CFPB)

- Noisy labels made the LLM look bad and could only be fixed with hours of hand review. Phase 2
  uses a dataset whose labels are **correct by construction**, so no hand review is needed.
- A cheap classic baseline (TF-IDF + logistic regression) beat Gemini on fine-grained labels.
  Always report the LLM next to a baseline.
- The label is one path (secondary category implies primary), so the headline metric is
  **secondary-category accuracy**; primary-category accuracy is reported as "routed to the right
  team" partial credit (Marco's point: a correct sub-category implies the category).

## Goals

- A repo of Databricks notebooks Marco runs in order in his **Databricks Free Edition** workspace,
  covering the five brief items plus a baseline, Jev validation and a Job that chains the steps.
- Each notebook opens with a short **"What you'll learn"** box: the concept in 2-4 plain sentences,
  which exam domain it belongs to (Design 14%, Data Prep 14%, App Development 30%, Deployment 22%,
  Governance 8%, Evaluation 12%), and one interview talking point. Code comments stay succinct.
- A `STUDY.md` cheat sheet: each component, the Databricks feature it uses, the exam domain, and
  a 2-3 sentence interview answer ("how would you…").
- Real numbers at the end: baseline vs zero-shot DSPy vs optimized DSPy accuracy on the test split,
  and how well Jev's confidence separates right from wrong predictions.

## Non-goals

- Hand validation (dropped by Marco for time).
- Gemini or any paid LLM other than Jev (cents).
- Using any real company data, name or branding. The repo is public; it is a generic
  "member support" demo. **Anti-goal:** anything that reads as if it were the employer's own code.
- Bringing your own S3 bucket (Free Edition uses Databricks-managed storage; the story is "in
  production this table lives on S3 behind a Unity Catalog external location").
- Production SLAs, CI deployment to Databricks, dashboards.

## Constraints

- **Databricks Free Edition:** serverless compute only; one 2X-Small SQL warehouse; limited
  model-serving endpoints (CPU only, some models unavailable); one Vector Search endpoint;
  outbound internet limited to an unpublished allow-list of domains; max 5 concurrent job tasks.
  Commercial use not allowed (this is a learning project).
- **Deadline:** notebooks ready for Marco to run on **Thursday 2026-10-29**; his running time should
  fit in about 4-6 hours across Thursday-Saturday alongside exam study (_my call; Marco did not
  give an hour budget_).
- **Marco is non-technical and wants zero surprises:** every notebook runs top to bottom with no
  edits except a few clearly marked settings at the top; every cell that can fail says in plain
  English what to do. Each step is rerunnable without duplicating data.
- **Secrets:** the Jev key never appears in a notebook, repo or chat. It lives in a Databricks
  secret scope (`support`, key `typesafe_api_key`), read with `dbutils.secrets.get`.
- Python: whatever Databricks serverless ships; extra packages installed per notebook with
  `%pip install` (pinned versions) followed by `dbutils.library.restartPython()`.

## Data: ABCD

**Action-Based Conversations Dataset** (ASAPP, NAACL 2021): 10,042 synthetic but realistic
customer-agent chat conversations for an online clothing retailer, MIT-licensed, at
`github.com/asappresearch/abcd` (`data/abcd_v1.1.json.gz`, already split into train/dev/test).
Per conversation it has the dialogue turns (speaker + text), and a scenario with **flow**
(~10 high-level intents) and **subflow** (55 specific intents), generated from the scenario, so the
labels are correct by construction. The customer's **membership level** (guest / bronze / silver /
gold) is a scenario slot.

Mapping: `primary_category` = flow, `secondary_category` = subflow, `customer_tier` = membership
level. `urgency_score` (1-5) and `concise_summary` have no ground truth; they are generated and
checked by Jev / spot-reading only.

The builder must confirm the exact JSON field names from the file (papers disagree slightly on the
subflow list). If a conversation has no membership level, `customer_tier = "unknown"`.

**Getting the file in:** notebook 01 first tries to download it from GitHub into a Unity Catalog
**Volume** (`<catalog>.support.landing`). If Free Edition's network blocks GitHub, it stops with
instructions to download the file on the laptop and upload it through the Catalog UI ("Upload to
volume"), a one-time 1-minute step. The raw file is never committed to the repo.

## Architecture (medallion, all in Unity Catalog)

Settings at the top of each notebook (widgets): `catalog` (default `workspace`, which Free Edition
provides), `schema` (default `support`), `llm_endpoint` (default chosen in notebook 00).

| Layer | Object | Contents |
|---|---|---|
| Landing | Volume `support.landing` | `abcd_v1.1.json.gz` |
| Bronze | `support.raw_calls` | one row per conversation: `call_id`, `split`, raw turns (array<struct<speaker,text>>), `flow`, `subflow`, `member_level`, `ingested_at` |
| Silver | `support.calls_clean` | `call_id`, `split`, `call_transcript` (turns joined as `Customer: …` / `Agent: …` lines, whitespace normalized, length-capped), `customer_tier`, `primary_category`, `secondary_category` |
| Gold | `support.predictions` | `call_id`, `model` (baseline / dspy_zero_shot / dspy_optimized / ai_query), all four outputs, `run_id`, `predicted_at` |
| Gold | `support.validations` | `call_id`, `model`, Jev probabilities per question |
| Gold | `support.metrics` | model, metric, value, split |
| Registry | model `support.call_categorizer_model` | the compiled DSPy program (UC-registered, alias `champion`) |

Tables get UC comments and column comments (governance talking point). Writes are idempotent
(`MERGE` on `call_id, model` or overwrite of the partition the notebook owns).

## Notebooks (run in order)

**00_setup_and_checks** (free). Creates schema and volume if missing. Lists the serving endpoints
the workspace has and picks the default LLM: a Databricks-hosted chat model (Foundation Model
APIs, e.g. a Llama 3.x 70B instruct endpoint), preferring the strongest available. Prints a
**connectivity check**: can it reach GitHub (data download), PyPI (pip), and the Jev API? Checks
the Jev secret exists. Each check prints OK or the plain-English fix. Teaches: Unity Catalog
three-level namespace, volumes vs tables, Foundation Model APIs, secret scopes.

**01_ingest_bronze_silver.** Load the JSON into bronze with PySpark (`explode`, struct access),
then build silver with PySpark cleaning (concat turns, `regexp_replace` whitespace, drop empty or
very short conversations, cap length). Shows Delta features briefly: `DESCRIBE HISTORY`, time
travel (`VERSION AS OF`), `OPTIMIZE`. Prints row counts per split and per primary category.
Teaches: Delta Lake, medallion architecture, PySpark transformations (Data Prep domain).

**02_baseline.** TF-IDF + logistic regression (scikit-learn on a pandas copy of silver; the data
is small) predicting `secondary_category` from the transcript; primary derived from the
subflow→flow map. Logged to MLflow (params, metrics, model). Writes predictions to gold.
Teaches: MLflow experiment tracking; why a baseline matters (Phase 1 story).

**03_dspy_categorizer.** The core of the brief.
- `dspy.LM("databricks/<llm_endpoint>")` (DSPy's Databricks provider, auth from the notebook).
- Signature `CallCategorizer(call_transcript, customer_tier -> primary_category,
  secondary_category, urgency_score, concise_summary)`. Categories are constrained to the label
  set (`Literal[...]` built from silver), urgency is an int 1-5, summary ≤ 40 words.
- Module: `dspy.ChainOfThought(CallCategorizer)`.
- Metric: 1.0 if secondary correct; 0.5 if only primary correct; else 0. (Both reported separately.)
- `mlflow.dspy.autolog()` on, so every call and the optimizer's trials are traced in MLflow.
- Zero-shot evaluation on a fixed sample of the dev split (size chosen so the run takes ~10-15 min
  on Free Edition rate limits; the builder sets it after a timing check, default 200).
- Optimize with `BootstrapFewShot` (train split examples) first; then `MIPROv2(auto="light")` as an
  optional cell if time allows. Evaluate the optimized program on the same test sample as the
  baseline.
- Log the compiled program with `mlflow.dspy.log_model` and register it in UC as
  `<catalog>.support.call_categorizer_model`, alias `champion`. Reload it from UC and run one
  example to prove the round trip.
- Writes zero-shot and optimized predictions to gold.
Teaches: DSPy signatures/modules/optimizers (App Development), MLflow tracing, UC model registry
(Deployment).

**04_jev_validation.** For each optimized-DSPy test prediction, ask Jev three yes/no questions
(is the primary category right for this transcript? is the secondary category right? is the
summary faithful to the transcript?), reusing the Phase 1 Jev client approach (one request per
call, `noul` probabilities). Writes to `support.validations`. Reports: AUROC of Jev's
secondary-category probability against actual correctness, and accuracy among calls Jev marks
confident (> 0.8) vs the rest, i.e. "would Jev let us auto-accept most calls and send the rest to a
human?" Also wraps Jev as an MLflow GenAI custom scorer so the same check appears in an MLflow
evaluation run. **If the connectivity check showed Jev is blocked from Free Edition**, the notebook
says so and the fallback is to run this one notebook in Colab (reads predictions exported from
Databricks as a small CSV, writes validations back as a CSV uploaded to the volume). The builder
decides the cleanest fallback; it must be a few clicks, not engineering for Marco.
Teaches: LLM evaluation, judges, calibration, human-in-the-loop routing (Evaluation domain).

**05_batch_ai_query.** Databricks SQL (`%sql` cells, also runnable in the SQL editor):
`ai_query('<llm_endpoint>', <prompt built with concat()>, responseFormat => <JSON schema>)` over
`calls_clean` for the test split, parsed with `from_json` into the four outputs, written to gold
as model `ai_query`. Then SQL comparing all models' accuracy from `predictions`. A short text cell
explains how the UC-registered DSPy model would instead be deployed on Model Serving and called
with `ai_query('<serving endpoint>', …)`; actually deploying it is optional (Free Edition endpoint
limits). Teaches: batch inference in SQL, Model Serving vs Foundation Model APIs (Deployment).

**06_job.** Creates a Databricks **Job** (Workflows) that chains 01 → 02 → 03 (evaluate-only mode,
reusing the registered model) → 05 as tasks, via the Databricks SDK from the notebook, and runs it
once. Teaches: orchestration and task dependencies (Deployment); the Airflow comparison for the
interview.

## Repo and how code reaches Databricks

New public GitHub repo `member-support-categorizer` (under `Dentoneer`, like Phase 1), cloned into
the workspace with a **Git folder** (Workspace > Create > Git folder). Notebooks are stored as
`.ipynb` (or Databricks `.py` source format, builder's choice) under `notebooks/`. Small shared
code (label map, Jev client, metric) lives in `src/support_categorizer/` and is imported from the
Git folder. Unit tests for pure-Python parts (cleaning helpers, metric, Jev parsing, prompt/schema
building) run locally with pytest and in GitHub Actions; Spark/Databricks parts are verified by
running the notebooks. `CLAUDE.md` + `STATE.md` at the root per Marco's convention; README with a
step-by-step run guide in Marco's plain style.

## Costs

Foundation Model API calls in Free Edition: no charge (rate-limited). Jev: one request per test
call, a few cents in total. No Gemini spend.

## Teaching (how Claude works with Marco during the build)

While Marco runs each notebook, Claude explains each step succinctly (what it is, why it exists,
how it would be asked in the exam or interview) and answers questions. The notebooks' "What you'll
learn" boxes and `STUDY.md` hold the same content so it can be reviewed later.

## Delivery in two stages (Marco, 2026-10-28)

Marco: build the **bare minimum functional product first**, then add the rest as enhancements, but
design stage 1 so stage 2 slots in without rework. (Named "stages" to avoid confusion with the
CFPB project's Phase 1/Phase 2.)

**Stage 1 — core, functional end to end (target: ready Thursday 2026-10-29, may slip to Friday
morning).** Marco moved the App, governance and Jev into stage 1 (2026-10-28):
- **00** setup/checks; **01** bronze→silver ingestion; **02** TF-IDF baseline; **03** DSPy
  categorizer (zero-shot + BootstrapFewShot, MLflow tracing, UC registration); **05** `ai_query`
  batch inference writing gold predictions plus the model comparison.
- **04 Jev validation + MLflow Evaluation:** Jev as validator and as a custom judge in
  `mlflow.evaluate`, plus built-in LLM-judge metrics (summary faithfulness).
- **07 Unity Catalog governance:** `GRANT`s for a reader group, a **column mask** on
  `customer_tier`, table/column comments, and a walkthrough of lineage in Catalog Explorer.
- **App** (`app/`, Streamlit, deployed as a **Databricks App**): supervisors browse predictions with
  Jev confidence, filter by category/confidence, and **override** a category; overrides go to Delta
  table `support.overrides`. Reads through the SQL warehouse. If Free Edition's App limits block
  deployment, the README says so with the click-path that does work.
Result: transcripts in, categorized + summarized + validated calls out, governed in UC, reviewable
in an app, with real accuracy numbers.

**Stage 2 — enhancements (target: Friday-Saturday), in this order:**
1. **RAG with Mosaic AI Vector Search:** ABCD ships the agent guidelines per subflow; put them in a
   Delta table, build a **Delta Sync index** (Databricks-managed embeddings), retrieve the top 5
   guidelines per call and pass them to the DSPy module as context. Evaluated on the same test
   sample as the stage 1 model (does retrieval lift accuracy?).
2. **Job** (notebook 06) chaining the pipeline (Databricks Workflows).
3. **Model Serving + Inference Tables:** deploy the registered DSPy model as an endpoint with
   inference tables on; call it from `ai_query`. Tried on Free Edition; if blocked, the notebook says
   so and explains the production setup.

**Talk-only (in `STUDY.md`, no code):** S3 + UC external locations, AWS MWAA/Airflow vs Workflows,
dbt-databricks vs PySpark, OCR/pytesseract for attachments, MCP bridge to CRM tools.

**Built for stage 2 from day one (stage 1 must include):**
- The DSPy Signature has an optional `guidelines` input (empty string in stage 1), so RAG only fills
  it; same program class, new version in the registry.
- `predictions` carries `model`, `model_version`, `run_id`, `predicted_at`; it is keyed
  `(call_id, model)` so Jev, RAG, serving and app-override rows join on it.
- The App reads only gold tables and the shared settings, so new models (RAG, served) appear in it
  without App changes.
- One shared settings module (`catalog`, `schema`, endpoint names, sample table) used by every
  notebook, so stage 2 notebooks and the App and Job read the same config.
- The fixed `support.eval_sample` table exists from stage 1, so every later model is scored on it.
- Notebooks run headless (no manual edits besides widgets), so the stage 2 Job can call them.

## Workspace facts (Marco's Free Edition, checked 2026-10-28)

- Catalogs: `samples`, `system`, `workspace` -> everything goes in `workspace.support`.
- Chat endpoints (Foundation Model APIs, `llm/v1/chat`): `databricks-gpt-oss-120b`,
  `databricks-gpt-oss-20b`, `databricks-qwen3-next-80b-a3b-instruct`,
  `databricks-qwen35-122b-a10b`, `databricks-gemma-3-12b`, `databricks-meta-llama-3-1-8b-instruct`,
  `databricks-meta-llama-3-3-70b-instruct`, `databricks-llama-4-maverick`,
  `databricks-deepseek-v4-flash-0731`.
- Embedding endpoints: `databricks-gte-large-en`, `databricks-bge-large-en`,
  `databricks-qwen3-embedding-0-6b` (stage 2 RAG uses `databricks-gte-large-en`).
- Reachable from serverless: GitHub raw (200), PyPI (200), `https://api.typesafe.ai` (404 on `/`,
  i.e. reachable). The Jev Colab fallback is not needed.
- Databricks Apps allowed (none exist yet).
- **LLM choice:** notebook 00 (or the top of 03) runs a small free bake-off (~30 dev calls each) over
  `databricks-qwen35-122b-a10b`, `databricks-llama-4-maverick`,
  `databricks-meta-llama-3-3-70b-instruct`, `databricks-gpt-oss-120b`, picks the best secondary
  accuracy that also returns valid structured output, and saves it to the shared settings
  (`llm_endpoint`). Fallback default if the bake-off is skipped: `databricks-meta-llama-3-3-70b-instruct`.

## Decisions from the builder check (Claude's calls, 2026-10-28)

1. **LLM endpoint:** notebook 00 lists the workspace's Foundation Model endpoints and prefers, in
   order, the largest instruct chat model that supports structured output (JSON schema). A weaker
   model is acceptable: this is a learning project, and the numbers are reported as they are. If
   the endpoint rejects `responseFormat`, notebook 05 falls back to prompt-and-parse (`from_json`
   on the model's text, with a JSON-only instruction), and DSPy uses its default ChatAdapter
   (Literal fields still validated by DSPy on the parsed output).
2. **Network:** PyPI is assumed reachable (Free Edition supports `%pip`); notebook 00 verifies it and
   stops with a plain-English message if not. ABCD: GitHub download, else the manual Volume upload
   (no second source). Every failure stops the notebook with the fix, never a raw traceback only.
3. **Model registry:** UC registration with `mlflow.dspy.log_model` is the plan; if registration
   fails on Free Edition, the notebook saves the compiled program (`program.save(...)` JSON) to the
   Volume and logs it as an MLflow artifact, and later notebooks load from there. The notebook says
   which path it took.
4. **Splits and fairness:** the optimizer trains on ABCD **train** and validates on **dev**. All
   final numbers (baseline, zero-shot, optimized, ai_query, Jev) use **one fixed test sample**,
   stratified by `secondary_category` and seeded, stored as table `support.eval_sample`. The
   baseline is also reported on the full test split for reference.
5. **Jev:** reuse the Phase 1 client from `cfpb-nlp-classifier/nlp/llm/jev.py` (endpoint, auth,
   request shape, `noul` parsing; that client ran successfully in Phase 1 notebook 04), trimmed of
   the Phase 1 ledger. If Jev is blocked from Free Edition, the Colab fallback is acceptable.
6. **Job (06):** creates the Job via the Databricks SDK and runs it once in a "smoke" mode (small
   sample) so it stays inside rate limits. If job creation is not allowed, the notebook prints the
   click-path to create it in the UI.
7. **Repo:** Claude creates the public repo `Dentoneer/member-support-categorizer` during the build
   (same owner and protections as Phase 1).

## Left to the builder

Exact package pins; the evaluation sample sizes after a timing check; notebook file format; how
the Jev secret is created with the fewest clicks (e.g. Databricks SDK `secrets.put_secret` from a
one-off cell that reads the key from a widget and then clears it, or the Databricks CLI); the Jev
fallback mechanics; the precise ABCD field names; and whether MIPROv2 runs by default or as an
optional cell.
