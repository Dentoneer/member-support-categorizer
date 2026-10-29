# Member-support call categorizer on Databricks

A working, end-to-end GenAI pipeline built on the Databricks stack. It reads customer-support
conversations and predicts a **category** and **sub-category**, an **urgency score (1-5)** and a
**short summary**. An independent judge (**Jev**) validates each answer, and a **supervisor app**
shows category volumes, lets people correct answers, and handles requests to change the category
list.

| Piece | Databricks feature |
|---|---|
| Raw → clean data | Delta Lake medallion (bronze → silver → gold), PySpark, UC volumes |
| Categorizer | DSPy signature + ChainOfThought, optimized with BootstrapFewShot / MIPROv2 |
| LLMs | Foundation Model APIs (Databricks-hosted Llama / Qwen / GPT-OSS, no API key) |
| Tracking | MLflow experiments + **MLflow Tracing** (`mlflow.dspy.autolog()`) |
| Model registry | Unity Catalog model `workspace.support.call_categorizer_model@champion` |
| Validation | Jev (TypeSafe AI) + **MLflow Evaluation** (`mlflow.genai.evaluate`) |
| Batch inference | `ai_query()` in Databricks SQL with a structured response format |
| Governance | Unity Catalog grants, column mask, tags, lineage; PII checks on generated summaries |
| RAG | Mosaic AI Vector Search Delta Sync index over category guidelines |
| Orchestration | Databricks Job (serverless task DAG) |
| Serving | Model Serving endpoint for the champion, AI Gateway inference table |
| App | Databricks App (Streamlit) on the SQL warehouse |

Data: [ABCD](https://github.com/asappresearch/abcd) (ASAPP, MIT licence), about 10,000 synthetic
customer-agent chats with exact labels (10 categories, 55 sub-categories) and the customer's membership
tier. A generic demo; no company data or branding.

`STUDY.md` explains every piece for the exam and for interviews.

## Run it (Databricks Free Edition)

**You need:** a Databricks Free Edition workspace and a Jev key from console.typesafe.ai (optional;
only notebook 04 uses it). Everything else is free.

### 1. Get the code into Databricks (once, 1 minute)
Left bar → **Workspace** → your folder → **Create** → **Git folder** → paste
`https://github.com/Dentoneer/member-support-categorizer` → **Create Git folder**.
Later, to get updates: open the folder → branch name at the top → **Pull**.

### 2. Run the notebooks in order
Open each one in `notebooks/` and click **Run all** at the top (compute: *Serverless*, the default).

| # | Notebook | Time | What you should see |
|---|---|---|---|
| 00 | `00_setup_and_checks` | 1 min | every line starts with `OK`. If the Jev key line says `XX`, do the "Store the Jev key" steps at the bottom of the notebook, then rerun the checks cell |
| 01 | `01_ingest` | 3 min | 10,042 calls; tables `raw_calls`, `calls_clean`, `eval_sample`, `taxonomy` |
| 02 | `02_baseline` | 2 min | the baseline's `secondary_accuracy` (the number to beat) |
| 03 | `03_dspy_categorizer` | 20-30 min | bake-off table, then zero-shot and optimized accuracy; "Registered ... version 1" |
| 04 | `04_jev_validation` | 10 min | Jev AUROC and auto-accept numbers |
| 05 | `05_batch_ai_query` | 5-10 min | the comparison table of every model |
| 07 | `07_governance` | 2 min | `OK` lines for grants, mask and tags. **Stop before part 4** until the app exists |

Settings (catalog, schema, model) are the boxes at the top of each notebook; leave them as they are.

### 2b. Stage 2: RAG, PII-safe summaries, Job, Serving
After a **Pull** of the Git folder, run these in order (Run all):

| # | Notebook | Time | What you should see |
|---|---|---|---|
| 03 | `03_dspy_categorizer` (again: the prompt now forbids personal data in summaries) | 20-30 min | new zero-shot/optimized numbers; "is now the champion" or "Kept the current champion" |
| 08 | `08_rag_vector_search` | 30-45 min first time | guidelines table; Vector Search endpoint + index ready; retrieval recall@8; RAG accuracy; promotion message |
| 04 | `04_jev_validation` | 10 min | "Validating model: dspy_rag", Jev numbers, the PII line |
| 05 | `05_batch_ai_query` | 5-10 min | comparison table with every model |
| 06 | `06_job` | 1 min (+20 min run) | a job link; open it and watch the task graph turn green |
| 09 | `09_model_serving` | 20-40 min | endpoint ready + 3 served answers, or a plain message that Free Edition doesn't allow it |

Then redeploy the app (it now opens on the most accurate model).

### 3. Deploy the review app (once, ~5 minutes)
1. **+ New** → **App** (Apps also live in the app switcher, the grid icon at the top right) →
   **Create a custom app**. Name: `support-review`.
2. Add the SQL warehouse as an **app resource** (on the create screen, or later under the app's
   *Authorization* / *Settings*): **SQL warehouse** → Serverless Starter Warehouse, permission
   *Can use*, resource key **`sql-warehouse`** (exactly).
3. When the app's compute is running, click **Deploy** and choose the **root of the Git folder**
   (`/Workspace/Users/<you>/member-support-categorizer`, not the `app` subfolder).
4. Back in notebook `07_governance`, run **part 4**. It gives the app access to the tables and makes
   you an approver.
5. Open the app's URL. Tabs: **Categories & volumes**, **Review & override**, **Category requests**.
   Stop the app (⋮ → Stop) when you're not using it.

### What to do if something stops
- **"Could not download the ABCD data"**: follow the message (download the file on your laptop and
  upload it to the `landing` volume), then rerun the cell.
- **"The Jev key is not stored"**: notebook 00, "Store the Jev key" cells.
- **"No trained categorizer found"** or **"No 'dspy_optimized' predictions yet"**: run notebook 03 first.
- **A call "gave up after 4 tries"** during 03/05: normal under Free Edition rate limits; that call
  counts as wrong and the run continues. Many of them means a busy endpoint: rerun later.
- **`XX ... not allowed here`** in 07: Free Edition doesn't offer that governance feature; the line
  says what it would do in production.
- **App says "Could not read the tables"**: run 01-05 and notebook 07 part 4, then reload the app.
- **Category list changed in the app**: rerun notebook 03 to retrain and register a new model version.

## Developing
`pip install -e ".[dev,app]"`, then `pytest -q` and `ruff check .`. Spark tests need Java (they run
in CI). Design: `docs/superpowers/specs/`, plan: `docs/superpowers/plans/`.
