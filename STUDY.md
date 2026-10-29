# Study guide: each piece, the exam domain, and how to explain it

Exam: Databricks Generative AI Engineer Associate. Domains: Design 14%, Data Prep 14%, App Development
30%, Deployment 22%, Governance 8%, Evaluation 12%.

## What we built

| Component | Feature | Domain | Interview answer (2-3 sentences) |
|---|---|---|---|
| Landing + bronze | UC **volume**, **Delta** table | Data Prep | "Raw transcripts land as files in a Unity Catalog volume and are loaded unchanged into a bronze Delta table, so we can always rebuild downstream tables from the original." |
| Silver cleaning | **PySpark** built-ins (`filter`, `transform`, `regexp_replace`), no UDFs | Data Prep | "Cleaning runs as native Spark expressions, so it scales and is optimized by Spark; we strip system events, normalize whitespace, cap length and normalize labels." |
| Delta features | `DESCRIBE HISTORY`, time travel, `OPTIMIZE`, `MERGE` | Data Prep | "Every write is versioned; I can audit or roll back with time travel, and gold tables use MERGE so reruns are idempotent." |
| Fixed eval sample | stratified, seeded table | Evaluation | "Every model is scored on the same 300 stratified test calls stored as a table, so comparisons are fair and reproducible." |
| Baseline | scikit-learn + **MLflow tracking** | Evaluation | "I always log a cheap TF-IDF baseline. In a previous project it beat an LLM on fine-grained labels, so the LLM has to earn its cost." |
| Signature | **DSPy Signature** `CallCategorizer` | App Dev | "The task is a typed contract: transcript + tier + guidelines in; category, sub-category, urgency and summary out, with the categories as an enum so the model can't invent labels." |
| Module | `dspy.ChainOfThought` | App Dev | "ChainOfThought adds a reasoning step before the answer; swapping modules doesn't touch the signature." |
| Optimizer | `BootstrapFewShot`, `MIPROv2` | App Dev | "Instead of hand-tuning prompts, DSPy runs the program on training data, keeps traces that score well on my metric, and compiles them into few-shot demos (MIPROv2 also rewrites instructions)." |
| Metric | 1 / 0.5 / 0 partial credit | Evaluation | "The label is a path, so the right sub-category implies the right category; the metric gives half credit when only the top level is right: routed to the right team." |
| Tracing | `mlflow.dspy.autolog()` | Evaluation / Ops | "Every LLM call, prompt and optimizer trial is traced in MLflow, which is how I debug wrong answers." |
| Registry | **UC model registry**, alias `champion` | Deployment | "The compiled program is versioned in Unity Catalog; batch jobs and serving load `@champion`, so promotion is moving an alias." |
| LLMs | **Foundation Model APIs** | Design / Deployment | "Pay-per-token endpoints hosted by Databricks, governed like everything else; I ran a small bake-off to pick the model on accuracy and valid-format rate." |
| Batch | **`ai_query()`** with `responseFormat`, `failOnError => false` | Deployment | "Batch categorization runs in SQL over the Delta table with a JSON-schema response, so analysts can run it and one bad row doesn't fail the job." |
| Judge | **Jev** + `mlflow.genai.evaluate` custom scorer, `Guidelines` judge | Evaluation | "An independent judge scores each prediction; calls above 0.8 are auto-accepted and the rest go to a supervisor queue. I check the judge itself with AUROC." |
| Governance | `GRANT`, **column mask**, tags, lineage | Governance | "Access is granted to groups in UC, the customer tier is masked by a policy function for non-admins, and lineage shows which notebook produced each table." |
| App | **Databricks Apps** (Streamlit), SQL warehouse resource, service principal | Deployment | "Supervisors see volumes per category, review low-confidence calls, override labels, and request taxonomy changes. The app runs as its own service principal with least-privilege grants." |
| Taxonomy as data | versioned Delta table + approval workflow | Design / Governance | "Categories live in a versioned table; changes go through request and approval, nothing is hard-deleted, and each change produces a new model version." |

## Stage 2 additions

| Component | Feature | Domain | Interview answer |
|---|---|---|---|
| PII in generated text | prompt rule + regex scorer + `Guidelines` judge | Governance / Evaluation | "Column masks don't cover what an LLM writes: the first run copied names and tiers into summaries. I added a no-PII rule to the prompt, a regex check and an LLM judge to evaluation, and report a PII rate per model." |
| Guidelines table | Delta table with **Change Data Feed** | Data Prep | "Each category has a guideline (intent plus contrast hints). Categories added through the review app flow into the same table, so retrieval learns new categories without retraining." |
| Retrieval | **Mosaic AI Vector Search**, **Delta Sync index**, managed embeddings (`databricks-gte-large-en`) | App Dev / Data Prep | "A Delta Sync index embeds the guidelines and stays in sync with the table. I query with the customer's own lines and pass the top 8 guidelines into the DSPy signature." |
| Retrieval quality | recall@k | Evaluation | "I measure retrieval separately (is the right guideline in the top 8?) before blaming the LLM." |
| Champion/challenger | UC model **aliases** + version tags | Deployment | "New programs register as `challenger` with their test score; they take `champion` only if they beat it, and serving and batch follow the alias." |
| Resources | `mlflow` model `resources` (LLM endpoint, VS index) | Deployment / Governance | "The model declares what it calls, so Model Serving gets scoped credentials automatically: no keys in code." |
| Job | **Databricks Workflows**: task DAG, retries, parameters, serverless | Deployment | "ingest → baseline / index sync / governance → batch scoring, upserted by name from the Git folder; Airflow (MWAA) could trigger the same job." |
| Serving | **Model Serving**, scale to zero, **AI Gateway inference table** | Deployment / Monitoring | "The champion is a REST endpoint; every request and response is logged to a Delta table for monitoring and future evaluation sets; `ai_query` can call it from SQL too." |

## Talk-only (not built here; how they would fit)

- **AWS S3 + UC external locations.** In production, transcripts land in an S3 bucket registered as a
  UC *external location*; the volume and bronze table point at it. Free Edition uses
  Databricks-managed storage instead.
- **Databricks Workflows vs AWS MWAA (Airflow).** Workflows (Jobs) chain notebooks with retries and
  schedules inside Databricks (stage 2 builds one). MWAA would trigger the same notebooks through the
  Databricks operator when orchestration spans many systems.
- **dbt-databricks vs PySpark.** dbt suits SQL transformations with tests and docs; PySpark suits
  text cleaning like ours. Both write Delta tables in UC.
- **OCR / pytesseract.** For PDF or image attachments: extract text in a bronze step, then treat it
  like any transcript.
- **MCP bridge.** Expose the categorizer as a tool (via a Model Serving endpoint) that agent
  frameworks and CRM assistants call through the Model Context Protocol.
- **AI Gateway guardrails and rate limits**: on the same serving endpoint, add PII and safety
  guardrails and per-user rate limits (built here: the inference table only).

## The story to tell

1. "I first tried an LLM on CFPB consumer complaints. The labels were noisy and a TF-IDF baseline beat
   Gemini on sub-categories, so I learned to measure against a baseline and to check label quality."
2. "I rebuilt it on Databricks with clean-label data: medallion Delta tables, a DSPy program
   optimized against accuracy, traced and registered in Unity Catalog, and batch-scored with ai_query."
3. "Every prediction gets a judge score; confident calls are auto-accepted, the rest go to a review
   app where supervisors override labels and request taxonomy changes through an approval flow."
