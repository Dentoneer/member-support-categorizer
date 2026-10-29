# STATE

_Last updated: 2026-10-29 (overnight stage 2 build done; reviewed + fixed; PR open for Marco)_

## Current status
- **Stage 1 merged (PR #1, #2) and run live on Marco's Free Edition (2026-10-28).** 00-05 + 07 ran;
  app `support-review` deployed from the Git folder root with resource `sql-warehouse`, grants done.
- **Live results (300-call eval sample):** baseline TF-IDF .913 sub / .960 cat; DSPy zero-shot
  .693/.753; DSPy optimized (BootstrapFewShot, 4 demos) .693/.757; ai_query .677/.727 (valid 100%).
  Jev: AUROC .784, auto-accept 55.7% of calls at 86.8% accuracy, rest 47.4%. MLflow eval (50 calls):
  exact .50, Jev mean .72, summary_faithful .66. LLM gap = cryptic label names (shipping_issue
  subflows `status/manage/cost/missing` almost never predicted) -> stage 2 RAG with guidelines.
- PR #2 fixes from the live run: 00 Run-all with empty key box, 07 part 4 standalone, README Apps
  steps (+ New -> App; resources under Authorization/Settings), app chart true-vs-predicted from 0.
- Git folder conflict (workspace copy edited by Assistant/Genie) -> fixed by deleting and
  re-creating the Git folder. Rule: never edit in the workspace; only Pull.

- **Stage 2 built overnight (2026-10-29), PR open, NOT run on Databricks yet** (branch `feat/stage2`):
  PII-safe summaries (prompt rule + regex `pii.py` + Guidelines judge; `summary_pii_rate` metric);
  RAG (notebook 08: ABCD guidelines -> `category_guidelines` (CDF) -> Vector Search endpoint
  `support-vs` + Delta Sync index `category_guidelines_index` (gte-large); program retrieves top 8
  itself; recall@8 metric; models `dspy_rag_zero_shot`, `dspy_rag`); champion/challenger aliases
  with score tags (03 and 08 both register `challenger`, promote only if better); Job (notebook 06,
  `support-categorizer-refresh`: ingest -> baseline / guidelines_index / governance -> ai_query);
  Model Serving (notebook 09, `support-categorizer`, scale-to-zero, AI Gateway inference table
  `serving_payload`); app defaults to best model; 04 validates dspy_rag + PII scorers.
  Final review (opus): 0 critical, 5 important — all fixed with tests (VS error handling/NotFound,
  fail fast on failed endpoint/index, wait for renamed keys in the index, champion replaced when
  trained on an older taxonomy version, serving rerun doesn't rebuild). 130 tests + 9 Spark pass.

## Recent decisions
- Data: ABCD instead of CFPB (clean labels, call transcripts, membership tier); hand review dropped.
- Two stages: stage 1 = core + Jev + UC governance + app (Marco moved those in); stage 2 = RAG
  (Vector Search over ABCD guidelines), Job (Workflows), Model Serving + inference tables.
- App: categories & volumes, review & override, category change requests with admin approval
  (taxonomy is a versioned table; retire not delete; rename keeps alias).
- Raw subflows normalized onto the 55 ontology subflows (FAQ variants collapsed; status_questions →
  status_active; status_delivery_date → status_delivery_time).

## Open threads / next steps
- Marco to try the app's review/override and category-request tabs and report.
- Check whether MLflow traces appear (Traces tab was empty early in the 03 run).
- Marco reviews/merges the stage 2 PR, Pulls, runs 03 -> 08 -> 04 -> 05 -> 06 -> 09, redeploys the app.
- Watch on first live run: VS endpoint provisioning time / Free Edition limit, query_index response
  shape, mlflow.dspy log with resources, serving custom model on Free Edition, ai_query on a custom
  endpoint, job notebook paths.
- Add a `protect-main` ruleset like the CFPB repo (PR + `lint-and-test` check) if Marco wants it.
- Local Spark testing: Docker image `msc-spark` (Dockerfile was in the session scratchpad; recreate:
  python:3.12-slim + openjdk-21-jre-headless + pyspark 4.0.1 / delta-spark 4.0.0).
- Add to sites-hub (controller).
