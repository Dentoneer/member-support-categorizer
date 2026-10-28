# STATE

_Last updated: 2026-10-28_

## Current status
- **Stage 1 code complete**, PR #1 (`feat/stage1` → `main`) open with CI green:
  https://github.com/Dentoneer/member-support-categorizer/pull/1. Notebooks 00-05 + 07, supervisor
  app, README run guide, STUDY.md. 83 unit tests + 8 Spark tests (Docker / CI). Notebook 01 rehearsed
  on the real ABCD file (10,042 calls, 55 sub-categories, 4 tiers).
- Final whole-branch review (opus): "ready with fixes" — all 4 fixed (app INSERTs used VALUES with a
  lambda → INSERT ... SELECT; ai_query JSON fallback; canonical labels after renames; 07 without app).
  Deferred minors: stale predictions after a retire; retired-then-re-added name; Streamlit
  use_container_width warnings; ai_query has no per-row retry (documented).
- **Not yet run on Databricks.** Repo public, secret scanning + push protection on (no ruleset yet).

## Recent decisions
- Data: ABCD instead of CFPB (clean labels, call transcripts, membership tier); hand review dropped.
- Two stages: stage 1 = core + Jev + UC governance + app (Marco moved those in); stage 2 = RAG
  (Vector Search over ABCD guidelines), Job (Workflows), Model Serving + inference tables.
- App: categories & volumes, review & override, category change requests with admin approval
  (taxonomy is a versioned table; retire not delete; rename keeps alias).
- Raw subflows normalized onto the 55 ontology subflows (FAQ variants collapsed; status_questions →
  status_active; status_delivery_date → status_delivery_time).

## Open threads / next steps
- Marco merges PR #1, then creates the Git folder in Databricks (README step 1).
- Marco runs 00 → 05, 07, deploys the app (README). Watch first live run: DSPy Databricks auth,
  ai_query responseFormat support per endpoint, UC registration on Free Edition, Guidelines judge.
- Stage 2 (Fri-Sat): RAG, Job, Serving.
- Add a `protect-main` ruleset like the CFPB repo (PR + `lint-and-test` check) if Marco wants it.
- Local Spark testing: Docker image `msc-spark` (Dockerfile was in the session scratchpad; recreate:
  python:3.12-slim + openjdk-21-jre-headless + pyspark 4.0.1 / delta-spark 4.0.0).
- Add to sites-hub (controller).
