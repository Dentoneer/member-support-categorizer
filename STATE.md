# STATE

_Last updated: 2026-10-28_

## Current status
- **Stage 1 code complete** on branch `feat/stage1` (local): notebooks 00-05 + 07, supervisor app,
  README run guide, STUDY.md. 80+ unit tests pass; Spark code verified in Docker (Spark 4.0 +
  Delta 4.0) including a rehearsal of notebook 01 on the real ABCD file (10,042 calls, 55
  sub-categories, 4 tiers). Notebooks have NOT yet run on Databricks.
- GitHub repo `Dentoneer/member-support-categorizer` not created yet (needs Marco's OK: public repo).

## Recent decisions
- Data: ABCD instead of CFPB (clean labels, call transcripts, membership tier); hand review dropped.
- Two stages: stage 1 = core + Jev + UC governance + app (Marco moved those in); stage 2 = RAG
  (Vector Search over ABCD guidelines), Job (Workflows), Model Serving + inference tables.
- App: categories & volumes, review & override, category change requests with admin approval
  (taxonomy is a versioned table; retire not delete; rename keeps alias).
- Raw subflows normalized onto the 55 ontology subflows (FAQ variants collapsed; status_questions →
  status_active; status_delivery_date → status_delivery_time).

## Open threads / next steps
- Create + push the GitHub repo (Marco OK), CI green, merge.
- Marco runs 00 → 05, 07, deploys the app (README). Watch first live run: DSPy Databricks auth,
  ai_query responseFormat support per endpoint, UC registration on Free Edition, Guidelines judge.
- Stage 2 (Fri-Sat): RAG, Job, Serving.
- Add to sites-hub (controller).
