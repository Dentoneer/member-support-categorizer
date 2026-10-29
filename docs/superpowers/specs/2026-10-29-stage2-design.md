# Stage 2: PII-safe summaries, RAG with Vector Search, Job, Model Serving: design

_2026-10-29. Builds on `2026-10-28-databricks-call-categorizer-design.md` (stage 2 list). Marco asked
for stage 2 to be built overnight without check-ins and delivered as a PR for his review. Calls below
marked "(Claude's call)" were not discussed with him._

## Why (from the first live run, 2026-10-28)

On the 300-call eval sample, TF-IDF scored 91.3% (sub-category) while DSPy zero-shot / optimized /
ai_query scored 69.3% / 69.3% / 67.7%.
- **Cryptic label names.** The model only sees names like `status`, `manage`, `cost`, `missing`, and
  confuses `shipping_issue/status` with `manage_account/status_shipping_question`. Shipping got 10
  predictions for 32 true calls; manage_account got 47 for 20.
- **PII leak.** The app showed summaries like "Bronze customer Rodriguez Domingo…". The LLM copies the
  customer's name and membership tier into free text, which gets around the `customer_tier` column
  mask.

## 1. PII-safe summaries
- The DSPy instruction and the `ai_query` instruction both say: the summary must not contain names,
  email addresses, phone numbers, order/account IDs, usernames or the membership level.
- New `pii.py` holds regex checks for:
  - membership words next to customer or member;
  - emails;
  - phone numbers;
  - 6 or more consecutive digits;
  - "customer/member/guest Firstname Lastname".
- `pii_rate` is reported as a metric per model.
- Notebook 04 adds a code scorer `summary_has_no_pii` and a `Guidelines` judge for "no personal data".
- Interview point: column masks don't cover generated text, so the prompt carries a PII rule and
  evaluation checks for PII.

## 2. RAG with Mosaic AI Vector Search
- **Knowledge.** ABCD `data/guidelines.json` gives, per subflow, an `instructions[0]` sentence that
  describes the customer's intent, often with contrast hints (for example "If an 'order' is
  mentioned, 'shipping cost' is probably not the right flow").
  - Display names map to ontology names through an explicit table (same order as the ontology;
    tested).
- **Table `category_guidelines`.** One row per *active* category in the latest taxonomy version, with
  columns `secondary_category` (PK), `primary_category`, `guideline_text`, `source` (`abcd` or
  `taxonomy`). Change Data Feed is on.
  - The text is the ABCD guideline for that category, found through the rename chain.
  - A category added in the app uses its taxonomy description instead.
  - This means the app's category requests flow into retrieval.
- **Index.** A Delta Sync index `category_guidelines_index` on the one Vector Search endpoint
  `support-vs`. It is TRIGGERED, embeds `guideline_text` with `databricks-gte-large-en`, and is synced
  after each rebuild.
- **Retrieval.**
  - **Query:** the transcript's customer lines (first 1,000 characters) (Claude's call). Customers
    state the issue early; agent boilerplate adds noise.
  - **Hits:** k = 8 (Claude's call; recall-oriented, since 55 short guidelines are cheap to read).
  - **Format:** retrieved guidelines go into the existing `guidelines` input as
    `- <secondary> (<primary>): <text>`. The model may still answer any label: retrieval informs, it
    does not restrict.
- **Evaluation.** Notebook 08 reports retrieval recall@k (true label among the hits) on the eval
  sample. It runs DSPy zero-shot + BootstrapFewShot *with* guidelines (models `dspy_rag_zero_shot`,
  `dspy_rag`) on the same sample and writes predictions and metrics.
- **Program.** `CallCategorizerProgram` gets an optional retriever.
  - When `guidelines` is empty and a retriever is attached, it retrieves by itself.
  - That way the registered model works the same in batch, serving and the app.
  - The retriever is picklable: index name and k only, with the client created lazily.
- **Registry: champion/challenger.**
  - The RAG program is registered as a new version of `call_categorizer_model` with alias
    `challenger`. It is logged with `resources` (the LLM serving endpoint and the Vector Search index)
    so Model Serving can pass credentials through automatically.
  - If its secondary accuracy beats the current champion's (stored as a model-version tag), it takes
    `champion` (Claude's call).

## 3. Job (Databricks Workflows), notebook 06
- Creates or updates (idempotent by name) the job `support-categorizer-refresh` from the Git-folder
  notebook paths.
- Tasks: `ingest` (01), then in parallel `baseline` (02) and `guidelines` (08 in `index_only` mode:
  rebuild the guidelines table and sync the index), then `batch_ai_query` (05).
- Serverless compute (no cluster spec), `max_retries=1`. Catalog and schema are passed as base
  parameters. Runs once now and prints the run URL. It also prints how to add a schedule (not
  scheduled by default, to save Free Edition compute; Claude's call).

## 4. Model Serving + inference tables, notebook 09
- Creates the endpoint `support-categorizer`, or updates it to the champion version.
  - Small CPU workload, scale-to-zero.
  - AI Gateway inference table on in `workspace.support` (prefix `serving`).
- Waits until it is ready, up to 30 minutes. Sends 3 calls through `serving_endpoints.query`. Shows
  an `ai_query('support-categorizer', ...)` SQL example. Reads the inference table once it appears
  (it can lag by up to an hour).
- Free Edition may refuse custom-model serving. Every step catches errors and prints a plain-English
  explanation plus what the step does in production. Nothing else depends on this notebook.

## 5. Everything else
- `ai_query` (05) gets the PII rule.
- The app's model selector defaults to the model with the best `secondary_accuracy` in `metrics`.
- Notebook 04 validates the best DSPy model (`dspy_rag` if present).
- README run order for stage 2: rerun 03 (new prompt), then 08, 04, 05, 06, 09. STUDY.md gains RAG,
  Vector Search, champion/challenger, Jobs, Serving and AI Gateway inference tables, and the PII
  lesson. STATE updated.

## Non-goals
- `vector_search()` inside `ai_query` SQL.
- Reranking.
- Agent Framework / MCP.
- Scheduling the job.
- Deleting old predictions after a retire (deferred minor from stage 1).
