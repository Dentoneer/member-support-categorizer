# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · RAG with Vector Search (free model calls, ~30-45 minutes the first time)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC **RAG (retrieval-augmented generation)** gives the model the knowledge it is missing at question
# MAGIC time. Our model only saw category *names* (`status`, `manage`, `cost` …) and mixed them up. Here every
# MAGIC category gets a **guideline** (what the customer wants, plus hints like "if an order is mentioned, it's
# MAGIC not shipping cost"). The guidelines live in a Delta table; a **Mosaic AI Vector Search Delta Sync
# MAGIC index** embeds them (`databricks-gte-large-en`) and keeps itself in sync with the table. For each call
# MAGIC we retrieve the 8 most similar guidelines and pass them to the DSPy program's `guidelines` input.
# MAGIC The new program is registered as a **challenger**. It becomes **champion** only if it is more
# MAGIC accurate.
# MAGIC
# MAGIC *Exam:* Data Prep (chunking/embeddings), App Development (RAG), Deployment (registry aliases).
# MAGIC *Interview line:* "I indexed the category guidelines with a Delta Sync Vector Search index, retrieved
# MAGIC the top 8 per call into the DSPy signature, measured retrieval recall separately, and promoted the
# MAGIC RAG program over the old champion because it was more accurate on the same test set."
# MAGIC
# MAGIC **How to run:** Run all. The first run waits for the Vector Search endpoint (5-15 min). Reruns are fast.

# COMMAND ----------

# MAGIC %pip install -q dspy==3.4.1 mlflow==3.17.2 httpx==0.28.1 databricks-sdk==0.157.0

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from support_categorizer.settings import (
    CALLS_CLEAN,
    EVAL_SAMPLE,
    EVAL_SEED,
    GUIDELINES,
    GUIDELINES_INDEX,
    experiment_name,
    notebook_settings,
)

s = notebook_settings(dbutils)  # noqa: F821
dbutils.widgets.dropdown("mode", "full", ["full", "index_only"])  # noqa: F821 - the Job uses index_only
MODE = dbutils.widgets.get("mode")  # noqa: F821
print(f"Model {s.llm_endpoint} | mode {MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · Build the guidelines table.** One row per *active* category in the current category list.
# MAGIC Renamed categories keep their original guideline, and categories added in the review app use the
# MAGIC description from their request. Change Data Feed is on, so the index syncs only what changed.

# COMMAND ----------

from support_categorizer.guidelines import (
    ABCD_GUIDELINES_URL,
    GUIDELINES_FILE,
    guideline_rows,
    load_guidelines_file,
)
from support_categorizer.ingest import download_file
from support_categorizer.tables import write_guidelines
from support_categorizer.taxonomy import read_taxonomy

path = download_file(ABCD_GUIDELINES_URL, s.volume_path(GUIDELINES_FILE), 10_000, GUIDELINES_FILE)
tax = read_taxonomy(spark, s)  # noqa: F821
rows = guideline_rows(tax, load_guidelines_file(path))
write_guidelines(spark, s, rows)  # noqa: F821
print(f"{len(rows)} guidelines written to {s.fqn(GUIDELINES)} "
      f"({(rows['source'] == 'abcd').sum()} from ABCD, {(rows['source'] == 'taxonomy').sum()} from the app)")
display(rows.head(8))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · Vector Search endpoint and index.** The endpoint is the serving infrastructure (Free Edition
# MAGIC allows one); the **Delta Sync index** reads the table, embeds `guideline_text` with a Databricks-hosted
# MAGIC embedding model, and is re-synced every time this cell runs. Waiting is normal the first time.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from support_categorizer.retrieval import ensure_endpoint, ensure_index, wait_endpoint, wait_index

w = WorkspaceClient()
index_name = s.fqn(GUIDELINES_INDEX)
endpoint = ensure_endpoint(w)
wait_endpoint(w, endpoint)
print(f"Index: {ensure_index(w, index_name, endpoint, s.fqn(GUIDELINES))}")
wait_index(w, index_name, expected_rows=len(rows), expected_keys=set(rows["secondary_category"]))
print(f"Index {index_name} is ready with {len(rows)} guidelines.")
if MODE == "index_only":
    dbutils.notebook.exit("guidelines table rebuilt and index synced")  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **3 · Is retrieval any good?** Before involving the LLM: for each of the 300 test calls, is the
# MAGIC *true* sub-category among the 8 retrieved guidelines? (*recall@8*). If the right guideline is not
# MAGIC retrieved, the LLM can't use it.

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor

import mlflow
import pandas as pd

from support_categorizer.retrieval import TOP_K, GuidelineRetriever, recall_at_k
from support_categorizer.tables import accuracy_rows, write_metrics, write_predictions
from support_categorizer.taxonomy import active_labels, canonical_truth, tax_version

flows, subflows, sub_to_flow = active_labels(tax)
tv = tax_version(tax)
silver = canonical_truth(spark.table(s.fqn(CALLS_CLEAN)).toPandas(), tax)  # noqa: F821
sample_ids = set(spark.table(s.fqn(EVAL_SAMPLE)).toPandas()["call_id"])  # noqa: F821
sample = silver[silver["call_id"].isin(sample_ids)]
dev = silver[silver["split"] == "dev"].sample(frac=1, random_state=EVAL_SEED)
train = silver[silver["split"] == "train"].sample(n=200, random_state=EVAL_SEED)

retriever = GuidelineRetriever(index_name, k=TOP_K)


errors = []


def labels_for(transcript):
    try:
        return [h["secondary_category"] for h in retriever.hits(transcript)]
    except Exception as e:  # noqa: BLE001
        errors.append(str(e)[:200])
        return []


with ThreadPoolExecutor(max_workers=4) as pool:
    retrieved = list(pool.map(labels_for, sample["call_transcript"]))
if errors:
    print(f"{len(errors)} retrieval queries failed (counted as misses). First error: {errors[0]}")
recall = recall_at_k(retrieved, list(sample["secondary_category"]))
write_metrics(spark, s, [{"model": "vector_search", "metric": f"retrieval_recall_at_{TOP_K}",  # noqa: F821
                          "value": recall, "n": len(sample)}])
print(f"Retrieval recall@{TOP_K}: {recall:.1%} of calls have their true category among the guidelines.")

# COMMAND ----------

# MAGIC %md
# MAGIC **4 · RAG zero-shot.** The same DSPy program as notebook 03, but it now retrieves guidelines for
# MAGIC every call before answering. Same 300 calls.

# COMMAND ----------

from support_categorizer.categorizer import CallCategorizerProgram, configure_lm, predict_rows

user = spark.sql("SELECT current_user() AS u").first().u  # noqa: F821
mlflow.set_experiment(experiment_name(user))
mlflow.dspy.autolog()
lm = configure_lm(s.llm_endpoint)
records = sample.to_dict("records")
with mlflow.start_run(run_name="dspy_rag_zero_shot") as run:
    mlflow.log_params({"endpoint": s.llm_endpoint, "taxonomy_version": tv, "top_k": TOP_K,
                       "index": index_name, "module": "ChainOfThought+retrieval"})
    rag = CallCategorizerProgram(flows, sub_to_flow, retriever=retriever)
    preds = predict_rows(rag, records, "dspy_rag_zero_shot", f"rag-zero-shot/tax{tv}", run.info.run_id, lm=lm)
    result = accuracy_rows(pd.DataFrame(preds), sample, "dspy_rag_zero_shot")
    mlflow.log_metrics({r["metric"]: r["value"] for r in result})
write_predictions(spark, s, preds)  # noqa: F821
write_metrics(spark, s, result)  # noqa: F821
display(pd.DataFrame(result))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **5 · RAG + BootstrapFewShot → challenger → champion?** Optimizes the RAG program on training calls
# MAGIC (the worked examples now include their retrieved guidelines), registers it as the **challenger**
# MAGIC version of `call_categorizer_model`, scores it on the same 300 calls, and promotes it to **champion**
# MAGIC only if it beats the current champion.

# COMMAND ----------

from support_categorizer.categorizer import examples_from, optimize
from support_categorizer.registry import (
    champion_info,
    model_resources,
    promote,
    register_program,
    should_promote,
    tag_score,
)
from support_categorizer.settings import CHALLENGER_ALIAS

with mlflow.start_run(run_name="dspy_rag") as run:
    mlflow.log_params({"endpoint": s.llm_endpoint, "taxonomy_version": tv, "top_k": TOP_K,
                       "optimizer": "BootstrapFewShot"})
    optimized = optimize(CallCategorizerProgram(flows, sub_to_flow, retriever=retriever),
                         examples_from(train), examples_from(dev.head(100)))
    example = {"call_transcript": records[0]["call_transcript"],
               "customer_tier": records[0]["customer_tier"], "guidelines": ""}
    reg = register_program(optimized, s, example, alias=CHALLENGER_ALIAS,
                           resources=model_resources(s.llm_endpoint, index_name),
                           fallback_path=s.volume_path("program_rag.json"))
    version = reg.get("version")
    preds = predict_rows(optimized, records, "dspy_rag", f"{version or 'file'}/tax{tv}", run.info.run_id, lm=lm)
    result = accuracy_rows(pd.DataFrame(preds), sample, "dspy_rag")
    mlflow.log_metrics({r["metric"]: r["value"] for r in result})
write_predictions(spark, s, preds)  # noqa: F821
write_metrics(spark, s, result)  # noqa: F821
display(pd.DataFrame(result))  # noqa: F821

score = next(r["value"] for r in result if r["metric"] == "secondary_accuracy")
old = spark.sql(f"SELECT max(value) AS v FROM {s.fqn('metrics')} "  # noqa: F821
                "WHERE model = 'dspy_optimized' AND metric = 'secondary_accuracy'").first().v
if version:
    tag_score(s, version, score, taxonomy_version=tv)
    current, current_tv = champion_info(s, fallback=old)
    if should_promote(score, current, new_taxonomy=tv, champion_taxonomy=current_tv):
        promote(s, version)
        print(f"Promoted: RAG {score:.1%} beats the champion's {current if current is None else f'{current:.1%}'}.")
    else:
        print(f"Kept the champion ({current:.1%}); the RAG challenger scored {score:.1%}.")
else:
    print("The program was saved as a file (no Unity Catalog registry here), so there is nothing to promote.")

# COMMAND ----------

# MAGIC %md
# MAGIC **6 · Compare every model** on the same 300 calls.

# COMMAND ----------

from support_categorizer.sql import comparison_sql

display(spark.sql(comparison_sql(s)))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC Done. Next: `04_jev_validation` (it now checks the best DSPy model), then `05_batch_ai_query`,
# MAGIC `06_job`, `09_model_serving`.
