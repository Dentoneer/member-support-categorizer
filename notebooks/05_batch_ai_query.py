# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Batch inference in SQL with `ai_query()` (free, ~5-10 minutes)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC `ai_query()` calls an LLM **from SQL**, one row at a time, over a whole table, with no Python.
# MAGIC Analysts can run batch categorization straight from the **SQL editor** or a scheduled query.
# MAGIC `responseFormat` forces the model to answer in a JSON shape we define (with the allowed categories
# MAGIC as an enum), and `failOnError => false` keeps one bad row from failing the whole query. Here it calls a
# MAGIC **Foundation Model API** endpoint. In production you can also point it at a **Model Serving**
# MAGIC endpoint that hosts our registered DSPy program (stage 2).
# MAGIC
# MAGIC *Exam:* Deployment (22%). *Interview line:* "For batch, I run `ai_query` over the Delta table with a
# MAGIC structured response format, so millions of calls get categorized in SQL with governed outputs."

# COMMAND ----------

# MAGIC %pip install -q dspy==3.4.1 mlflow==3.17.2 httpx==0.28.1

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from support_categorizer.settings import CALLS_CLEAN, EVAL_SAMPLE, notebook_settings

s = notebook_settings(dbutils)  # noqa: F821
print(f"Model: {s.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · The query.** This is the exact SQL that runs next. You could paste it into the SQL editor.

# COMMAND ----------

from support_categorizer.sql import ai_query_sql
from support_categorizer.taxonomy import active_labels, read_taxonomy, tax_version

tax = read_taxonomy(spark, s)  # noqa: F821
flows, subflows, sub_to_flow = active_labels(tax)
query = ai_query_sql(s, s.llm_endpoint, flows, sub_to_flow)
print(query[:1500] + "\n...")

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · Run it over the 300-call eval sample.** If this model doesn't support `responseFormat`, the cell
# MAGIC says so and reruns with a "JSON only" instruction instead.

# COMMAND ----------

try:
    result = spark.sql(query).toPandas()  # noqa: F821
    mode = "structured"
except Exception as e:  # noqa: BLE001
    print(f"responseFormat not accepted by {s.llm_endpoint} ({str(e)[:200]}). Retrying with a JSON-only "
          "instruction.")
    result = spark.sql(ai_query_sql(s, s.llm_endpoint, flows, sub_to_flow, structured=False)).toPandas()  # noqa: F821
    mode = "json-instruction"
errors = result["error"].notna().sum()
print(f"{len(result)} calls categorized ({mode}); {int(result['valid'].sum())} valid answers, "
      f"{errors} row errors.")
display(result.head(10))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **3 · Save to gold and score** (model `ai_query`), then compare every model on the same 300 calls.

# COMMAND ----------

from support_categorizer.sql import comparison_sql
from support_categorizer.tables import accuracy_rows, write_metrics, write_predictions

rows = [{"call_id": r.call_id, "model": "ai_query", "model_version": f"{s.llm_endpoint}/tax{tax_version(tax)}",
         "run_id": mode, "primary_category": r.primary_category if r.valid else None,
         "secondary_category": r.secondary_category if r.valid else None,
         "urgency_score": int(r.urgency_score) if r.valid and r.urgency_score == r.urgency_score else None,
         "concise_summary": r.concise_summary if r.valid else None, "valid": bool(r.valid)}
        for r in result.itertuples()]
write_predictions(spark, s, rows)  # noqa: F821
truth = spark.sql(f"SELECT c.* FROM {s.fqn(CALLS_CLEAN)} c JOIN {s.fqn(EVAL_SAMPLE)} USING (call_id)").toPandas()  # noqa: F821
import pandas as pd

write_metrics(spark, s, accuracy_rows(pd.DataFrame(rows), truth, "ai_query"))  # noqa: F821
display(spark.sql(comparison_sql(s)))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **Model Serving (stage 2, read only).** The DSPy program registered in notebook 03 can be deployed as a
# MAGIC **Model Serving endpoint** (Serving → Create endpoint → our `call_categorizer_model@champion`). Then
# MAGIC `ai_query('<serving endpoint>', ...)` runs *our optimized program* in batch, and **inference tables**
# MAGIC log every request and response for monitoring. Free Edition limits serving endpoints, so this is stage 2.
# MAGIC
# MAGIC Done. Next: `07_governance`, then deploy the review app (see README).
