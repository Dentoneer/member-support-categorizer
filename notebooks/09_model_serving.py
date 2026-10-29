# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · Model Serving + inference tables (free on Free Edition if allowed; ~20-40 minutes the first time)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC **Model Serving** turns the **champion** version of our Unity Catalog model into a REST endpoint
# MAGIC (`support-categorizer`). Apps, other services or SQL (`ai_query`) can then categorize a call in real
# MAGIC time with the exact same program the batch job uses. Because the model was logged with its
# MAGIC **resources** (the LLM endpoint and the Vector Search index), serving gets credentials for them
# MAGIC automatically. **AI Gateway inference tables** log every request and response to a Delta table
# MAGIC (`serving_payload`) for monitoring, debugging and building new evaluation sets. **Scale to zero**
# MAGIC means it costs nothing while idle.
# MAGIC
# MAGIC *Exam:* Deployment (22%), Governance/Monitoring. *Interview line:* "The champion alias is served on a
# MAGIC scale-to-zero endpoint with an inference table; promoting a new champion is just re-pointing the
# MAGIC endpoint, and the inference table feeds monitoring and the next evaluation set."
# MAGIC
# MAGIC **Free Edition note:** serving custom models may be limited. If a step says it isn't allowed here,
# MAGIC nothing else depends on this notebook. The message explains what would happen in production.

# COMMAND ----------

# MAGIC %pip install -q dspy==3.4.1 mlflow==3.17.2 httpx==0.28.1 databricks-sdk==0.157.0

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from support_categorizer.settings import CALLS_CLEAN, EVAL_SAMPLE, SERVING_ENDPOINT, notebook_settings

s = notebook_settings(dbutils)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · Create (or update) the endpoint with the champion version.** Then wait until it is ready.
# MAGIC Building the container takes 10-30 minutes the first time. If the wait times out, just rerun the cell.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from support_categorizer.serving import champion_version, upsert_endpoint, wait_ready

w = WorkspaceClient()
version = champion_version(s)
try:
    print(f"Endpoint {upsert_endpoint(w, s, version)}: serving champion version {version}.")
    wait_ready(w)
    print(f"'{SERVING_ENDPOINT}' is ready.")
except Exception as e:  # noqa: BLE001 - Free Edition may not allow custom-model serving
    print(f"Model Serving is not available here: {str(e)[:400]}\n\nIn production this endpoint would "
          "serve the champion program over REST, scale to zero when idle, and log every call to the "
          "inference table. The rest of the project does not depend on it.")

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · Call it like any other service** (3 test calls). This is exactly what an app or another system
# MAGIC would send.

# COMMAND ----------

from support_categorizer.serving import serving_payload

rows = spark.sql(f"SELECT c.* FROM {s.fqn(CALLS_CLEAN)} c JOIN {s.fqn(EVAL_SAMPLE)} USING (call_id) LIMIT 3").toPandas()  # noqa: F821
try:
    resp = w.serving_endpoints.query(SERVING_ENDPOINT, dataframe_records=serving_payload(rows.to_dict("records")))
    for truth, pred in zip(rows["secondary_category"], resp.predictions, strict=False):
        print(f"true: {truth:<28} served answer: {pred}")
except Exception as e:  # noqa: BLE001
    print(f"The endpoint did not answer ({str(e)[:300]}). Check step 1's message.")

# COMMAND ----------

# MAGIC %md
# MAGIC **3 · The same endpoint from SQL.** `ai_query` works with *your own* served models too, not just
# MAGIC Foundation Model APIs. Batch jobs and analysts can then use the champion program from SQL.

# COMMAND ----------

q = f"""SELECT call_id, secondary_category AS true_label,
  ai_query('{SERVING_ENDPOINT}',
           named_struct('call_transcript', call_transcript, 'customer_tier', customer_tier, 'guidelines', ''),
           failOnError => false) AS served
FROM {s.fqn(CALLS_CLEAN)} JOIN {s.fqn(EVAL_SAMPLE)} USING (call_id) LIMIT 5"""
print(q)
try:
    display(spark.sql(q))  # noqa: F821
except Exception as e:  # noqa: BLE001
    print(f"ai_query on the served model did not run here ({str(e).splitlines()[0][:300]}).")

# COMMAND ----------

# MAGIC %md
# MAGIC **4 · The inference table.** Every request and response lands in `serving_payload`, with a delay
# MAGIC (up to an hour). Rerun this cell later to see your calls from steps 2 and 3.

# COMMAND ----------

from support_categorizer.serving import INFERENCE_TABLE_PREFIX

table = s.fqn(f"{INFERENCE_TABLE_PREFIX}_payload")
try:
    display(spark.sql(f"SELECT * FROM {table} ORDER BY request_time DESC LIMIT 10"))  # noqa: F821
except Exception as e:  # noqa: BLE001
    print(f"{table} isn't there yet ({str(e).splitlines()[0][:160]}). It appears after the first logged "
          "requests; check back later.")

# COMMAND ----------

# MAGIC %md
# MAGIC Done. The endpoint scales to zero by itself when idle. To remove it: **Serving** → `support-categorizer`
# MAGIC → **⋮ → Delete**.
