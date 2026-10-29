# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Orchestrate with a Databricks Job (free, ~1 minute to set up; the run takes ~20 minutes)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC A **Job** (Databricks *Workflows*, *Jobs & Pipelines* in the left bar) runs notebooks as **tasks** with
# MAGIC **dependencies**, retries, parameters, schedules and alerts, with no human clicking "Run all". Ours
# MAGIC refreshes the pipeline:
# MAGIC `ingest` → (`baseline`, `guidelines_index`, `governance` in parallel) → `batch_ai_query`.
# MAGIC Each task runs on **serverless** compute and gets `catalog`/`schema` as parameters (the same boxes
# MAGIC you see at the top of each notebook).
# MAGIC
# MAGIC *Exam:* Deployment (22%). *Interview line:* "The pipeline is a Databricks Job: a DAG of notebook tasks
# MAGIC with retries and parameters. In an Airflow shop (AWS MWAA) I'd trigger the same job from a DAG with the
# MAGIC Databricks operator, so orchestration and compute stay separate."

# COMMAND ----------

# MAGIC %pip install -q httpx==0.28.1 databricks-sdk==0.157.0

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from support_categorizer.settings import JOB_NAME, notebook_settings

s = notebook_settings(dbutils)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · Create (or update) the job.** It points at the notebooks in *this* Git folder, so after a
# MAGIC **Pull** the job runs the new code. Rerunning this cell updates the same job, never a duplicate.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from support_categorizer.jobs import job_tasks, upsert_job

w = WorkspaceClient()
here = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
notebooks_dir = os.path.dirname(here)
job_id = upsert_job(w, job_tasks(notebooks_dir, s))
print(f"Job: {w.config.host}/jobs/{job_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · Run it once now.** Click the printed link to watch the task graph turn green. You don't have
# MAGIC to wait here. The job keeps running if you close this notebook.

# COMMAND ----------

run = w.jobs.run_now(job_id=job_id)
print(f"Run started: {w.config.host}/jobs/{job_id}/runs/{run.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Schedule it (optional).** In the job page: **Schedules & Triggers → Add trigger → Scheduled**, e.g.
# MAGIC daily at 6:00. We leave it unscheduled so it doesn't use up Free Edition compute. Other trigger types:
# MAGIC **file arrival** (run when new transcripts land in the volume) and **table update**.
# MAGIC
# MAGIC Done. Next: `09_model_serving`.
