# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup and checks (free)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC **Unity Catalog** organizes everything in three levels: `catalog.schema.object`. Our tables live
# MAGIC in `workspace.support`. A **volume** (`workspace.support.landing`) holds raw *files*; a **table**
# MAGIC holds rows. **Foundation Model APIs** are LLMs Databricks hosts for you, called by endpoint name with
# MAGIC no API key. **Secret scopes** keep keys (our Jev key) out of code.
# MAGIC
# MAGIC *Exam:* Governance (8%), Deployment (22%). *Interview line:* "Raw files land in a UC volume,
# MAGIC curated data in UC tables, and credentials in secret scopes. Nothing sensitive lives in code."
# MAGIC
# MAGIC **How to run:** run every cell from top to bottom. Every line should start with `OK`. If one says
# MAGIC `XX`, it tells you what to do.

# COMMAND ----------

# MAGIC %pip install -q httpx==0.28.1

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 - dbutils is provided by Databricks

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))  # the project code in this Git folder
from support_categorizer.settings import notebook_settings

s = notebook_settings(dbutils)  # noqa: F821
print(f"Catalog {s.catalog} | schema {s.schema} | LLM {s.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Run all checks.** Creates the schema and volume if they are missing, lists the AI models, tests the
# MAGIC network, and looks for the Jev key.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from support_categorizer.checks import print_report, run_all

w = WorkspaceClient()
print_report(run_all(spark, w, dbutils, s))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store the Jev key (once)
# MAGIC Only needed if the check above says the Jev key is not stored. **Lesson:** a secret is written once
# MAGIC and then only *read* by code (`dbutils.secrets.get`); Databricks hides its value if you print it.
# MAGIC 1. Run the next cell: a box called **jev_key** appears at the top of the notebook.
# MAGIC 2. Paste your TypeSafe key into that box.
# MAGIC 3. Run the cell after it. It stores the key and removes the box.

# COMMAND ----------

dbutils.widgets.text("jev_key", "")  # noqa: F821

# COMMAND ----------

from support_categorizer.checks import store_jev_key

key = dbutils.widgets.get("jev_key")  # noqa: F821
if not key.strip():  # e.g. during "Run all": nothing to store yet
    print("The jev_key box at the top is empty. Paste your key there, then run just this cell.")
else:
    store_jev_key(w, key)
    dbutils.widgets.remove("jev_key")  # noqa: F821
    print("Jev key stored in secret scope 'support'. Rerun the checks cell: it should say OK.")
