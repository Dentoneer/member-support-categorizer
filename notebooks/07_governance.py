# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Unity Catalog governance (free, ~2 minutes)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC **Unity Catalog** is where access is decided, once, for every tool (SQL, notebooks, apps, models).
# MAGIC - **GRANT** gives a group rights on a catalog, schema or table (*role-based access control*).
# MAGIC - A **column mask** is a SQL function that rewrites a column depending on *who* is asking. Here, only
# MAGIC   admins see the real `customer_tier`; everyone else sees `restricted`.
# MAGIC - **Tags** classify data (layer, domain, sensitivity) so it can be found and governed.
# MAGIC - **Lineage** is recorded automatically: which notebook or query produced which table and column.
# MAGIC
# MAGIC *Exam:* Governance (8%). *Interview line:* "Access is granted to groups in Unity Catalog, sensitive
# MAGIC columns are masked by policy, and lineage shows exactly which job produced each table."

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from support_categorizer.settings import APP_ADMINS, CALLS_CLEAN, notebook_settings

s = notebook_settings(dbutils)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · Apply the rules.** Each line prints `OK` or why it could not run. Free Edition may not allow
# MAGIC every feature; the message then says what the statement would do in production.

# COMMAND ----------

from support_categorizer.sql import governance_sql

for what, stmt in governance_sql(s):
    try:
        spark.sql(stmt)  # noqa: F821
        print(f"OK  {what}")
    except Exception as e:  # noqa: BLE001
        print(f"XX  {what}: not allowed here ({str(e).splitlines()[0][:160]})\n    Statement: {stmt}")

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · See the mask work.** You are an admin, so you still see real tiers. Someone in `account users`
# MAGIC running the same query would see `restricted` (the review app's own identity does too).

# COMMAND ----------

display(spark.sql(f"SELECT call_id, customer_tier FROM {s.fqn(CALLS_CLEAN)} LIMIT 5"))  # noqa: F821
display(spark.sql(f"SHOW GRANTS ON SCHEMA {s.catalog}.{s.schema}"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **3 · Lineage.** Recorded automatically from notebooks 01-05. In the UI: *Catalog* → `workspace` →
# MAGIC `support` → `predictions` → **Lineage** tab → *See lineage graph*. The same data is in a system table:

# COMMAND ----------

try:
    display(spark.sql(f"""
        SELECT source_table_full_name, target_table_full_name, entity_type, max(event_time) AS last_seen
        FROM system.access.table_lineage
        WHERE target_table_full_name LIKE '{s.catalog}.{s.schema}.%'
        GROUP BY ALL ORDER BY last_seen DESC"""))  # noqa: F821
except Exception as e:  # noqa: BLE001
    print(f"The lineage system table is not available here ({str(e).splitlines()[0][:160]}). "
          "Use the Lineage tab in Catalog Explorer instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC **4 · After deploying the review app (README step 4):** give the app's own identity (a *service
# MAGIC principal*) read access, and write access to the three tables it changes. Type the app's name in the
# MAGIC `app_name` box at the top, then run the cell.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))  # so this cell also works when run on its own
from databricks.sdk import WorkspaceClient

from support_categorizer.settings import APP_ADMINS, notebook_settings
from support_categorizer.sql import app_grants_sql

s = notebook_settings(dbutils)  # noqa: F821

dbutils.widgets.text("app_name", "support-review")  # noqa: F821
try:
    app = WorkspaceClient().apps.get(dbutils.widgets.get("app_name"))  # noqa: F821
except Exception:  # noqa: BLE001
    app = None
    print(f"No app named '{dbutils.widgets.get('app_name')}' yet. Create and deploy it first "  # noqa: F821
          "(README step 3), then rerun this cell.")
principal = app.service_principal_client_id if app else None
for stmt in app_grants_sql(s, principal) if principal else []:
    spark.sql(stmt)  # noqa: F821
    print(f"OK  {stmt}")
# You approve category requests in the app: add yourself to the app's admin list (stored in UC, not code).
me = spark.sql("SELECT current_user() AS u").first().u  # noqa: F821
spark.sql(f"MERGE INTO {s.fqn(APP_ADMINS)} t USING (SELECT '{me}' AS email) n ON t.email = n.email "  # noqa: F821
          "WHEN NOT MATCHED THEN INSERT *")
print(f"OK  {me} can approve category requests in the app")

# COMMAND ----------

# MAGIC %md
# MAGIC **Undo the mask (optional).** Run only if you want the real tiers visible to everyone again.

# COMMAND ----------

# from support_categorizer.sql import undo_mask_sql
# spark.sql(undo_mask_sql(s))
