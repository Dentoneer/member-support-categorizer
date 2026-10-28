# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest: bronze → silver (free, ~3 minutes)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC The **medallion architecture**: *bronze* keeps the raw conversations exactly as they arrived (so you
# MAGIC can always rebuild), *silver* holds the cleaned transcripts models read. Both are **Delta tables**:
# MAGIC Parquet files plus a transaction log, which gives you history, **time travel** and safe overwrites.
# MAGIC The cleaning uses **PySpark** built-in functions (no Python loops), so it scales to millions of rows.
# MAGIC
# MAGIC *Exam:* Data Prep (14%). *Interview line:* "I land raw transcripts in bronze, clean them with
# MAGIC PySpark into silver, and every table is Delta, so I can audit and roll back any change."

# COMMAND ----------

# MAGIC %pip install -q httpx==0.28.1

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from support_categorizer.settings import CALLS_CLEAN, RAW_CALLS, notebook_settings

s = notebook_settings(dbutils)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · Land the raw file in the volume.** Downloads the ABCD support conversations (37 MB) once.
# MAGIC In production this is where call transcripts arrive from S3.

# COMMAND ----------

from support_categorizer.ingest import ABCD_FILE, download_abcd

path = download_abcd(s.volume_path(ABCD_FILE))

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · Bronze.** One row per conversation, untouched: the turns, the true category (`flow`), the
# MAGIC sub-category (`subflow`) and the customer's membership level.

# COMMAND ----------

from support_categorizer.ingest import bronze_from_json

bronze = bronze_from_json(spark, path)  # noqa: F821
(bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(s.fqn(RAW_CALLS)))
spark.sql(f"COMMENT ON TABLE {s.fqn(RAW_CALLS)} IS "  # noqa: F821
          "'Bronze: raw ABCD support conversations, one row per call, as ingested.'")
print(f"{s.fqn(RAW_CALLS)}: {spark.table(s.fqn(RAW_CALLS)).count():,} calls")  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **3 · Silver.** PySpark cleaning: drop system "action" lines, label each line `Customer:` /
# MAGIC `Agent:`, collapse spaces, cap at 4,000 characters, map raw sub-categories onto the 55 official
# MAGIC ones, and drop calls with fewer than 3 real turns. Membership level becomes `customer_tier`.

# COMMAND ----------

from support_categorizer.ingest import silver_from_bronze

silver = silver_from_bronze(spark.table(s.fqn(RAW_CALLS)))  # noqa: F821
(silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(s.fqn(CALLS_CLEAN)))
for stmt in [
    f"COMMENT ON TABLE {s.fqn(CALLS_CLEAN)} IS 'Silver: cleaned call transcripts with true labels.'",
    f"ALTER TABLE {s.fqn(CALLS_CLEAN)} ALTER COLUMN call_transcript "
    "COMMENT 'Customer/Agent lines, actions removed, whitespace normalized, max 4000 chars'",
    f"ALTER TABLE {s.fqn(CALLS_CLEAN)} ALTER COLUMN customer_tier "
    "COMMENT 'Membership level: guest/bronze/silver/gold, or unknown'",
    f"ALTER TABLE {s.fqn(CALLS_CLEAN)} ALTER COLUMN secondary_category "
    "COMMENT 'True sub-category (one of 55); implies primary_category'",
]:
    spark.sql(stmt)  # noqa: F821
display(spark.table(s.fqn(CALLS_CLEAN)).limit(5))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **4 · Look at the data.** Calls per split, and per category.

# COMMAND ----------

display(spark.sql(f"SELECT split, count(*) AS calls FROM {s.fqn(CALLS_CLEAN)} GROUP BY split"))  # noqa: F821
display(spark.sql(f"""SELECT primary_category, count(DISTINCT secondary_category) AS sub_categories,
                      count(*) AS calls FROM {s.fqn(CALLS_CLEAN)} GROUP BY 1 ORDER BY calls DESC"""))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **5 · Delta features (the exam asks about these).** `DESCRIBE HISTORY` shows every write; `VERSION
# MAGIC AS OF` reads an old version (time travel); `OPTIMIZE` compacts small files for faster reads.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {s.fqn(CALLS_CLEAN)}").select("version", "timestamp", "operation"))  # noqa: F821
first = spark.sql(f"SELECT count(*) AS n FROM {s.fqn(CALLS_CLEAN)} VERSION AS OF 0").first().n  # noqa: F821
print(f"Rows in the very first version (time travel): {first:,}")
display(spark.sql(f"OPTIMIZE {s.fqn(CALLS_CLEAN)}"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC Done. Next: `02_baseline`.
