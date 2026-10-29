# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Jev validation + MLflow Evaluation (a few cents, ~10 minutes)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC An **LLM judge** checks another model's answers. **Jev** (TypeSafe AI) reads each call and the
# MAGIC model's answer and returns a *probability* that the category is right and the summary is faithful.
# MAGIC A good judge is **calibrated**: high confidence should mean usually right. That lets you
# MAGIC **auto-accept** confident calls and send the rest to a human (*human-in-the-loop*).
# MAGIC **MLflow Evaluation** (`mlflow.genai.evaluate`) runs judges like Jev (a custom scorer) and built-in
# MAGIC ones (`Guidelines`) over a dataset and shows the results per row in the MLflow UI.
# MAGIC
# MAGIC *Exam:* Evaluation (12%). *Interview line:* "Every prediction gets an independent judge score; above
# MAGIC 0.8 we auto-accept, below it goes to a supervisor queue, and I track how accurate each bucket is."

# COMMAND ----------

# MAGIC %pip install -q dspy==3.4.1 mlflow==3.17.2 httpx==0.28.1

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
from support_categorizer.settings import (
    CALLS_CLEAN,
    EVAL_SAMPLE,
    PREDICTIONS,
    experiment_name,
    notebook_settings,
)

s = notebook_settings(dbutils)  # noqa: F821
# Jev checks the best DSPy model: the RAG one (notebook 08) if it has run, else the stage 1 one.
_have = {r.model for r in spark.sql(f"SELECT DISTINCT model FROM {s.fqn(PREDICTIONS)}").collect()}  # noqa: F821
MODEL = "dspy_rag" if "dspy_rag" in _have else "dspy_optimized"
print(f"Validating model: {MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · Load the predictions to check** (from notebook 03) with their transcripts.

# COMMAND ----------

from support_categorizer.errors import UserFacingError

rows_pdf = spark.sql(f"""
    SELECT p.call_id, p.model, p.primary_category, p.secondary_category, p.concise_summary,
           c.call_transcript, c.customer_tier
    FROM {s.fqn(PREDICTIONS)} p
    JOIN {s.fqn(CALLS_CLEAN)} c USING (call_id)
    JOIN {s.fqn(EVAL_SAMPLE)} e USING (call_id)
    WHERE p.model = '{MODEL}'""").toPandas()  # noqa: F821
if rows_pdf.empty:
    raise UserFacingError(f"No '{MODEL}' predictions yet: run notebook 03 first.")
from support_categorizer.taxonomy import canonical_truth, read_taxonomy

truth = canonical_truth(spark.table(s.fqn(CALLS_CLEAN)).toPandas(), read_taxonomy(spark, s))  # noqa: F821
truth = truth[truth["call_id"].isin(rows_pdf["call_id"])]
print(f"{len(rows_pdf)} predictions to validate.")

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · Ask Jev about every prediction** (one request per call, three questions: right category? right
# MAGIC sub-category? faithful summary?). A call Jev can't answer stays blank and the run continues.

# COMMAND ----------

from support_categorizer.jev import HttpJevClient, get_jev_key, jev_report, validate_rows
from support_categorizer.tables import write_metrics, write_validations

client = HttpJevClient(get_jev_key(dbutils, s))  # noqa: F821
validations = validate_rows(client, rows_pdf.to_dict("records"), s.jev_model)
write_validations(spark, s, validations)  # noqa: F821
import pandas as pd

val_pdf = pd.DataFrame(validations)
print(f"Validated {val_pdf['secondary_ok'].notna().sum()} of {len(val_pdf)} calls.")

# COMMAND ----------

# MAGIC %md
# MAGIC **3 · Is Jev a good judge?** *AUROC* = how well Jev's confidence separates right from wrong answers
# MAGIC (0.5 = coin flip, 1.0 = perfect). Then: if we auto-accepted every call Jev rates above 0.8, how many
# MAGIC calls would skip review, and how accurate would they be?

# COMMAND ----------

rep = jev_report(val_pdf, rows_pdf, truth)


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


print(f"Jev AUROC: {rep['auroc'] if rep['auroc'] is None else round(rep['auroc'], 3)}")
print(f"Auto-accepted (Jev > 0.8): {pct(rep['auto_accept_rate'])} of calls, "
      f"{pct(rep['accuracy_auto_accepted'])} of them correct")
print(f"Sent to a human: {pct(1 - rep['auto_accept_rate'])} of calls, "
      f"{pct(rep['accuracy_sent_to_human'])} of them correct")
from support_categorizer.pii import pii_rate

print(f"Summaries containing personal data (regex check): {pii_rate(rows_pdf['concise_summary']):.0%}")
write_metrics(spark, s, [  # noqa: F821
    {"model": f"jev_on_{MODEL}", "metric": k, "value": float(v), "n": rep["validated_calls"]}
    for k, v in rep.items() if k != "validated_calls" and v is not None])

# COMMAND ----------

# MAGIC %md
# MAGIC **4 · MLflow Evaluation (50 calls).** Five scorers in one evaluation run: an exact-match check
# MAGIC against the true label, **Jev** as a custom scorer, two built-in **Guidelines** judges (summary
# MAGIC faithfulness, and *no personal data*), and a free regex **PII check**. Column masks don't cover
# MAGIC text the LLM writes, so the summary is checked for names, emails, order numbers and tiers. Open the run → *Evaluations*
# MAGIC to see every row with its scores and reasoning.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import Guidelines, scorer

from support_categorizer.jev import jev_scorer
from support_categorizer.pii import pii_findings

user = spark.sql("SELECT current_user() AS u").first().u  # noqa: F821
mlflow.set_experiment(experiment_name(user))
true_sec = dict(zip(truth["call_id"], truth["secondary_category"], strict=True))
data = [{"inputs": {"call_transcript": r.call_transcript, "customer_tier": r.customer_tier},
         "outputs": {"primary_category": r.primary_category, "secondary_category": r.secondary_category,
                     "concise_summary": r.concise_summary or ""},
         "expectations": {"secondary_category": true_sec[r.call_id]}}
        for r in rows_pdf.dropna(subset=["secondary_category"]).head(50).itertuples()]


@scorer
def secondary_correct(outputs, expectations) -> bool:
    return outputs["secondary_category"] == expectations["secondary_category"]


faithful = Guidelines(name="summary_faithful",  # on Databricks the default judge is Databricks-hosted
                      guidelines="The concise_summary only states facts that appear in the call "
                                 "transcript and describes the customer's actual request.")
no_personal_data = Guidelines(name="summary_no_personal_data",
                              guidelines="The concise_summary contains no personal data: no names, email "
                                         "addresses, phone numbers, usernames, order or account numbers, "
                                         "and not the customer's membership level.")


@scorer
def summary_has_no_pii(outputs) -> bool:
    """Code check (regex) - fast and free, complements the LLM judge above."""
    return not pii_findings(outputs.get("concise_summary"))
try:
    with mlflow.start_run(run_name=f"evaluate_{MODEL}"):
        res = mlflow.genai.evaluate(data=data, scorers=[secondary_correct, jev_scorer(client, s.jev_model),
                                                        faithful, no_personal_data, summary_has_no_pii])
    display(pd.DataFrame([res.metrics]).T.rename(columns={0: "value"}))  # noqa: F821
except Exception as e:  # the Jev numbers above are already saved; this part is a bonus
    print(f"MLflow Evaluation could not run here ({str(e)[:300]}). The Jev results above are saved.")

# COMMAND ----------

# MAGIC %md
# MAGIC Done. Next: `05_batch_ai_query`.
