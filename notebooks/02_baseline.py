# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Baseline: TF-IDF + logistic regression (free, ~2 minutes)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC **MLflow experiment tracking**: every training run records its parameters, metrics and the model
# MAGIC itself, so you can compare runs and reload any model later (click *Experiments* in the left bar).
# MAGIC A **baseline** is the cheapest model that could work. Any LLM must beat it to justify its cost. In
# MAGIC the CFPB project this simple model beat Gemini on sub-categories, which is why we always measure it.
# MAGIC
# MAGIC *Exam:* Evaluation (12%). *Interview line:* "Before trusting an LLM I log a classic baseline in
# MAGIC MLflow on the same test set. The LLM has to beat it, or it isn't worth the latency and cost."

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
    experiment_name,
    notebook_settings,
)

s = notebook_settings(dbutils)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **Train, evaluate and log.** Trains on the `train` split, checks on `dev`, and scores the fixed
# MAGIC 300-call test sample (plus the full test split for reference). Everything goes into one MLflow run.

# COMMAND ----------

import mlflow

from support_categorizer.baseline import predict_baseline, train_baseline
from support_categorizer.labels import flow_of
from support_categorizer.tables import accuracy_rows, write_metrics, write_predictions

user = spark.sql("SELECT current_user() AS u").first().u  # noqa: F821
mlflow.set_experiment(experiment_name(user))
silver = spark.table(s.fqn(CALLS_CLEAN)).toPandas()  # noqa: F821
sample_ids = set(spark.table(s.fqn(EVAL_SAMPLE)).toPandas()["call_id"])  # noqa: F821
train = silver[silver["split"] == "train"]
dev = silver[silver["split"] == "dev"]
test = silver[silver["split"] == "test"]
sample = test[test["call_id"].isin(sample_ids)]


def acc(model, df):
    return float((df["secondary_category"] == predict_baseline(model, list(df["call_transcript"]))).mean())


with mlflow.start_run(run_name="tfidf_baseline") as run:
    model = train_baseline(list(train["call_transcript"]), list(train["secondary_category"]))
    mlflow.log_params({"model": "tfidf+logreg", "ngram_range": "1-2", "min_df": 2, "C": 4.0,
                       "train_rows": len(train)})
    metrics = {"dev_secondary_accuracy": acc(model, dev),
               "test_full_secondary_accuracy": acc(model, test),
               "eval_sample_secondary_accuracy": acc(model, sample)}
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, name="model", input_example=list(sample["call_transcript"][:2]))
print({k: round(v, 3) for k, v in metrics.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC **Save predictions to gold.** One row per sample call, model `baseline`. The primary category comes
# MAGIC from the predicted sub-category (each sub-category belongs to exactly one category).

# COMMAND ----------

preds = predict_baseline(model, list(sample["call_transcript"]))
rows = [{"call_id": cid, "model": "baseline", "model_version": "tfidf-1", "run_id": run.info.run_id,
         "primary_category": flow_of(p), "secondary_category": p, "urgency_score": None,
         "concise_summary": None, "valid": True}
        for cid, p in zip(sample["call_id"], preds, strict=True)]
write_predictions(spark, s, rows)  # noqa: F821
import pandas as pd

result = accuracy_rows(pd.DataFrame(rows), sample, "baseline")
write_metrics(spark, s, result)  # noqa: F821
display(pd.DataFrame(result))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC Done. The number to beat is `secondary_accuracy` above. Next: `03_dspy_categorizer`.
