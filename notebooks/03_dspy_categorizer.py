# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · DSPy categorizer (free model calls, ~20-30 minutes)
# MAGIC
# MAGIC ## What you'll learn
# MAGIC **DSPy** replaces hand-written prompts with code. A **Signature** declares inputs → outputs (here
# MAGIC `call_transcript, customer_tier, guidelines → primary_category, secondary_category, urgency_score,
# MAGIC concise_summary`), a **Module** (`ChainOfThought`) decides *how* the LLM reasons, and an **optimizer**
# MAGIC (`BootstrapFewShot`, `MIPROv2`) improves the prompt automatically against a metric, here accuracy.
# MAGIC **MLflow Tracing** (`mlflow.dspy.autolog()`) records every LLM call. The finished program is
# MAGIC registered in the **Unity Catalog model registry** with the alias `champion`.
# MAGIC
# MAGIC *Exam:* App Development (30%), Deployment (22%). *Interview line:* "I define the task as a DSPy
# MAGIC signature, optimize it against labelled data instead of hand-tuning prompts, trace it with MLflow,
# MAGIC and register the compiled program in Unity Catalog so serving and batch jobs use the same version."

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
    CANDIDATE_LLMS,
    EVAL_SAMPLE,
    EVAL_SEED,
    experiment_name,
    notebook_settings,
    save_chosen_llm,
)

s = notebook_settings(dbutils)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **Load data and the active category list.** The labels come from the `taxonomy` table (not from
# MAGIC code), so categories approved in the review app are picked up the next time this notebook runs.

# COMMAND ----------

import mlflow

from support_categorizer.taxonomy import active_labels, canonical_truth, read_taxonomy, tax_version

tax = read_taxonomy(spark, s)  # noqa: F821
flows, subflows, sub_to_flow = active_labels(tax)
tv = tax_version(tax)
silver = canonical_truth(spark.table(s.fqn(CALLS_CLEAN)).toPandas(), tax)  # noqa: F821
sample_ids = set(spark.table(s.fqn(EVAL_SAMPLE)).toPandas()["call_id"])  # noqa: F821
sample = silver[silver["call_id"].isin(sample_ids)]
dev = silver[silver["split"] == "dev"].sample(frac=1, random_state=EVAL_SEED)
train = silver[silver["split"] == "train"].sample(n=200, random_state=EVAL_SEED)
print(f"Category list v{tv}: {len(flows)} categories, {len(subflows)} sub-categories. "
      f"Eval sample {len(sample)} calls.")
user = spark.sql("SELECT current_user() AS u").first().u  # noqa: F821
mlflow.set_experiment(experiment_name(user))
mlflow.dspy.autolog()  # every LLM call below is traced: open a run, then the "Traces" tab

# COMMAND ----------

# MAGIC %md
# MAGIC **1 · Pick the model (optional, ~5-10 minutes).** Runs the same 30 dev calls through the candidate
# MAGIC models Databricks hosts for free and keeps the most accurate one that answers in the right format.
# MAGIC The winner is saved, so notebooks 04 and 05 use it too. Set `RUN_BAKE_OFF = False` to skip it and
# MAGIC keep the current model.

# COMMAND ----------

import dataclasses

from support_categorizer.categorizer import CallCategorizerProgram, bake_off, configure_lm, pick_endpoint

RUN_BAKE_OFF = True
if RUN_BAKE_OFF:
    with mlflow.start_run(run_name="bake_off"):
        bake = bake_off(CANDIDATE_LLMS, lambda: CallCategorizerProgram(flows, sub_to_flow), dev, n=30)
        display(bake)  # noqa: F821
        chosen = pick_endpoint(bake, s.llm_endpoint)
        mlflow.log_param("chosen_endpoint", chosen)
    save_chosen_llm(s, chosen)
    s = dataclasses.replace(s, llm_endpoint=chosen)
print(f"Using model: {s.llm_endpoint}")
lm = configure_lm(s.llm_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC **2 · Zero-shot.** The program with no examples: just the signature and ChainOfThought. Scores the
# MAGIC 300-call eval sample. A call that fails 4 times is recorded as invalid (counted wrong) and the run
# MAGIC continues.

# COMMAND ----------

import pandas as pd

from support_categorizer.categorizer import predict_rows
from support_categorizer.tables import accuracy_rows, write_metrics, write_predictions

rows = sample.to_dict("records")
with mlflow.start_run(run_name="dspy_zero_shot") as run:
    mlflow.log_params({"endpoint": s.llm_endpoint, "taxonomy_version": tv, "module": "ChainOfThought"})
    zero = CallCategorizerProgram(flows, sub_to_flow)
    preds = predict_rows(zero, rows, "dspy_zero_shot", f"zero-shot/tax{tv}", run.info.run_id, lm=lm)
    result = accuracy_rows(pd.DataFrame(preds), sample, "dspy_zero_shot")
    mlflow.log_metrics({r["metric"]: r["value"] for r in result})
write_predictions(spark, s, preds)  # noqa: F821
write_metrics(spark, s, result)  # noqa: F821
display(pd.DataFrame(result))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **3 · Optimize with BootstrapFewShot (~5 minutes).** DSPy runs the program on training calls, keeps
# MAGIC the runs that got the right answer (judged by our metric), and puts up to 4 of them into the prompt
# MAGIC as worked examples. Then it registers the optimized program in Unity Catalog and scores it on the same
# MAGIC 300 calls.

# COMMAND ----------

from support_categorizer.categorizer import examples_from, optimize
from support_categorizer.registry import register_program

RUN_MIPRO = False  # True = also try MIPROv2 (rewrites the instructions too; ~20-40 extra minutes)
with mlflow.start_run(run_name="dspy_optimized") as run:
    mlflow.log_params({"endpoint": s.llm_endpoint, "taxonomy_version": tv,
                       "optimizer": "MIPROv2-light" if RUN_MIPRO else "BootstrapFewShot"})
    trainset, valset = examples_from(train), examples_from(dev.head(100))
    optimized = optimize(CallCategorizerProgram(flows, sub_to_flow), trainset, valset,
                         mode="mipro" if RUN_MIPRO else "bootstrap")
    example = {"call_transcript": rows[0]["call_transcript"], "customer_tier": rows[0]["customer_tier"],
               "guidelines": ""}
    reg = register_program(optimized, s, example)
    version = f"{reg.get('version', 'file')}/tax{tv}"
    preds = predict_rows(optimized, rows, "dspy_optimized", version, run.info.run_id, lm=lm)
    result = accuracy_rows(pd.DataFrame(preds), sample, "dspy_optimized")
    mlflow.log_metrics({r["metric"]: r["value"] for r in result})
write_predictions(spark, s, preds)  # noqa: F821
write_metrics(spark, s, result)  # noqa: F821
display(pd.DataFrame(result))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **4 · Prove the round trip.** Load the champion back from Unity Catalog (or the saved file) and
# MAGIC categorize one call.

# COMMAND ----------

import dspy

from support_categorizer.registry import load_program

champion = load_program(s, flows, sub_to_flow)
with dspy.context(lm=lm):
    p = champion(call_transcript=rows[0]["call_transcript"], customer_tier=rows[0]["customer_tier"])
print(f"True: {rows[0]['secondary_category']} | predicted: {p.secondary_category} | "
      f"urgency {p.urgency_score} | {p.concise_summary}")

# COMMAND ----------

# MAGIC %md
# MAGIC **5 · Compare.** All models so far, on the same 300 calls.

# COMMAND ----------

q = f"SELECT model, metric, round(value, 3) AS value, n FROM {s.fqn('metrics')} ORDER BY metric, value DESC"
display(spark.sql(q))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC Done. Next: `04_jev_validation`.
