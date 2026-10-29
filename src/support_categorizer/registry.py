"""Log + register the DSPy program in Unity Catalog (alias `champion`); Volume file fallback."""

from __future__ import annotations

import os

import mlflow

from support_categorizer.categorizer import CallCategorizerProgram
from support_categorizer.errors import UserFacingError
from support_categorizer.settings import MODEL_ALIAS, MODEL_NAME, Settings

PROGRAM_FILE = "program.json"


def register_program(program, s: Settings, input_example: dict, fallback_path: str | None = None,
                     alias: str = MODEL_ALIAS, resources: list | None = None,
                     score: float | None = None) -> dict:
    """Inside an active MLflow run. Returns {"path": "uc", "version": v} or the Volume fallback.

    `alias` is "champion" (stage 1) or "challenger" (a candidate to compare). `resources` lists what
    the model calls (LLM endpoint, Vector Search index) so Model Serving can pass credentials through.
    `score` (eval-sample secondary accuracy) is stored as a version tag for champion/challenger.
    """
    name = s.fqn(MODEL_NAME)
    try:
        mlflow.set_registry_uri("databricks-uc")
        pkg = os.path.dirname(os.path.abspath(__file__))  # so the model can import our code anywhere
        info = mlflow.dspy.log_model(program, name="model", input_example=input_example,
                                     registered_model_name=name, code_paths=[pkg],
                                     resources=resources)  # fmt: skip
        version = str(info.registered_model_version)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(name, alias, version)
        if score is not None:
            client.set_model_version_tag(name, version, "secondary_accuracy", f"{score:.4f}")
        print(f"Registered {name} version {version} with alias '{alias}' in Unity Catalog.")
        return {"path": "uc", "version": version}
    except Exception as e:  # noqa: BLE001 - Free Edition limits: keep going with a file
        path = fallback_path or s.volume_path(PROGRAM_FILE)
        program.save(path)
        try:
            mlflow.log_artifact(path)
        except Exception:  # noqa: BLE001
            pass
        print(f"Unity Catalog registration was not possible ({str(e)[:160]}). Saved the program to "
              f"{path} instead; later notebooks load it from there.")  # fmt: skip
        return {"path": "volume", "file": path}


def load_program(s: Settings, flows: list[str], sub_to_flow: dict[str, str],
                 fallback_path: str | None = None):
    """The champion from Unity Catalog, else the Volume file rebuilt on the given label set."""
    try:
        mlflow.set_registry_uri("databricks-uc")
        return mlflow.dspy.load_model(f"models:/{s.fqn(MODEL_NAME)}@{MODEL_ALIAS}")
    except Exception as e:  # noqa: BLE001
        path = fallback_path or s.volume_path(PROGRAM_FILE)
        if not os.path.exists(path):
            raise UserFacingError(
                "No trained categorizer found (not in Unity Catalog, no program file). Run notebook 03 "
                f"first. ({str(e)[:120]})"
            ) from None
        prog = CallCategorizerProgram(flows, sub_to_flow)
        prog.load(path)
        return prog


def model_resources(llm_endpoint: str, index_name: str | None = None) -> list:
    """What the registered program calls at inference time (for Model Serving auth passthrough)."""
    from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

    res = [DatabricksServingEndpoint(endpoint_name=llm_endpoint)]
    if index_name:
        res.append(DatabricksVectorSearchIndex(index_name=index_name))
    return res


def should_promote(challenger_score: float, champion_score: float | None) -> bool:
    """The challenger replaces the champion only if it is strictly more accurate."""
    return champion_score is None or challenger_score > champion_score


def promote(s: Settings, version: str) -> None:
    """Move the `champion` alias to `version` (batch jobs and serving follow the alias)."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.MlflowClient().set_registered_model_alias(s.fqn(MODEL_NAME), MODEL_ALIAS, version)
    print(f"Version {version} is now the champion.")


def tag_score(s: Settings, version: str, score: float) -> None:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.MlflowClient().set_model_version_tag(s.fqn(MODEL_NAME), version, "secondary_accuracy",
                                                f"{score:.4f}")  # fmt: skip


def champion_score(s: Settings, fallback: float | None = None) -> float | None:
    """The champion version's stored score; `fallback` (e.g. from the metrics table) if untagged."""
    try:
        mlflow.set_registry_uri("databricks-uc")
        mv = mlflow.MlflowClient().get_model_version_by_alias(s.fqn(MODEL_NAME), MODEL_ALIAS)
        tag = (mv.tags or {}).get("secondary_accuracy")
        return float(tag) if tag is not None else fallback
    except Exception:  # noqa: BLE001 - no champion yet
        return fallback
