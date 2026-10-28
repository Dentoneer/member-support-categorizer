"""Log + register the DSPy program in Unity Catalog (alias `champion`); Volume file fallback."""

from __future__ import annotations

import os

import mlflow

from support_categorizer.categorizer import CallCategorizerProgram
from support_categorizer.errors import UserFacingError
from support_categorizer.settings import MODEL_ALIAS, MODEL_NAME, Settings

PROGRAM_FILE = "program.json"


def register_program(program, s: Settings, input_example: dict, fallback_path: str | None = None) -> dict:
    """Inside an active MLflow run. Returns {"path": "uc", "version": v} or the Volume fallback."""
    name = s.fqn(MODEL_NAME)
    try:
        mlflow.set_registry_uri("databricks-uc")
        pkg = os.path.dirname(os.path.abspath(__file__))  # so the model can import our code anywhere
        info = mlflow.dspy.log_model(program, name="model", input_example=input_example,
                                     registered_model_name=name, code_paths=[pkg])  # fmt: skip
        version = str(info.registered_model_version)
        mlflow.MlflowClient().set_registered_model_alias(name, MODEL_ALIAS, version)
        print(f"Registered {name} version {version} with alias '{MODEL_ALIAS}' in Unity Catalog.")
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
