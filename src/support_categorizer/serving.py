"""Model Serving: the champion DSPy program as a REST endpoint, with AI Gateway inference tables."""

from __future__ import annotations

import time

from support_categorizer.errors import UserFacingError
from support_categorizer.settings import MODEL_ALIAS, MODEL_NAME, SERVING_ENDPOINT, Settings

INFERENCE_TABLE_PREFIX = "serving"


def served_entity(s: Settings, version: str):
    from databricks.sdk.service.serving import ServedEntityInput

    return ServedEntityInput(entity_name=s.fqn(MODEL_NAME), entity_version=str(version),
                             workload_size="Small", scale_to_zero_enabled=True)  # fmt: skip


def inference_table(s: Settings):
    from databricks.sdk.service.serving import AiGatewayInferenceTableConfig

    return AiGatewayInferenceTableConfig(catalog_name=s.catalog, schema_name=s.schema,
                                         table_name_prefix=INFERENCE_TABLE_PREFIX, enabled=True)  # fmt: skip


def champion_version(s: Settings) -> str:
    import mlflow

    try:
        mlflow.set_registry_uri("databricks-uc")
        return str(mlflow.MlflowClient().get_model_version_by_alias(s.fqn(MODEL_NAME), MODEL_ALIAS).version)
    except Exception as e:  # noqa: BLE001
        raise UserFacingError("No champion model in Unity Catalog yet: run notebook 03 (and 08) first. "
                              f"({str(e)[:120]})") from None  # fmt: skip


def upsert_endpoint(w, s: Settings, version: str, name: str = SERVING_ENDPOINT) -> str:
    """Create the endpoint, or point the existing one at `version`. Does not wait."""
    from databricks.sdk.service.serving import AiGatewayConfig, EndpointCoreConfigInput

    try:
        w.serving_endpoints.get(name)
        exists = True
    except Exception:  # noqa: BLE001 - not found
        exists = False
    if exists:
        w.serving_endpoints.update_config(name=name, served_entities=[served_entity(s, version)])
        try:
            w.serving_endpoints.put_ai_gateway(name=name, inference_table_config=inference_table(s))
        except Exception as e:  # noqa: BLE001 - logging is a bonus; serving still works
            print(f"(could not update the inference table settings: {str(e)[:120]})")
        return "updated"
    w.serving_endpoints.create(
        name=name,
        config=EndpointCoreConfigInput(name=name, served_entities=[served_entity(s, version)]),
        ai_gateway=AiGatewayConfig(inference_table_config=inference_table(s)),
    )
    return "created"


def wait_ready(w, name: str = SERVING_ENDPOINT, timeout: float = 2400, poll: float = 30,
               sleep=time.sleep, now=time.time) -> None:
    """Until the endpoint serves the new config; plain-English errors on failure or timeout."""
    start = now()
    while True:
        st = w.serving_endpoints.get(name).state
        ready = getattr(getattr(st, "ready", None), "value", None)
        update = getattr(getattr(st, "config_update", None), "value", None)
        if update == "UPDATE_FAILED":
            raise UserFacingError(f"Serving endpoint '{name}' failed to start. Open Serving > {name} > "
                                  "Events/Logs for the reason (on Free Edition this is often a "
                                  "serving limit).")  # fmt: skip
        if ready == "READY" and update in ("NOT_UPDATING", None):
            return
        if now() - start > timeout:
            raise UserFacingError(f"Serving endpoint '{name}' is still starting ({ready}, {update}). "
                                  "Rerun this cell in 10 minutes.")  # fmt: skip
        print(f"{name}: {ready} / {update} ... waiting (building the container takes 10-30 minutes)")
        sleep(poll)


def serving_payload(rows: list[dict]) -> list[dict]:
    """dataframe_records for the served DSPy program (it retrieves guidelines itself)."""
    return [{"call_transcript": r["call_transcript"], "customer_tier": r["customer_tier"], "guidelines": ""}
            for r in rows]  # fmt: skip
