from types import SimpleNamespace as NS

import pytest

from support_categorizer.errors import UserFacingError
from support_categorizer.serving import serving_payload, upsert_endpoint, wait_ready
from support_categorizer.settings import Settings


def _w(exists):
    calls = []

    def get(name):
        if not exists:
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
        return NS()

    se = NS(get=get,
            create=lambda **kw: calls.append(("create", kw)),
            update_config=lambda **kw: calls.append(("update", kw)),
            put_ai_gateway=lambda **kw: calls.append(("gateway", kw)))
    return NS(serving_endpoints=se), calls


def test_create_with_champion_version_scale_to_zero_and_inference_table():
    w, calls = _w(exists=False)
    assert upsert_endpoint(w, Settings(), "4") == "created"
    kw = calls[0][1]
    ent = kw["config"].served_entities[0]
    assert ent.entity_name == "workspace.support.call_categorizer_model" and ent.entity_version == "4"
    assert ent.scale_to_zero_enabled and ent.workload_size == "Small"
    it = kw["ai_gateway"].inference_table_config
    assert (it.catalog_name, it.schema_name, it.enabled) == ("workspace", "support", True)


def test_update_existing_endpoint():
    w, calls = _w(exists=True)
    assert upsert_endpoint(w, Settings(), "5") == "updated"
    assert [c[0] for c in calls] == ["update", "gateway"]
    assert calls[0][1]["served_entities"][0].entity_version == "5"


def _state(ready, update):
    return NS(state=NS(ready=NS(value=ready), config_update=NS(value=update)))


def test_wait_ready_success_failure_timeout():
    seq = iter([_state("NOT_READY", "IN_PROGRESS"), _state("READY", "NOT_UPDATING")])
    wait_ready(NS(serving_endpoints=NS(get=lambda n: next(seq))), sleep=lambda s: None)
    failed = NS(serving_endpoints=NS(get=lambda n: _state("NOT_READY", "UPDATE_FAILED")))
    with pytest.raises(UserFacingError, match="failed to start"):
        wait_ready(failed, sleep=lambda s: None)
    slow = NS(serving_endpoints=NS(get=lambda n: _state("NOT_READY", "IN_PROGRESS")))
    t = iter(range(0, 100_000, 1000))
    with pytest.raises(UserFacingError, match="Rerun this cell"):
        wait_ready(slow, timeout=1500, sleep=lambda s: None, now=lambda: next(t))


def test_serving_payload():
    assert serving_payload([{"call_transcript": "t", "customer_tier": "gold", "x": 1}]) == [
        {"call_transcript": "t", "customer_tier": "gold", "guidelines": ""}]
