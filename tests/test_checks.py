from types import SimpleNamespace

import pytest

from support_categorizer.checks import run_all, store_jev_key
from support_categorizer.settings import Settings


class _Spark:
    def __init__(self):
        self.sql_calls = []

    def sql(self, q):
        self.sql_calls.append(q)


def _w(names):
    eps = [SimpleNamespace(name=n) for n in names]
    return SimpleNamespace(serving_endpoints=SimpleNamespace(list=lambda: eps))


class _NoSecret:
    class secrets:  # noqa: N801
        @staticmethod
        def get(scope, key):
            raise RuntimeError("Secret does not exist")


def test_checks_report_missing_secret_and_endpoints():
    spark = _Spark()
    res = run_all(spark, _w(["databricks-meta-llama-3-3-70b-instruct"]), _NoSecret, Settings(),
                  reach_fn=lambda url: (True, "reachable (HTTP 200)"))  # fmt: skip
    by = {name: (ok, msg) for name, ok, msg in res}
    assert by["Jev key secret"][0] is False and "Store the Jev key" in by["Jev key secret"][1]
    assert by["Endpoint databricks-meta-llama-3-3-70b-instruct"][0] is True
    assert by["Endpoint databricks-gpt-oss-120b"][0] is False
    assert by["Schema"][0] and any("CREATE VOLUME" in q for q in spark.sql_calls)


def test_store_jev_key_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        store_jev_key(None, "  ")


def test_store_jev_key_creates_scope_once():
    calls = []
    secrets = SimpleNamespace(
        list_scopes=lambda: [],
        create_scope=lambda scope: calls.append(("scope", scope)),
        put_secret=lambda scope, key, string_value: calls.append(("put", scope, key, string_value)),
    )
    store_jev_key(SimpleNamespace(secrets=secrets), " k ")
    assert calls == [("scope", "support"), ("put", "support", "typesafe_api_key", "k")]
