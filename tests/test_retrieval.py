import pickle
from types import SimpleNamespace as NS

import pytest

from support_categorizer.errors import UserFacingError
from support_categorizer.retrieval import (
    GuidelineRetriever,
    ensure_endpoint,
    ensure_index,
    format_guidelines,
    recall_at_k,
    retrieval_query,
    wait_endpoint,
    wait_index,
)

TRANSCRIPT = "Agent: Hi!\nCustomer: my package says delayed\nAgent: ok\nCustomer: is it lost?"


def test_retrieval_query_uses_customer_lines():
    assert retrieval_query(TRANSCRIPT) == "my package says delayed is it lost?"
    assert retrieval_query("Agent: only agent") == "Agent: only agent"
    assert len(retrieval_query("Customer: " + "x" * 5000)) == 1000


def test_format_and_recall():
    hits = [{"secondary_category": "status", "primary_category": "shipping_issue", "guideline_text": "t"}]
    assert format_guidelines(hits) == "- status (shipping_issue): t"
    assert recall_at_k([["a", "b"], ["c"]], ["b", "x"]) == 0.5


class _Indexes:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def query_index(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("index not ready")
        cols = [NS(name=c) for c in ("secondary_category", "primary_category", "guideline_text", "score")]
        return NS(manifest=NS(columns=cols), result=NS(data_array=[["status", "shipping_issue", "t", 0.9]]))


def test_retriever_formats_hits_and_survives_failures():
    idx = _Indexes()
    r = GuidelineRetriever("cat.sch.idx", k=3, client_factory=lambda: NS(vector_search_indexes=idx))
    assert r(TRANSCRIPT) == "- status (shipping_issue): t"
    assert idx.calls[0]["num_results"] == 3 and idx.calls[0]["query_text"].startswith("my package")
    bad = GuidelineRetriever("i", client_factory=lambda: NS(vector_search_indexes=_Indexes(fail=True)))
    assert bad(TRANSCRIPT) == ""


def test_retriever_pickles_without_client():
    r = GuidelineRetriever("cat.sch.idx", client_factory=lambda: NS(vector_search_indexes=_Indexes()))
    r(TRANSCRIPT)  # creates the client
    clone = pickle.loads(pickle.dumps(r))
    assert clone.index_name == "cat.sch.idx" and clone._client is None and clone.client_factory is None


def _w_endpoints(names, created):
    return NS(vector_search_endpoints=NS(
        list_endpoints=lambda: [NS(name=n) for n in names],
        create_endpoint=lambda name, endpoint_type: created.append(name)))


def test_ensure_endpoint_reuses_or_creates():
    created = []
    assert ensure_endpoint(_w_endpoints(["support-vs"], created)) == "support-vs"
    assert ensure_endpoint(_w_endpoints(["other"], created)) == "other" and created == []
    assert ensure_endpoint(_w_endpoints([], created)) == "support-vs" and created == ["support-vs"]


def test_wait_endpoint_times_out_with_plain_message():
    w = NS(vector_search_endpoints=NS(get_endpoint=lambda n: NS(endpoint_status=NS(state=NS(value="PROVISIONING")))))
    t = iter(range(0, 10_000, 100))
    with pytest.raises(UserFacingError, match="rerun this cell"):
        wait_endpoint(w, "support-vs", timeout=250, sleep=lambda s: None, now=lambda: next(t))


def test_wait_index_until_all_rows():
    states = iter([NS(ready=False, indexed_row_count=0), NS(ready=True, indexed_row_count=55)])
    w = NS(vector_search_indexes=NS(get_index=lambda n: NS(status=next(states))))
    wait_index(w, "i", 55, sleep=lambda s: None)


def test_ensure_index_creates_then_syncs():
    made, synced = [], []
    from databricks.sdk.errors import NotFound

    missing = NS(vector_search_indexes=NS(get_index=lambda n: (_ for _ in ()).throw(NotFound("404")),
                                          create_index=lambda **kw: made.append(kw)))
    assert ensure_index(missing, "c.s.idx", "support-vs", "c.s.category_guidelines") == "created"
    assert made[0]["primary_key"] == "secondary_category"
    spec = made[0]["delta_sync_index_spec"]
    assert spec.source_table == "c.s.category_guidelines"
    assert spec.embedding_source_columns[0].embedding_model_endpoint_name == "databricks-gte-large-en"
    ready = NS(vector_search_indexes=NS(get_index=lambda n: NS(status=NS(ready=True)),
                                        sync_index=lambda n: synced.append(n)))
    assert ensure_index(ready, "c.s.idx", "support-vs", "t") == "synced" and synced == ["c.s.idx"]


# ---- review fixes -------------------------------------------------------------------------


def test_ensure_index_only_treats_not_found_as_missing():
    from databricks.sdk.errors import NotFound, PermissionDenied

    made = []
    w = NS(vector_search_indexes=NS(get_index=lambda n: (_ for _ in ()).throw(NotFound("no index")),
                                    create_index=lambda **kw: made.append(kw)))
    assert ensure_index(w, "c.s.idx", "support-vs", "t") == "created"
    denied = NS(vector_search_indexes=NS(get_index=lambda n: (_ for _ in ()).throw(PermissionDenied("nope")),
                                         create_index=lambda **kw: made.append("BAD")))
    with pytest.raises(UserFacingError, match="Could not read"):
        ensure_index(denied, "c.s.idx", "support-vs", "t")
    assert "BAD" not in made


def test_ensure_index_create_failure_is_plain_english():
    from databricks.sdk.errors import NotFound

    def create(**kw):
        raise RuntimeError("embedding endpoint does not exist")

    w = NS(vector_search_indexes=NS(get_index=lambda n: (_ for _ in ()).throw(NotFound("x")), create_index=create))
    with pytest.raises(UserFacingError, match="embedding endpoint does not exist"):
        ensure_index(w, "c.s.idx", "support-vs", "t")


def test_sync_already_running_is_not_an_error():
    def sync(n):
        raise RuntimeError("A sync is already in progress for this index")

    w = NS(vector_search_indexes=NS(get_index=lambda n: NS(status=NS(ready=True)), sync_index=sync))
    assert "already" in ensure_index(w, "c.s.idx", "support-vs", "t")


def test_ensure_endpoint_errors_are_plain_english():
    def boom():
        raise RuntimeError("quota exceeded")

    with pytest.raises(UserFacingError, match="Vector Search"):
        ensure_endpoint(NS(vector_search_endpoints=NS(list_endpoints=boom)))


def test_wait_endpoint_fails_fast_on_terminal_state():
    st = NS(state=NS(value="RED_STATE"), message="disk failure")
    w = NS(vector_search_endpoints=NS(get_endpoint=lambda n: NS(endpoint_status=st)))
    calls = []
    with pytest.raises(UserFacingError, match="disk failure"):
        wait_endpoint(w, "support-vs", sleep=lambda s: calls.append(s))
    assert calls == []  # no waiting


def test_wait_index_waits_for_the_new_keys_and_reports_failures():
    keys = iter([{"cost"}, {"shipping_cost"}])
    states = NS(ready=True, indexed_row_count=1, message="")
    w = NS(vector_search_indexes=NS(get_index=lambda n: NS(status=states)))
    seen = []
    wait_index(w, "i", 1, expected_keys={"shipping_cost"}, keys_fn=lambda: next(keys),
               sleep=lambda s: seen.append(s))
    assert len(seen) == 1  # waited once for the rename to show up
    failed = NS(vector_search_indexes=NS(get_index=lambda n: NS(status=NS(
        ready=False, indexed_row_count=0, message="Pipeline FAILED: source table missing CDF"))))
    with pytest.raises(UserFacingError, match="source table missing CDF"):
        wait_index(failed, "i", 1, sleep=lambda s: None)


def test_retriever_prints_failure_once():
    import io
    from contextlib import redirect_stdout

    bad = GuidelineRetriever("i", client_factory=lambda: NS(vector_search_indexes=_Indexes(fail=True)))
    buf = io.StringIO()
    with redirect_stdout(buf):
        for _ in range(5):
            bad("Customer: x")
    assert buf.getvalue().count("retrieval failed") == 1 and bad.failures == 5
