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
    missing = NS(vector_search_indexes=NS(get_index=lambda n: (_ for _ in ()).throw(RuntimeError("404")),
                                          create_index=lambda **kw: made.append(kw)))
    assert ensure_index(missing, "c.s.idx", "support-vs", "c.s.category_guidelines") == "created"
    assert made[0]["primary_key"] == "secondary_category"
    spec = made[0]["delta_sync_index_spec"]
    assert spec.source_table == "c.s.category_guidelines"
    assert spec.embedding_source_columns[0].embedding_model_endpoint_name == "databricks-gte-large-en"
    ready = NS(vector_search_indexes=NS(get_index=lambda n: NS(status=NS(ready=True)),
                                        sync_index=lambda n: synced.append(n)))
    assert ensure_index(ready, "c.s.idx", "support-vs", "t") == "synced" and synced == ["c.s.idx"]
