"""RAG retrieval: Vector Search endpoint + Delta Sync index over category guidelines."""

from __future__ import annotations

import time
from collections.abc import Callable

from support_categorizer.errors import UserFacingError

VS_ENDPOINT = "support-vs"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
TOP_K = 8
COLUMNS = ["secondary_category", "primary_category", "guideline_text"]


def retrieval_query(transcript: str, max_chars: int = 1000) -> str:
    """The customer's own lines (they state the issue early); the whole transcript if none."""
    lines = [ln.removeprefix("Customer:").strip() for ln in (transcript or "").splitlines()
             if ln.startswith("Customer:")]  # fmt: skip
    text = " ".join(x for x in lines if x) or (transcript or "")
    return text[:max_chars]


def format_guidelines(hits: list[dict]) -> str:
    return "\n".join(f"- {h['secondary_category']} ({h['primary_category']}): {h['guideline_text']}"
                     for h in hits)  # fmt: skip


def recall_at_k(retrieved: list[list[str]], truth: list[str]) -> float:
    """Share of calls whose true sub-category is among the retrieved ones."""
    pairs = list(zip(retrieved, truth, strict=True))
    return sum(t in r for r, t in pairs) / len(pairs) if pairs else 0.0


class GuidelineRetriever:
    """Callable: transcript -> formatted guidelines (the top-k by similarity). Picklable, so it can
    live inside the registered DSPy program; the workspace client is created lazily."""

    def __init__(self, index_name: str, k: int = TOP_K, client_factory: Callable | None = None):
        self.index_name = index_name
        self.k = k
        self.client_factory = client_factory
        self._client = None
        self.failures = 0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_client"] = None  # never pickle a live connection
        state["client_factory"] = None
        return state

    def _w(self):
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                from databricks.sdk import WorkspaceClient

                self._client = WorkspaceClient()
        return self._client

    def hits(self, transcript: str) -> list[dict]:
        resp = self._w().vector_search_indexes.query_index(
            index_name=self.index_name, columns=COLUMNS, query_text=retrieval_query(transcript),
            num_results=self.k)  # fmt: skip
        names = [c.name for c in resp.manifest.columns]
        rows = (resp.result.data_array if resp.result else None) or []
        return [dict(zip(names, row, strict=False)) for row in rows]

    def __call__(self, transcript: str) -> str:
        """Formatted guidelines, or "" if retrieval fails (the call is then categorized without)."""
        try:
            return format_guidelines(self.hits(transcript))
        except Exception as e:  # noqa: BLE001 - retrieval must never stop a batch
            self.failures += 1
            if self.failures == 1:  # say it once, not once per call
                print(f"(retrieval failed, continuing without guidelines: {str(e)[:160]})")
            return ""


def ensure_endpoint(w, name: str = VS_ENDPOINT) -> str:
    """Our endpoint if it exists; else any existing one (Free Edition allows one); else create it."""
    from databricks.sdk.service.vectorsearch import EndpointType

    try:
        existing = [e.name for e in w.vector_search_endpoints.list_endpoints()]
        if name in existing:
            return name
        if existing:
            print(f"Reusing the existing Vector Search endpoint '{existing[0]}' (Free Edition allows one).")
            return existing[0]
        w.vector_search_endpoints.create_endpoint(name=name, endpoint_type=EndpointType.STANDARD)
    except Exception as e:  # noqa: BLE001
        raise UserFacingError(
            f"Could not set up the Vector Search endpoint ({str(e)[:200]}). Check Compute > AI Search (Vector "
            "Search) in the left bar; if an endpoint is listed as failed, delete it and rerun this cell."
        ) from None
    print(f"Creating Vector Search endpoint '{name}' (takes 5-15 minutes the first time).")
    return name


def _wait(check: Callable[[], tuple[bool, str]], what: str, timeout: float, poll: float,
          sleep: Callable[[float], None], now: Callable[[], float]) -> None:
    start = now()
    while True:
        ok, state = check()
        if ok:
            return
        if now() - start > timeout:
            raise UserFacingError(f"{what} is still getting ready ({state}). Nothing is lost: rerun this "
                                  "cell in about 10 minutes.")  # fmt: skip
        print(f"{what}: {state} ... waiting")
        sleep(poll)


ENDPOINT_FAILED = {"OFFLINE", "RED_STATE", "STOPPED", "DELETED"}


def wait_endpoint(w, name: str, timeout: float = 1800, poll: float = 30,
                  sleep=time.sleep, now=time.time) -> None:
    def check():
        st = w.vector_search_endpoints.get_endpoint(name).endpoint_status
        state = getattr(getattr(st, "state", None), "value", str(getattr(st, "state", "unknown")))
        msg = getattr(st, "message", None) or ""
        if state in ENDPOINT_FAILED:  # waiting will not fix these
            raise UserFacingError(f"Vector Search endpoint '{name}' is {state}: {msg or 'no details'}. "
                                  "Open Compute > AI Search to see why (delete and rerun if it failed).")
        return state == "ONLINE", f"{state} {msg}".strip()

    _wait(check, f"Vector Search endpoint '{name}'", timeout, poll, sleep, now)


def ensure_index(w, index_name: str, endpoint: str, source_table: str,
                 embedding_endpoint: str = EMBEDDING_ENDPOINT) -> str:
    """Create the Delta Sync index, or trigger a sync of the existing one. Returns what it did."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest,
        EmbeddingSourceColumn,
        PipelineType,
        VectorIndexType,
    )

    try:
        idx = w.vector_search_indexes.get_index(index_name)
    except NotFound:
        idx = None
    except Exception as e:  # noqa: BLE001 - permissions, transient errors: don't try to re-create
        raise UserFacingError(f"Could not read the index {index_name} ({str(e)[:200]}). Rerun this cell; if it "
                              "keeps failing, check the index in Catalog > workspace > support.") from None
    if idx is None:
        try:
            w.vector_search_indexes.create_index(
                name=index_name, endpoint_name=endpoint, primary_key="secondary_category",
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                    source_table=source_table, pipeline_type=PipelineType.TRIGGERED,
                    embedding_source_columns=[EmbeddingSourceColumn(
                        name="guideline_text", embedding_model_endpoint_name=embedding_endpoint)]))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            raise UserFacingError(f"Could not create the Vector Search index ({str(e)[:250]}).") from None
        return "created"
    if not getattr(getattr(idx, "status", None), "ready", False):
        return "still building (first sync running)"
    try:
        w.vector_search_indexes.sync_index(index_name)
    except Exception as e:  # noqa: BLE001
        text = str(e).lower()
        if "already" in text or "in progress" in text or "running" in text:
            return "sync already running"
        raise UserFacingError(f"Could not sync the index ({str(e)[:250]}). Rerun this cell.") from None
    return "synced"


def index_keys(w, index_name: str, limit: int) -> set[str]:
    """The categories currently in the index (a broad query returning every row)."""
    resp = w.vector_search_indexes.query_index(index_name=index_name, columns=["secondary_category"],
                                               query_text="customer support request",
                                               num_results=limit)  # fmt: skip
    names = [c.name for c in resp.manifest.columns]
    col = names.index("secondary_category")
    return {row[col] for row in ((resp.result.data_array if resp.result else None) or [])}


def wait_index(w, index_name: str, expected_rows: int, timeout: float = 1800, poll: float = 30,
               sleep=time.sleep, now=time.time, expected_keys: set[str] | None = None,
               keys_fn: Callable[[], set[str]] | None = None) -> None:
    """Ready, holding every guideline row, and (if given) exactly the expected categories, so a
    rename or a new category is really searchable before we use the index."""
    if expected_keys is not None and keys_fn is None:
        def keys_fn():
            return index_keys(w, index_name, max(len(expected_keys) + 20, 100))

    def check():
        st = w.vector_search_indexes.get_index(index_name).status
        n = getattr(st, "indexed_row_count", None) or 0
        ready = bool(getattr(st, "ready", False))
        msg = getattr(st, "message", None) or ""
        if "fail" in msg.lower():  # a failed sync pipeline will not fix itself
            raise UserFacingError(f"The index sync failed: {msg[:300]}. Fix the cause, then rerun this cell.")
        ok = ready and n >= expected_rows
        if ok and expected_keys is not None:
            missing = expected_keys - keys_fn()
            if missing:
                return False, f"ready, waiting for {len(missing)} new/renamed categories to be indexed"
        return ok, f"ready={ready}, {n}/{expected_rows} rows indexed {msg[:80]}".strip()

    _wait(check, f"Index '{index_name}'", timeout, poll, sleep, now)
