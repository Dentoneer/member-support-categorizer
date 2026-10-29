from types import SimpleNamespace as NS

from support_categorizer.jobs import job_tasks, upsert_job
from support_categorizer.settings import Settings

NB = "/Users/me/member-support-categorizer/notebooks"


def test_job_tasks_graph_and_parameters():
    tasks = {t.task_key: t for t in job_tasks(NB + "/", Settings())}
    assert set(tasks) == {"ingest", "baseline", "guidelines_index", "governance", "batch_ai_query"}
    assert tasks["ingest"].depends_on is None
    assert [d.task_key for d in tasks["batch_ai_query"].depends_on] == ["baseline", "guidelines_index"]
    assert tasks["guidelines_index"].notebook_task.notebook_path == NB + "/08_rag_vector_search"
    assert tasks["guidelines_index"].notebook_task.base_parameters == {
        "catalog": "workspace", "schema": "support", "mode": "index_only"}
    assert all(t.max_retries == 1 and t.new_cluster is None for t in tasks.values())  # serverless


def _w(existing):
    calls = []
    jobs = NS(list=lambda name: existing,
              reset=lambda job_id, new_settings: calls.append(("reset", job_id, new_settings.name)),
              create=lambda **kw: calls.append(("create", kw["name"])) or NS(job_id=99))
    return NS(jobs=jobs), calls


def test_upsert_creates_then_updates_by_name():
    w, calls = _w([])
    assert upsert_job(w, []) == 99 and calls == [("create", "support-categorizer-refresh")]
    w, calls = _w([NS(job_id=7, settings=NS(name="support-categorizer-refresh"))])
    assert upsert_job(w, []) == 7 and calls == [("reset", 7, "support-categorizer-refresh")]
