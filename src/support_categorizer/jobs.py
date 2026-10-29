"""Databricks Job (Workflows) that refreshes the pipeline: ingest -> baseline/guidelines/governance ->
batch ai_query. Created or updated by name, so rerunning notebook 06 never makes duplicates."""

from __future__ import annotations

from support_categorizer.settings import JOB_NAME, Settings

# task key -> (notebook, depends on, extra parameters)
TASKS: dict[str, tuple[str, list[str], dict[str, str]]] = {
    "ingest": ("01_ingest", [], {}),
    "baseline": ("02_baseline", ["ingest"], {}),
    "guidelines_index": ("08_rag_vector_search", ["ingest"], {"mode": "index_only"}),
    "governance": ("07_governance", ["ingest"], {}),
    "batch_ai_query": ("05_batch_ai_query", ["baseline", "guidelines_index"], {}),
}


def job_tasks(notebooks_dir: str, s: Settings) -> list:
    """Serverless notebook tasks (no cluster spec), each retried once."""
    from databricks.sdk.service.jobs import NotebookTask, Task, TaskDependency

    params = {"catalog": s.catalog, "schema": s.schema}
    return [
        Task(task_key=key,
             notebook_task=NotebookTask(notebook_path=f"{notebooks_dir.rstrip('/')}/{nb}",
                                        base_parameters={**params, **extra}),
             depends_on=[TaskDependency(task_key=d) for d in deps] or None,
             max_retries=1, timeout_seconds=3600)
        for key, (nb, deps, extra) in TASKS.items()
    ]  # fmt: skip


def upsert_job(w, tasks: list, name: str = JOB_NAME) -> int:
    """Create the job, or replace the settings of the existing job with the same name."""
    from databricks.sdk.service.jobs import JobSettings

    existing = [j for j in w.jobs.list(name=name) if j.settings and j.settings.name == name]
    tags = {"project": "member-support-categorizer"}
    if existing:
        job_id = existing[0].job_id
        w.jobs.reset(job_id=job_id, new_settings=JobSettings(name=name, tasks=tasks, max_concurrent_runs=1,
                                                             tags=tags))  # fmt: skip
        print(f"Updated job '{name}' ({job_id}).")
        return job_id
    job_id = w.jobs.create(name=name, tasks=tasks, max_concurrent_runs=1, tags=tags).job_id
    print(f"Created job '{name}' ({job_id}).")
    return job_id
