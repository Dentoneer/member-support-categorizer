"""Notebook 00 checks: schema/volume, LLM endpoints, network reach, Jev secret."""

from __future__ import annotations

from support_categorizer.settings import CANDIDATE_LLMS, SECRET_KEY, SECRET_SCOPE, VOLUME, Settings

ABCD_README = "https://raw.githubusercontent.com/asappresearch/abcd/master/README.md"
SITES = [
    ("GitHub (data download)", ABCD_README),
    ("PyPI (installing DSPy)", "https://pypi.org/simple/dspy/"),
    ("Jev (TypeSafe)", "https://api.typesafe.ai"),
]
SECRET_HELP = (
    "The Jev key is not stored yet. Run the 'Store the Jev key' cells at the bottom of this "
    "notebook once (paste the key into the jev_key box at the top, run the cell; the box is "
    "removed afterwards)."
)


def reach(url: str) -> tuple[bool, str]:
    """Any HTTP answer counts as reachable; only a network error means blocked."""
    import httpx

    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
    except httpx.HTTPError as e:
        return False, f"blocked ({type(e).__name__})"
    return True, f"reachable (HTTP {r.status_code})"


def run_all(spark, w, dbutils, s: Settings, reach_fn=reach) -> list[tuple[str, bool, str]]:
    """(check, ok, plain-English message) per check; never raises."""
    out: list[tuple[str, bool, str]] = []

    def step(name, fn):
        try:
            out.append((name, True, fn()))
        except Exception as e:  # each check reports instead of stopping the notebook
            out.append((name, False, str(e)[:300]))

    def schema():
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {s.catalog}.{s.schema}")
        return f"{s.catalog}.{s.schema} ready"

    def volume():
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {s.fqn(VOLUME)}")
        return f"{s.fqn(VOLUME)} ready"

    step("Schema", schema)
    step("Volume", volume)
    try:
        have = {e.name for e in w.serving_endpoints.list()}
    except Exception as e:
        have = set()
        out.append(("AI model endpoints", False, f"could not list endpoints: {str(e)[:200]}"))
    for ep in dict.fromkeys([s.llm_endpoint, *CANDIDATE_LLMS]):
        ok = ep in have
        msg = "available" if ok else "not in this workspace (skipped in the bake-off)"
        out.append((f"Endpoint {ep}", ok, msg))
    for name, url in SITES:
        ok, msg = reach_fn(url)
        out.append((name, ok, msg))

    def secret():
        try:
            dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)
        except Exception:
            raise RuntimeError(SECRET_HELP) from None
        return "stored"

    step("Jev key secret", secret)
    return out


def store_jev_key(w, key: str) -> None:
    """Create the secret scope if needed and store the Jev key in it."""
    key = (key or "").strip()
    if not key:
        raise ValueError("The jev_key box is empty: paste the key, then run this cell again.")
    if SECRET_SCOPE not in {sc.name for sc in w.secrets.list_scopes()}:
        w.secrets.create_scope(scope=SECRET_SCOPE)
    w.secrets.put_secret(scope=SECRET_SCOPE, key=SECRET_KEY, string_value=key)


def print_report(results: list[tuple[str, bool, str]]) -> None:
    for name, ok, msg in results:
        print(f"{'OK' if ok else 'XX'}  {name}: {msg}")
