"""Spark ingestion: ABCD JSON -> bronze `raw_calls` -> silver `calls_clean`."""

from __future__ import annotations

import gzip
import json
import os

from support_categorizer.errors import UserFacingError
from support_categorizer.labels import normalize_subflow
from support_categorizer.transcript import MAX_CHARS, MIN_TURNS

ABCD_URL = "https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz"
ABCD_FILE = "abcd_v1.1.json.gz"

UPLOAD_HELP = (
    "Could not download the ABCD data from GitHub ({err}). One-time fix (1 minute): on your laptop "
    f"open {ABCD_URL} to download {ABCD_FILE}; in Databricks go to Catalog > workspace > support > "
    "landing > 'Upload to this volume' and upload it; then rerun this cell."
)


def download_abcd(dest_path: str, url: str = ABCD_URL) -> str:
    """Download the ABCD file into the Volume path unless it is already there."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        print(f"Already downloaded: {dest_path}")
        return dest_path
    import httpx

    tmp = dest_path + ".part"
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except (httpx.HTTPError, OSError) as e:
        raise UserFacingError(UPLOAD_HELP.format(err=type(e).__name__)) from None
    os.replace(tmp, dest_path)
    print(f"Downloaded {os.path.getsize(dest_path) / 1e6:.0f} MB to {dest_path}")
    return dest_path


BRONZE_SCHEMA = (
    "call_id string, split string, turns array<struct<speaker:string,text:string>>, "
    "flow string, subflow string, member_level string"
)


def read_abcd_rows(path: str) -> list[tuple]:
    """Rows for bronze, read on the driver (37 MB file; no Spark schema inference)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for split, convos in data.items():
        for c in convos:
            sc = c["scenario"]
            rows.append(
                (
                    str(c["convo_id"]),
                    split,
                    [(str(s), str(t)) for s, t in c["original"]],
                    sc.get("flow"),
                    sc.get("subflow"),
                    (sc.get("personal") or {}).get("member_level"),
                )
            )
    return rows


def bronze_from_json(spark, path: str):
    from pyspark.sql import functions as F

    df = spark.createDataFrame(read_abcd_rows(path), schema=BRONZE_SCHEMA)
    return df.withColumn("ingested_at", F.current_timestamp())


def subflow_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """`flow|raw_subflow` -> normalized subflow, for every pair that normalizes."""
    out = {}
    for flow, sub in pairs:
        norm = normalize_subflow(flow, sub) if flow and sub else None
        if norm:
            out[f"{flow}|{sub}"] = norm
    return out


def silver_from_bronze(bronze):
    """Clean transcripts with Spark built-ins only (no Python UDF), normalize labels."""
    from pyspark.sql import functions as F

    pairs = [(r.flow, r.subflow) for r in bronze.select("flow", "subflow").distinct().collect()]
    mapping = subflow_map(pairs)
    label_map = F.create_map(*[F.lit(x) for kv in mapping.items() for x in kv])
    real = F.filter("turns", lambda t: t.speaker.isin("agent", "customer"))
    lines = F.transform(
        real,
        lambda t: F.concat(
            F.initcap(t.speaker), F.lit(": "), F.trim(F.regexp_replace(t.text, r"\s+", " "))
        ),
    )
    return (
        bronze.withColumn("_real", F.size(real))
        .withColumn("call_transcript", F.substring(F.array_join(lines, "\n"), 1, MAX_CHARS))
        .withColumn("secondary_category", label_map[F.concat_ws("|", "flow", "subflow")])
        .where(F.col("secondary_category").isNotNull() & (F.col("_real") >= MIN_TURNS))
        .select(
            "call_id",
            "split",
            "call_transcript",
            F.coalesce("member_level", F.lit("unknown")).alias("customer_tier"),
            F.col("flow").alias("primary_category"),
            "secondary_category",
        )
    )
