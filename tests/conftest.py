import shutil

import pytest


def pytest_collection_modifyitems(config, items):
    if shutil.which("java"):
        return
    skip = pytest.mark.skip(reason="no java on PATH (Spark tests run in CI)")
    for item in items:
        if "spark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    wh = tmp_path_factory.mktemp("warehouse")
    builder = (
        SparkSession.builder.master("local[1]")
        .appName("tests")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.warehouse.dir", str(wh))
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
    )
    s = configure_spark_with_delta_pip(builder).getOrCreate()
    yield s
    s.stop()
