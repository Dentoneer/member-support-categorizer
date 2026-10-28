from support_categorizer.tools.check_notebooks import main

HEADER = "# Databricks notebook source\n# MAGIC %md\n# MAGIC ## What you'll learn\n"


def _nb(tmp_path, body, header=HEADER):
    d = tmp_path / "nb"
    d.mkdir()
    (d / "00_x.py").write_text(header + "\n# COMMAND ----------\n" + body, encoding="utf-8")
    return d


def test_check_notebooks_ok(tmp_path):
    d = _nb(tmp_path, "from support_categorizer.settings import Settings, fqn_note\n")
    assert main([str(d)]) == 1  # fqn_note does not exist
    d2 = tmp_path / "ok"
    d2.mkdir()
    (d2 / "a.py").write_text(
        HEADER + "\n# COMMAND ----------\nfrom support_categorizer.settings import Settings\n",
        encoding="utf-8",
    )
    assert main([str(d2)]) == 0


def test_check_notebooks_flags_missing_name(tmp_path):
    d = _nb(tmp_path, "from support_categorizer.settings import NOPE\n")
    assert main([str(d)]) == 1


def test_check_notebooks_requires_learn_box(tmp_path):
    d = _nb(tmp_path, "x = 1\n", header="# Databricks notebook source\n# MAGIC %md\n# MAGIC hi\n")
    assert main([str(d)]) == 1


def test_check_notebooks_requires_databricks_header(tmp_path):
    d = _nb(tmp_path, "x = 1\n", header="# MAGIC %md\n# MAGIC ## What you'll learn\n")
    assert main([str(d)]) == 1


def test_app_file_only_checks_imports(tmp_path):
    d = tmp_path / "app"
    d.mkdir()
    src = "from support_categorizer.settings import Settings\n"
    (d / "app.py").write_text(src, encoding="utf-8")
    assert main([str(d)]) == 0
