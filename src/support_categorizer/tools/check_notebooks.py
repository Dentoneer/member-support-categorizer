"""Check notebooks and the app: imports from support_categorizer resolve, learn box present.

Usage: python -m support_categorizer.tools.check_notebooks notebooks app
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

HEADER = "# Databricks notebook source"


def _problems(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out = []
    if path.name != "app.py":
        if not text.startswith(HEADER):
            out.append(f"{path.name}: must start with '{HEADER}'")
        first_cell = text.split("# COMMAND ----------")[0]
        if "What you'll learn" not in first_cell:
            out.append(f"{path.name}: first cell needs a \"What you'll learn\" box")
    try:
        tree = ast.parse(text)  # MAGIC cells are comments, so a notebook is valid Python
    except SyntaxError as e:
        return [*out, f"{path.name}: syntax error line {e.lineno}: {e.msg}"]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("support_")):
            continue
        try:
            mod = importlib.import_module(node.module)
        except ImportError as e:
            out.append(f"{path.name}: cannot import {node.module} ({e})")
            continue
        for alias in node.names:
            if not hasattr(mod, alias.name):
                out.append(f"{path.name}: {node.module} has no {alias.name}")
    return out


def main(argv: list[str]) -> int:
    problems = []
    for d in argv:
        for p in sorted(Path(d).glob("*.py")):
            problems += _problems(p)
    for p in problems:
        print(p)
    if not problems:
        print(f"OK: {', '.join(argv)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
