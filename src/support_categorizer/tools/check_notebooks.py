"""Check notebooks and the app: imports from support_categorizer resolve, learn box present.

Usage: python -m support_categorizer.tools.check_notebooks notebooks app
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

HEADER = "# Databricks notebook source"
IMPORT = re.compile(r"^\s*(?:# MAGIC\s+)?from (support_categorizer[\w.]*) import \(?([^)#\n]+)")


def _problems(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out = []
    if path.name != "app.py":
        if not text.startswith(HEADER):
            out.append(f"{path.name}: must start with '{HEADER}'")
        first_cell = text.split("# COMMAND ----------")[0]
        if "What you'll learn" not in first_cell:
            out.append(f"{path.name}: first cell needs a \"What you'll learn\" box")
    for line in text.splitlines():
        m = IMPORT.match(line)
        if not m:
            continue
        module, names = m.group(1), [n.strip() for n in m.group(2).split(",") if n.strip()]
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            out.append(f"{path.name}: cannot import {module} ({e})")
            continue
        for n in names:
            n = n.split(" as ")[0].strip()
            if not hasattr(mod, n):
                out.append(f"{path.name}: {module} has no {n}")
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
