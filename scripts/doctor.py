"""Lightweight repository health checks for Squirrel Scanner."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def main() -> int:
    checks = {
        "main.py": exists("main.py"),
        "scanner_gui.py": exists("scanner_gui.py"),
        "src package": exists("src/__init__.py"),
        "GUI requirements": exists("requirements-guiscan.txt"),
        "pipeline requirements": exists("requirements-prod.txt"),
        "README": exists("README.md"),
        "gitignore": exists(".gitignore"),
    }

    print("Squirrel Scanner doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project root: {ROOT}")

    for label, ok in checks.items():
        print(f"{label}: {'OK' if ok else 'MISSING'}")

    if os.environ.get("ROBOFLOW_API_KEY"):
        print("Roboflow key: present")
    else:
        print("Roboflow key: not set")

    missing = [label for label, ok in checks.items() if not ok]
    if missing:
        print("Doctor: FAIL")
        return 1

    print("Doctor: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
