"""Config validation CLI (Stage 30, planner v2 §13): checks every path in
config.yaml resolves, every referenced model id is registered, and the
localization weights sum sanely.

Run:
    conda run -n ML python scripts/validate_config.py

build_map.py runs the same checks as its first line; this script exists for
standalone use and CI. Exit code 1 when hard errors are found.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, print_config_issues, validate_config


def main() -> None:
    issues = validate_config(load_config())
    print_config_issues(issues)
    if any(level == "error" for level, _ in issues):
        raise SystemExit(1)
    print("config valid")


if __name__ == "__main__":
    main()
