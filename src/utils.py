"""Small shared helpers: config loading and a consistent logger.

Kept deliberately simple — one YAML file, one dict, no schema validation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the project config.yaml into a plain dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def project_root() -> Path:
    """Root of the project (the folder containing config.yaml)."""
    return DEFAULT_CONFIG_PATH.parent


def resolve_path(relative_path: str) -> Path:
    """Resolve a config path (relative to project root) to an absolute Path."""
    return project_root() / relative_path


def setup_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent, readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
