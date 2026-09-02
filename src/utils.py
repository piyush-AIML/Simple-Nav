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


# Stage 30 (planner v2 §13): config validation — run as the first line of
# build_map.py so a fresh clone fails loudly instead of silently misbehaving.
KNOWN_DETECTOR_MODELS = ("yolov8n.pt", "yolo26n.pt")
KNOWN_VLM_MODELS = (
    "LiquidAI/LFM2.5-VL-450M",
    "HuggingFaceTB/SmolVLM2-500M-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)


def validate_config(config: dict) -> list[tuple[str, str]]:
    """Check the config for the Stage 30 invariants. Returns a list of
    (level, message) with level in {"error", "warning"}; empty list = valid."""
    issues: list[tuple[str, str]] = []

    # every paths.* entry resolves to an existing parent (the file itself may
    # legitimately be absent, e.g. the walkthrough video — that's a warning)
    for key, rel in (config.get("paths") or {}).items():
        if not isinstance(rel, str):
            continue
        p = resolve_path(rel)
        if not p.parent.exists():
            issues.append(("error", f"paths.{key}: parent dir missing: {p.parent}"))
        elif key == "video" and not p.exists():
            issues.append(("warning", f"paths.{key}: video missing (provisional dataset): {p}"))

    # perception model ids must be registered; int4 on LFM2 is forbidden
    perception = config.get("perception", {}) or {}
    det = perception.get("detector_model")
    if det not in KNOWN_DETECTOR_MODELS:
        issues.append(("error", f"perception.detector_model: unknown model {det!r} "
                                f"(known: {list(KNOWN_DETECTOR_MODELS)})"))
    vlm = perception.get("vlm_model")
    if vlm not in KNOWN_VLM_MODELS:
        issues.append(("error", f"perception.vlm_model: unknown model {vlm!r} "
                                f"(known: {list(KNOWN_VLM_MODELS)})"))
    quant = perception.get("vlm_quantization")
    if vlm and "LFM2" in vlm and quant == "4bit":
        issues.append(("error", "perception.vlm_quantization: LFM2 must never run at 4bit "
                                "(planner v2 §5.2)"))

    # embedding.model must be a registered encoder (planner v3 §6). Lazy
    # import: src/embeddings/encoder.py imports setup_logger from this
    # module, so a module-level import here would be circular.
    from src.embeddings.encoder import _REGISTRY as KNOWN_ENCODERS

    emb = (config.get("embedding") or {}).get("model")
    if emb not in KNOWN_ENCODERS:
        issues.append(("error", f"embedding.model: unknown model {emb!r} "
                                f"(known: {sorted(KNOWN_ENCODERS)})"))

    # localization weights: each in [0, 1], non-trivial sum
    loc = config.get("localization", {}) or {}
    weights = {k: float(loc.get(k, 0.0)) for k in
               ("w_visual", "w_semantic", "w_temporal", "w_graph")}
    for k, v in weights.items():
        if not 0.0 <= v <= 1.0:
            issues.append(("error", f"localization.{k}: weight {v} outside [0, 1]"))
    total = sum(weights.values())
    if total <= 0.0 or total > 2.0:
        issues.append(("error", f"localization weights sum {total} is not sane"))
    return issues


def print_config_issues(issues: list[tuple[str, str]], logger=None) -> None:
    """Print validation issues; the caller decides whether errors abort."""
    log = logger or setup_logger("config")
    for level, msg in issues:
        (log.warning if level == "warning" else log.error)(f"config check: {msg}")
