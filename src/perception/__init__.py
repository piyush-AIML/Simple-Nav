"""Perception layer (planner v2 §5): detector + VLM scene tagger, both
EVIDENCE only — never ground truth for topology/coordinates (Rule 3)."""


def backend_banner(config: dict) -> str:
    """One-line startup banner (planner v2 §13): which backends the config
    selects. The factories log the actually-loaded backend and warn on any
    StubDetector/StubTagger fallback; this line makes the intent visible at
    startup so a silent fallback during a demo can't go unnoticed."""
    p = config.get("perception", {}) or {}
    return (
        f"perception backends — detector: {p.get('detector_model', 'off')}, "
        f"tagger: {p.get('vlm_model', 'off')} "
        f"({p.get('vlm_quantization', '?')}, enabled={p.get('vlm_enabled', True)})"
    )
