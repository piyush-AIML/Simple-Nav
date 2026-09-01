"""VLM scene/landmark tagging (§7) — semantic EVIDENCE for place
discrimination, with hard constraints:

- fixed JSON schema, validated before storage;
- the VLM NEVER emits coordinates, distances, graph topology, or a location —
  the schema structurally forbids it (§7 "must NOT produce");
- deterministic-ish: temperature 0, fixed prompt, closed scene vocabulary;
- tagger failures degrade to unknown/[] — never crash the pipeline.

Backends (config perception.vlm_model):
- Qwen/Qwen2.5-VL-3B-Instruct (default) — 4-bit quantized, fits 6 GB VRAM
- HuggingFaceTB/SmolVLM2-2.2B-Instruct — lighter fallback
- StubTagger (vlm_enabled: false, or no GPU/network) — deterministic, offline
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.utils import setup_logger

logger = setup_logger("scene_tagger")

SCENE_TYPES = ("room", "corridor", "corridor_junction", "stairs", "elevator", "entrance", "unknown")
VALID_NAV_TAGS = ("junction", "stairs", "elevator", "entrance", "sign", "door",
                  "reception", "desk", "fire_equipment")

PROMPT = """You are a navigation assistant analyzing ONE indoor camera frame.
Respond with JSON ONLY, using exactly this schema:
{"scene_type": <one of: room, corridor, corridor_junction, stairs, elevator, entrance, unknown>,
 "landmarks": [<short noun phrases, up to 6, e.g. "blue room-number sign", "staircase on left">],
 "navigation_relevance": [<subset of: junction, stairs, elevator, entrance, sign, door, reception, desk, fire_equipment>],
 "description": <one short sentence, max 20 words>}

Rules:
- scene_type must be one of the listed values.
- List only VISIBLE, navigation-useful landmarks.
- If unsure, use "unknown" with empty lists — never guess.
- Do NOT output anything except the JSON."""


@dataclass
class SceneTags:
    scene_type: str = "unknown"
    landmarks: list[str] = field(default_factory=list)
    navigation_relevance: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "landmarks": self.landmarks,
            "navigation_relevance": self.navigation_relevance,
            "description": self.description,
        }


UNKNOWN_TAGS = SceneTags()


def parse_and_validate(text: str) -> SceneTags:
    """Parse model output into SceneTags. ANY deviation from the schema falls
    back to unknown/[] — we never store unvalidated model output (§7 instr 3)."""
    if not text:
        return UNKNOWN_TAGS
    text = text.strip()
    # strip markdown fences if the model wraps the JSON
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return UNKNOWN_TAGS
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return UNKNOWN_TAGS

    scene_type = data.get("scene_type", "unknown")
    if scene_type not in SCENE_TYPES:
        scene_type = "unknown"

    landmarks = data.get("landmarks", [])
    if not isinstance(landmarks, list):
        landmarks = []
    landmarks = [str(l).strip()[:80] for l in landmarks if isinstance(l, str)][:8]

    nav = data.get("navigation_relevance", [])
    if not isinstance(nav, list):
        nav = []
    nav = [str(t) for t in nav if str(t) in VALID_NAV_TAGS][:8]

    desc = data.get("description", "")
    if not isinstance(desc, str):
        desc = ""
    desc = desc[:160]

    return SceneTags(scene_type=scene_type, landmarks=landmarks,
                     navigation_relevance=nav, description=desc)


class SceneTagger(ABC):
    @abstractmethod
    def tag(self, image, objects: list | None = None) -> SceneTags:
        """Tag one image. Must never raise on bad input."""

    @abstractmethod
    def name(self) -> str:
        ...


class _HFVisionTagger(SceneTagger):
    """Base for transformers-based VLMs: loads model+processor on first tag."""

    model_id: str = ""
    _loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from transformers import BitsAndBytesConfig

        self._quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
        self._do_load()
        self._loaded = True

    def _do_load(self) -> None:  # implemented by subclasses
        ...

    def tag(self, image, objects: list | None = None) -> SceneTags:
        try:
            self._load()
            extra = ""
            if objects:
                names = ", ".join(o.class_name if hasattr(o, "class_name") else str(o) for o in objects)
                extra = f"\nObjects visible in the frame: {names}"
            text = self._generate(image, PROMPT + extra)
            return parse_and_validate(text)
        except Exception as e:
            logger.warning(f"VLM tagging failed ({e}) — returning unknown")
            return UNKNOWN_TAGS

    def _generate(self, image, prompt: str) -> str:
        raise NotImplementedError


def _strip_prompt_echo(output: str) -> str:
    """Qwen2.5-VL (and SmolVLM) sometimes echo the rendered chat template
    before answering. Everything up to the last assistant marker is the
    prompt — keep only what follows it."""
    marker = output.rfind("assistant")
    if marker != -1:
        return output[marker + len("assistant"):].lstrip("\n ")
    return output


class QwenVLTagger(_HFVisionTagger):
    """Qwen2.5-VL-3B-Instruct, 4-bit — default backend (fits 6 GB VRAM)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens: int = 256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def _do_load(self) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=self._quant, device_map="auto"
        )
        logger.info(f"Qwen2.5-VL loaded: {self.model_id} (4-bit)")

    def _generate(self, image, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        output = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_prompt_echo(output)

    def name(self) -> str:
        return f"qwen2.5vl-3b:{self.model_id}"


class SmolVLMTagger(_HFVisionTagger):
    """SmolVLM2-2.2B-Instruct — lighter fallback for 6 GB-class GPUs."""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct", max_new_tokens: int = 256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def _do_load(self) -> None:
        from transformers import AutoProcessor, SmolVLMForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = SmolVLMForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=self._quant, device_map="auto"
        )
        logger.info(f"SmolVLM2 loaded: {self.model_id} (4-bit)")

    def _generate(self, image, prompt: str) -> str:
        from PIL import Image

        if isinstance(image, str):
            image = Image.open(image)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        prompt_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt_text, images=[image], return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        output = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_prompt_echo(output)

    def name(self) -> str:
        return f"smolvlm2:{self.model_id}"


class StubTagger(SceneTagger):
    """Deterministic offline tagger: derives a scene type from detector objects
    (a door object -> 'room'/'entrance' hint, many people -> 'room', else
    'corridor'/'unknown'). Used when vlm_enabled is false or the model cannot
    load. Never used as ground truth."""

    name_ = "stub"

    def tag(self, image, objects: list | None = None) -> SceneTags:
        nav = []
        scene = "unknown"
        if objects:
            names = [o.class_name if hasattr(o, "class_name") else str(o) for o in objects]
            if any("refrigerator" in n or "tv" in n or "couch" in n or "chair" in n for n in names):
                scene = "room"
            if any("bench" in n or "person" in n for n in names):
                scene = "corridor" if scene == "unknown" else scene
            if any("fire hydrant" in n for n in names):
                nav.append("fire_equipment")
        return SceneTags(scene_type=scene, landmarks=[], navigation_relevance=nav)

    def name(self) -> str:
        return self.name_


def get_scene_tagger(config: dict) -> SceneTagger:
    """Build a tagger from the perception config. Never raises — on any
    failure (no network, no GPU, bad model id) falls back to the stub."""
    perception = config.get("perception", {})
    if not perception.get("vlm_enabled", True):
        return StubTagger()
    model_id = perception.get("vlm_model", "Qwen/Qwen2.5-VL-3B-Instruct")
    max_tokens = int(perception.get("vlm_max_tokens", 128))
    tagger_cls = QwenVLTagger if "Qwen" in model_id else SmolVLMTagger
    try:
        tagger = tagger_cls(model_id=model_id, max_new_tokens=max_tokens)
        tagger._load()  # fail fast: verify the model actually loads
        return tagger
    except Exception as e:
        logger.warning(f"Could not load VLM {model_id}: {e} — using StubTagger")
        return StubTagger()
