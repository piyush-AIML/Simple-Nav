"""VLM scene/landmark tagging (§7) — semantic EVIDENCE for place
discrimination, with hard constraints:

- fixed JSON schema, validated before storage;
- the VLM NEVER emits coordinates, distances, graph topology, or a location —
  the schema structurally forbids it (§7 "must NOT produce");
- deterministic-ish: temperature 0, fixed prompt, closed scene + landmark
  vocabularies; sign_text and walkable directions are evidence only;
- tagger failures degrade to unknown/[] — never crash the pipeline.

Backends (config perception.vlm_model):
- LiquidAI/LFM2.5-VL-450M (default) — bf16, no quantization, fits 6 GB VRAM
  (planner v2 §5.2; never int4 — it degrades this model sharply)
- HuggingFaceTB/SmolVLM2-500M-Instruct — lighter fallback
- Qwen/Qwen2.5-VL-3B-Instruct — legacy opt-in only; its 4-bit output is
  broken by a documented transformers >= 4.50 regression (planner v2 §4)
- StubTagger (vlm_enabled: false, or no GPU/network) — deterministic, offline
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.utils import setup_logger

logger = setup_logger("scene_tagger")

SCENE_TYPES = ("room", "corridor", "junction", "stairs", "elevator", "entrance", "lobby", "unknown")
# v1 schema used "corridor_junction"; v2 renamed it to "junction". Legacy
# stored data is normalized via normalize_scene_type() (never re-tagged).
LEGACY_SCENE_ALIASES = {"corridor_junction": "junction"}
LANDMARK_TYPES = ("door", "room_sign", "direction_sign", "stairs", "elevator",
                  "entrance", "reception", "desk", "junction",
                  "corridor_opening", "fire_equipment", "other")
WALKABLE_DIRS = ("left", "forward", "right")

# NOTE: the schema below must match the model's actual output habits — the
# default backend (LFM2.5-VL-450M) echoes "scene_type": "a | b | ..." verbatim
# when the enum is presented as bare pipe text, so the JSON template uses
# explicit <one of: ...> / [<plain strings>] placeholders plus a worked
# example. Content = the scene-tagger v2 spec; format = what the 450M
# actually follows (verified on real frames).
PROMPT = """You are an indoor navigation vision assistant analyzing ONE camera frame.
Return JSON ONLY, every list item a plain quoted string:
{"scene_type": <one of: room, corridor, junction, stairs, elevator, entrance, lobby, unknown>,
 "landmarks": [<plain strings, subset of: door, room_sign, direction_sign, stairs, elevator, entrance, reception, desk, junction, corridor_opening, fire_equipment, other>],
 "sign_text": [<plain strings: text actually readable on visible signs>],
 "walkable": [<plain strings, subset of: left, forward, right>],
 "description": <one short sentence, max 15 words>}
Example: {"scene_type": "corridor", "landmarks": ["door", "direction_sign"], "sign_text": ["Room 101"], "walkable": ["forward"], "description": "Long corridor with doors on both sides."}

Rules:
- Report only what is clearly visible. NEVER guess.
- If unsure use scene_type "unknown" — but still list clearly visible landmarks. Do not say "unknown" just because you cannot name the whole scene.
- left/forward/right mean LEFT/FRONT/RIGHT in the image; report a direction only when an open path is visibly present. A door is not automatically a walkable path.
- junction means two or more visible paths branch/intersect.
- Never invent sign text — write only text that can actually be read. If nothing useful is identifiable, use "unknown" and empty lists.
- No coordinates, distances, map locations, compass directions, or graph info.

Return exactly one JSON object."""


def normalize_scene_type(scene_type: str) -> str:
    """Map legacy stored scene values onto the current vocabulary
    (v1 "corridor_junction" == v2 "junction"). Unknown values pass through
    unchanged — callers handle unrecognized values themselves."""
    if scene_type in SCENE_TYPES:
        return scene_type
    return LEGACY_SCENE_ALIASES.get(scene_type, scene_type)


@dataclass
class SceneTags:
    scene_type: str = "unknown"
    landmarks: list[str] = field(default_factory=list)  # subset of LANDMARK_TYPES
    sign_text: list[str] = field(default_factory=list)
    walkable: list[str] = field(default_factory=list)  # subset of WALKABLE_DIRS
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "landmarks": self.landmarks,
            "sign_text": self.sign_text,
            "walkable": self.walkable,
            "description": self.description,
        }


UNKNOWN_TAGS = SceneTags()


def _str_list(value, cap: int) -> list[str]:
    """Coerce JSON value to a cleaned string list: strings only, stripped,
    length-capped, order-preserving dedupe (no item cap — callers filter
    against vocabularies before slicing)."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()[:cap]
        if item and item not in out:
            out.append(item)
    return out


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

    scene_type = normalize_scene_type(str(data.get("scene_type", "unknown")))
    if scene_type not in SCENE_TYPES:
        scene_type = "unknown"

    # vocabulary filter BEFORE the item cap: an out-of-vocab token must never
    # displace a valid one
    landmarks = [lm for lm in _str_list(data.get("landmarks"), cap=80) if lm in LANDMARK_TYPES][:12]

    sign_text = _str_list(data.get("sign_text"), cap=80)[:8]

    walkable = [d for d in _str_list(data.get("walkable"), cap=16) if d in WALKABLE_DIRS][:3]

    desc = data.get("description", "")
    if not isinstance(desc, str):
        desc = ""
    desc = desc[:160]

    return SceneTags(scene_type=scene_type, landmarks=landmarks,
                     sign_text=sign_text, walkable=walkable, description=desc)


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

    def _build_prompt(self, objects: list | None) -> str:
        extra = ""
        if objects:
            names = ", ".join(o.class_name if hasattr(o, "class_name") else str(o) for o in objects)
            extra = f"\nObjects visible in the frame: {names}"
        return PROMPT + extra

    def tag(self, image, objects: list | None = None) -> SceneTags:
        try:
            self._load()
            text = self._generate(image, self._build_prompt(objects))
            return parse_and_validate(text)
        except Exception as e:
            logger.warning(f"VLM tagging failed ({e}) — returning unknown")
            return UNKNOWN_TAGS

    def tag_batch(self, images, objects_lists: list | None = None) -> list[SceneTags]:
        """Offline mapping pass (Stage 24, planner v2 §7). Base = sequential
        loop; LFM2VLTagger overrides with a true batched vision-tower pass."""
        objects_lists = objects_lists or [None] * len(images)
        return [self.tag(img, objs) for img, objs in zip(images, objects_lists)]

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


class LFM2VLTagger(_HFVisionTagger):
    """LiquidAI LFM2.5-VL-450M — default backend (bf16, fits 6 GB VRAM
    without quantization; do not use int4 — see planner v2 §5.2)."""

    def __init__(self, model_id: str = "LiquidAI/LFM2.5-VL-450M", max_new_tokens: int = 256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def _load(self) -> None:
        # override base _load: skip the 4-bit BitsAndBytesConfig entirely
        if self._loaded:
            return
        self._do_load()
        self._loaded = True

    def _do_load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        # decoder-only: batched generation needs LEFT padding. Right padding
        # corrupts outputs — measured on the 301-frame offline pass: the
        # right-padded batch run kept ~49% unknown while the same frames
        # single-image scored ~77%+ known. The stored v1 tags carry this
        # artifact too (that pass was also batched).
        try:
            self._processor.tokenizer.padding_side = "left"
        except Exception as e:
            logger.warning(f"Could not set left padding ({e}) — batch tagging may degrade")
        logger.info(f"LFM2.5-VL loaded: {self.model_id} (bf16)")

    @staticmethod
    def _pil(image) -> "PIL.Image.Image":
        from PIL import Image
        from transformers.image_utils import load_image

        if isinstance(image, str):
            return load_image(image)
        if not isinstance(image, Image.Image):
            import numpy as np

            return Image.fromarray(np.asarray(image)).convert("RGB")
        return image

    def _generate(self, image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": self._pil(image)}, {"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        output = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_prompt_echo(output)

    def tag_batch(self, images, objects_lists: list | None = None) -> list[SceneTags]:
        """True batching (Stage 24): one vision-tower pass over the whole
        group, then per-item JSON validation. Falls back to per-image
        tagging on any failure (VRAM, template quirks) — never crashes."""
        try:
            self._load()
            objects_lists = objects_lists or [None] * len(images)
            conversations = [
                [{"role": "user", "content": [
                    {"type": "image", "image": self._pil(img)},
                    {"type": "text", "text": self._build_prompt(objs)},
                ]}]
                for img, objs in zip(images, objects_lists)
            ]
            inputs = self._processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(self._model.device)
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            outputs = self._processor.batch_decode(generated, skip_special_tokens=True)
            return [parse_and_validate(_strip_prompt_echo(out)) for out in outputs]
        except Exception as e:
            logger.warning(f"Batched VLM tagging failed ({e}) — falling back to per-image")
            return [self.tag(img, objs) for img, objs in zip(images, objects_lists)]

    def name(self) -> str:
        return f"lfm2.5vl-450m:{self.model_id}"


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
    """SmolVLM2-500M-Instruct — lighter fallback for 6 GB-class GPUs."""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Instruct", max_new_tokens: int = 256):
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
        landmarks = []
        scene = "unknown"
        if objects:
            names = [o.class_name if hasattr(o, "class_name") else str(o) for o in objects]
            if any("refrigerator" in n or "tv" in n or "couch" in n or "chair" in n for n in names):
                scene = "room"
            if any("bench" in n or "person" in n for n in names):
                scene = "corridor" if scene == "unknown" else scene
            if any("fire hydrant" in n for n in names):
                landmarks.append("fire_equipment")
        return SceneTags(scene_type=scene, landmarks=landmarks)

    def name(self) -> str:
        return self.name_


def get_scene_tagger(config: dict) -> SceneTagger:
    """Build a tagger from the perception config. Never raises — on any
    failure (no network, no GPU, bad model id) falls back to the stub."""
    perception = config.get("perception", {})
    if not perception.get("vlm_enabled", True):
        return StubTagger()
    model_id = perception.get("vlm_model", "LiquidAI/LFM2.5-VL-450M")
    max_tokens = int(perception.get("vlm_max_tokens", 128))
    tagger_cls = (
        LFM2VLTagger if "LFM2" in model_id else
        QwenVLTagger if "Qwen" in model_id else
        SmolVLMTagger
    )
    try:
        tagger = tagger_cls(model_id=model_id, max_new_tokens=max_tokens)
        tagger._load()  # fail fast: verify the model actually loads
        return tagger
    except Exception as e:
        logger.warning(f"Could not load VLM {model_id}: {e} — using StubTagger")
        return StubTagger()
