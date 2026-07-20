from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Optional

from .base import GeneratorBackend
from .templates import build_pair_generation_prompt, build_judge_prompt

logger = logging.getLogger(__name__)


def _extract_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in generator output: {text[:200]}")
    return json.loads(text[start:end + 1])


def _extract_float(text: str) -> Optional[float]:
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        return None
    val = float(match.group())
    return max(0.0, min(1.0, val))


class AnthropicGenerator(GeneratorBackend):
    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic  # local import so this is only required if used

        self.client = anthropic.Anthropic()
        self.model = model

    def generate_pairs(
        self,
        topic: str,
        persona: Optional[str],
        n_pairs: int,
        behavior_description: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        system, user = build_pair_generation_prompt(topic, persona, n_pairs, behavior_description)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        pairs = _extract_json_array(text)
        return [
            {"prompt": p["prompt"], "compliant": p["compliant"], "non_compliant": p["non_compliant"]}
            for p in pairs
        ]

    def score_behavior(self, text: str, behavior_description: str) -> Optional[float]:
        system, user = build_judge_prompt(text, behavior_description)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=20,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            out = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return _extract_float(out)
        except Exception as e:
            logger.warning("Anthropic judge call failed: %s", e)
            return None
