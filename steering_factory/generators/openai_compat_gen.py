from __future__ import annotations

import logging
import os
from typing import List, Dict, Optional

from .base import GeneratorBackend
from .templates import build_pair_generation_prompt, build_judge_prompt
from .anthropic_gen import _extract_json_array, _extract_float

logger = logging.getLogger(__name__)


class OpenAICompatGenerator(GeneratorBackend):
    """Works against any OpenAI-compatible chat completions endpoint --
    vLLM, Ollama (`/v1`), LM Studio, text-generation-webui, or actual
    OpenAI. Point base_url at a local server for a fully offline pipeline."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY"):
        from openai import OpenAI  # local import

        api_key = os.environ.get(api_key_env, "not-needed-for-local-servers")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def generate_pairs(
        self,
        topic: str,
        persona: Optional[str],
        n_pairs: int,
        behavior_description: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        system, user = build_pair_generation_prompt(topic, persona, n_pairs, behavior_description)
        text = self._chat(system, user, max_tokens=4000)
        pairs = _extract_json_array(text)
        return [
            {"prompt": p["prompt"], "compliant": p["compliant"], "non_compliant": p["non_compliant"]}
            for p in pairs
        ]

    def score_behavior(self, text: str, behavior_description: str) -> Optional[float]:
        system, user = build_judge_prompt(text, behavior_description)
        try:
            out = self._chat(system, user, max_tokens=20)
            return _extract_float(out)
        except Exception as e:
            logger.warning("OpenAI-compatible judge call failed: %s", e)
            return None
