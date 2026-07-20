from __future__ import annotations

import logging
from typing import List, Dict, Optional

import torch

from .base import GeneratorBackend
from .templates import build_pair_generation_prompt, build_judge_prompt
from .anthropic_gen import _extract_json_array, _extract_float
from ..model_utils import format_chat

logger = logging.getLogger(__name__)


class LocalHFGenerator(GeneratorBackend):
    """Self-generates pairs (and judge scores) using a local HF model --
    either the same model being steered, or a separate one you point at
    in config.yaml under generator.local_hf. Fully offline, lower quality
    than an API model for this kind of structured-output task, so expect
    to hand-edit the results more.
    """

    def __init__(self, loaded_model):
        # loaded_model: a model_utils.LoadedModel, passed in by app.py so
        # we don't duplicate model-loading logic here.
        self.loaded = loaded_model

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str:
        device = next(self.loaded.model.parameters()).device
        prompt_text = format_chat(self.loaded.tokenizer, user, system=system)
        enc = self.loaded.tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.loaded.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.loaded.tokenizer.pad_token_id
                or self.loaded.tokenizer.eos_token_id,
            )
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        return self.loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate_pairs(
        self,
        topic: str,
        persona: Optional[str],
        n_pairs: int,
        behavior_description: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        system, user = build_pair_generation_prompt(topic, persona, n_pairs, behavior_description)
        text = self._generate(system, user, max_new_tokens=200 * max(1, n_pairs // 4 + 1))
        try:
            pairs = _extract_json_array(text)
        except Exception as e:
            raise ValueError(
                "Local model did not return parseable JSON. Smaller models "
                "often struggle with this -- try a smaller n_pairs, a "
                "different model, or write/edit pairs by hand in the UI. "
                f"Raw output: {text[:300]}"
            ) from e
        return [
            {"prompt": p["prompt"], "compliant": p["compliant"], "non_compliant": p["non_compliant"]}
            for p in pairs
        ]

    def score_behavior(self, text: str, behavior_description: str) -> Optional[float]:
        system, user = build_judge_prompt(text, behavior_description)
        try:
            out = self._generate(system, user, max_new_tokens=10)
            return _extract_float(out)
        except Exception as e:
            logger.warning("Local HF judge call failed: %s", e)
            return None
