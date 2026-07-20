from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class GeneratorBackend(ABC):
    """Produces compliant/non_compliant contrastive pairs for a topic, and
    optionally acts as an LLM judge to score how strongly a piece of text
    exhibits a target behavior (used by the coefficient sweep)."""

    @abstractmethod
    def generate_pairs(
        self,
        topic: str,
        persona: Optional[str],
        n_pairs: int,
        behavior_description: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Returns a list of dicts: {"prompt", "compliant", "non_compliant"}."""
        raise NotImplementedError

    def score_behavior(self, text: str, behavior_description: str) -> Optional[float]:
        """Optional: returns a 0-1 score of how strongly `text` exhibits
        the behavior described by `behavior_description`. Default
        implementation declines (returns None) -- subclasses can implement
        this with an LLM-judge call. Sweep falls back to JS-divergence-only
        ranking when this returns None."""
        return None
