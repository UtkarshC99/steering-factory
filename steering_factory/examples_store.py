from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from .storage import _slugify


class ExamplesStore:
    """One JSON file per topic under examples_dir, holding the editable
    list of {"prompt", "compliant", "non_compliant"} pairs plus metadata
    about how they were generated."""

    def __init__(self, examples_dir: str):
        self.examples_dir = examples_dir
        os.makedirs(self.examples_dir, exist_ok=True)

    def _path(self, topic: str) -> str:
        return os.path.join(self.examples_dir, f"{_slugify(topic)}.json")

    def save(
        self,
        topic: str,
        pairs: List[Dict[str, str]],
        persona: Optional[str] = None,
        generator_backend: Optional[str] = None,
    ):
        payload = {
            "topic": topic,
            "persona": persona,
            "generator_backend": generator_backend,
            "updated_at": time.time(),
            "pairs": pairs,
        }
        with open(self._path(topic), "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, topic: str) -> Optional[Dict[str, Any]]:
        path = self._path(topic)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    def list_topics(self) -> List[str]:
        topics = []
        for fname in sorted(os.listdir(self.examples_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(self.examples_dir, fname), "r") as f:
                    try:
                        topics.append(json.load(f)["topic"])
                    except Exception:
                        continue
        return topics
