"""Dataset adapters with deterministic splits and provenance preservation.

Network-backed Hugging Face datasets are intentionally loaded only at runtime,
never at import time, which keeps unit tests and offline analysis lightweight.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .experiment_types import BehaviorSpec, ContrastiveExample, SplitPlan


def stable_split(records: List[Dict[str, Any]], plan: SplitPlan) -> Dict[str, str]:
    """Return deterministic, optionally category-stratified steer/validation/test assignments."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(plan.group_key, "__all__"))].append(record)
    assignments: Dict[str, str] = {}
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: str(r.get("id", r.get("prompt", ""))))
        rng = random.Random(f"{plan.seed}:{key}")
        rng.shuffle(rows)
        n = len(rows)
        steer_n = max(1, int(n * plan.steer_fraction)) if n >= 3 else 0
        val_n = max(1, int(n * plan.validation_fraction)) if n - steer_n >= 2 else 0
        for index, record in enumerate(rows):
            rid = str(record.get("id", record.get("prompt")))
            assignments[rid] = "steer" if index < steer_n else "validation" if index < steer_n + val_n else "test"
    return assignments


def leakage_report(records: List[Dict[str, Any]], assignments: Dict[str, str]) -> Dict[str, Any]:
    seen, duplicates = {}, []
    for record in records:
        rid = str(record.get("id", record.get("prompt")))
        normalized = " ".join(str(record.get("prompt", "")).lower().split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in seen and assignments.get(seen[digest]) != assignments.get(rid):
            duplicates.append({"first_id": seen[digest], "duplicate_id": rid})
        seen[digest] = rid
    return {"records": len(records), "exact_cross_split_duplicates": duplicates, "has_leakage": bool(duplicates)}


class LocalJsonlAdapter:
    name = "local_jsonl"

    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = Path(config["path"])
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def normalize(self, record: Dict[str, Any], behavior: BehaviorSpec) -> ContrastiveExample:
        prompt = str(record["prompt"])
        return ContrastiveExample(
            id=str(record.get("id", hashlib.sha256(prompt.encode()).hexdigest()[:16])), prompt=prompt,
            positive=str(record.get("positive", record.get("target", ""))),
            negative=str(record.get("negative", "")), behavior_id=behavior.id,
            split=str(record.get("split", "test")), source=str(record.get("source", self.name)),
            category=record.get("category"), metadata=dict(record.get("metadata", {})),
        )


class HuggingFaceAdapter:
    """Generic column-mapping adapter for Hub datasets and controlled local caching."""
    name = "huggingface"

    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the optional 'datasets' dependency to load benchmark adapters.") from exc
        ds = load_dataset(config["dataset"], config.get("subset"), split=config.get("split", "train"), cache_dir=config.get("cache_dir"))
        mapping = config.get("columns", {})
        rows = []
        for index, row in enumerate(ds):
            item = {"id": str(row.get(mapping.get("id", "id"), index))}
            for target in ("prompt", "target", "category", "positive", "negative"):
                column = mapping.get(target)
                if column is not None:
                    item[target] = row.get(column)
            item["source"] = config["dataset"]
            rows.append(item)
        return rows

    def normalize(self, record: Dict[str, Any], behavior: BehaviorSpec) -> ContrastiveExample:
        return LocalJsonlAdapter().normalize(record, behavior)


ADAPTERS = {"local_jsonl": LocalJsonlAdapter, "huggingface": HuggingFaceAdapter}


def build_adapter(name: str):
    try:
        return ADAPTERS[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown dataset adapter '{name}'. Available: {sorted(ADAPTERS)}") from exc
