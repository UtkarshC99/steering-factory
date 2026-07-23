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


def _require_reviewed_dataset_card(config: Dict[str, Any], adapter_name: str, dataset_id: str, license_str: str) -> None:
    """Every real-benchmark adapter requires an explicit
    `dataset.reviewed_dataset_card: true` in the manifest before it will
    load anything. This is a deliberate speed bump, not a technical
    necessity: these are licensed/gated datasets (HarmBench and XSTest in
    particular ship real harmful-request text), and the manifest is the
    place provenance and license already get recorded
    (`runner._load_normalized_records` writes `data/<recipe>.provenance.json`
    including `dataset` verbatim) -- so the flag doubles as an auditable
    record that a human looked at the dataset card, not just a runtime gate.
    """
    if not config.get("reviewed_dataset_card"):
        raise RuntimeError(
            f"{adapter_name} loads {dataset_id} ({license_str} license), which requires reviewing "
            f"the dataset card (terms of use, intended research use, any access request) before use. "
            f"Set `dataset.reviewed_dataset_card: true` in the manifest once you have done so."
        )


class HarmBenchAdapter:
    """HarmBench (Mazeika et al. 2024, MIT-licensed harness/code; dataset
    access is gated on HuggingFace and requires agreeing to share contact
    info) behavior prompts, for the `harmful_instruction_compliance` /
    `jailbreak_susceptibility` safety behaviors.

    Schema (as documented by the HarmBench paper/repo -- VERIFY against the
    live dataset card / viewer once you have gated access on Colab, before
    trusting a real run; this adapter cannot be validated against live
    rows from this environment): `Behavior` (the harmful instruction
    text), `FunctionalCategory`, `SemanticCategory`, `Tags`, `BehaviorID`.

    HarmBench ships prompts, not (positive, negative) completions -- this
    adapter synthesizes the contrast: `positive` is a short, generic
    refusal (the SAFE completion, matching this taxonomy's sign
    convention), `negative` is a short, generic compliance stub. Neither
    string reproduces or elaborates on the harmful behavior itself.
    """
    name = "harmbench"
    default_dataset = "walledai/HarmBench"

    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        _require_reviewed_dataset_card(config, "HarmBenchAdapter", config.get("dataset", self.default_dataset), "MIT (code) / gated access (data)")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the optional 'datasets' dependency to load benchmark adapters.") from exc
        ds = load_dataset(config.get("dataset", self.default_dataset), config.get("subset", "standard"),
                           split=config.get("split", "train"), cache_dir=config.get("cache_dir"))
        mapping = config.get("columns", {"prompt": "Behavior", "category": "SemanticCategory", "id": "BehaviorID"})
        rows = []
        for index, row in enumerate(ds):
            rows.append({
                "id": str(row.get(mapping.get("id", "BehaviorID"), index)),
                "prompt": row.get(mapping.get("prompt", "Behavior")),
                "category": row.get(mapping.get("category", "SemanticCategory")),
                "source": config.get("dataset", self.default_dataset),
            })
        return rows

    def normalize(self, record: Dict[str, Any], behavior: BehaviorSpec) -> ContrastiveExample:
        prompt = str(record["prompt"])
        return ContrastiveExample(
            id=str(record.get("id", hashlib.sha256(prompt.encode()).hexdigest()[:16])), prompt=prompt,
            positive="I can't help with that. I can suggest safe, lawful alternatives if that would help.",
            negative="Sure, here is how to do that.",
            behavior_id=behavior.id, split=str(record.get("split", "test")),
            source=str(record.get("source", self.name)), category=record.get("category"),
            metadata={"benchmark": "HarmBench"},
        )


class AbstentionBenchAdapter:
    """AbstentionBench (facebook/AbstentionBench, cc-by-nc-4.0, requires
    `trust_remote_code=True` per its dataset card) for the
    `appropriate_abstention` behavior.

    Schema (confirmed from the dataset card): `question` (str),
    `reference_answers` (list[str] | None), `should_abstain` (bool),
    `metadata_json` (dict). `positive` is an abstention phrase when
    `should_abstain` is True, else the first reference answer; `negative`
    is the opposite, so the extracted direction points toward
    *appropriate* abstention rather than abstention in general.
    """
    name = "abstentionbench"
    default_dataset = "facebook/AbstentionBench"

    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        _require_reviewed_dataset_card(config, "AbstentionBenchAdapter", config.get("dataset", self.default_dataset), "cc-by-nc-4.0")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the optional 'datasets' dependency to load benchmark adapters.") from exc
        ds = load_dataset(config.get("dataset", self.default_dataset), config.get("subset"),
                           split=config.get("split", "train"), cache_dir=config.get("cache_dir"),
                           trust_remote_code=True)
        rows = []
        for index, row in enumerate(ds):
            answers = row.get("reference_answers") or []
            rows.append({
                "id": str(index), "prompt": row.get("question"),
                "answerable": not bool(row.get("should_abstain")),
                "answer": answers[0] if answers else None,
                "source": config.get("dataset", self.default_dataset),
            })
        return rows

    def normalize(self, record: Dict[str, Any], behavior: BehaviorSpec) -> ContrastiveExample:
        prompt = str(record["prompt"])
        answerable = bool(record.get("answerable"))
        answer = record.get("answer") or "the answer"
        abstain_text = "I don't have enough reliable information to answer that."
        positive = answer if answerable else abstain_text
        negative = abstain_text if answerable else answer
        return ContrastiveExample(
            id=str(record.get("id", hashlib.sha256(prompt.encode()).hexdigest()[:16])), prompt=prompt,
            positive=str(positive), negative=str(negative), behavior_id=behavior.id,
            split=str(record.get("split", "test")), source=str(record.get("source", self.name)),
            category=record.get("category"),
            metadata={"benchmark": "AbstentionBench", "answerable": answerable, "target": record.get("answer")},
        )


class JSONSchemaBenchAdapter:
    """JSONSchemaBench (epfl-dlab/JSONSchemaBench, MIT license, public) for
    the `structured_output` behavior.

    Schema (confirmed from the dataset card): `json_schema` (str, the
    schema definition), `unique_id` (str). Has real train/validation/test
    splits and 11 domain subsets (Github_*, Glaiveai2K, JsonSchemaStore,
    Kubernetes, Snowplow, WashingtonPost) -- pass `subset` in the manifest
    to pick one. `positive` is a request-formatted prompt whose target is
    "produce valid JSON for this schema" (the actual instance is not
    generated here -- evaluation scores parse/schema-validity against the
    schema itself, same as the existing structured_output evaluator);
    `negative` is a deliberately non-JSON free-text stub.
    """
    name = "jsonschemabench"
    default_dataset = "epfl-dlab/JSONSchemaBench"

    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        _require_reviewed_dataset_card(config, "JSONSchemaBenchAdapter", config.get("dataset", self.default_dataset), "MIT")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the optional 'datasets' dependency to load benchmark adapters.") from exc
        ds = load_dataset(config.get("dataset", self.default_dataset), config.get("subset", "JsonSchemaStore"),
                           split=config.get("split", "train"), cache_dir=config.get("cache_dir"))
        rows = []
        for row in ds:
            rows.append({
                "id": str(row.get("unique_id")), "schema": row.get("json_schema"),
                "source": config.get("dataset", self.default_dataset),
            })
        return rows

    def normalize(self, record: Dict[str, Any], behavior: BehaviorSpec) -> ContrastiveExample:
        schema = record.get("schema")
        prompt = f"Return a single JSON object that strictly satisfies this JSON Schema:\n{schema}"
        return ContrastiveExample(
            id=str(record.get("id", hashlib.sha256(prompt.encode()).hexdigest()[:16])), prompt=prompt,
            positive="{}", negative="I cannot provide that in JSON format, but here is a description instead.",
            behavior_id=behavior.id, split=str(record.get("split", "test")),
            source=str(record.get("source", self.name)), category=record.get("category"),
            metadata={"benchmark": "JSONSchemaBench", "schema": schema},
        )


class XSTestAdapter:
    """XSTest (paul-rottger/xstest via walledai/XSTest mirror, CC-BY-4.0
    prompts) -- a BENIGN OVER-REFUSAL control set, not an extraction
    source. 250 safe prompts across 10 categories that a model should
    answer, paired with 200 matched "contrast" prompts (superficially
    similar, genuinely unsafe) that a model should refuse.

    Schema (confirmed from the paper/repo): `prompt` (str), `type` (str;
    unsafe/contrast categories are `contrast_`-prefixed, matching each safe
    category by suffix) -- VERIFY exact column names against the live
    dataset card once you have access, since the HF mirror's exact field
    names were not independently confirmed from this environment (gated).

    Unlike the other three adapters, this one does not synthesize a
    (positive, negative) pair for steering extraction -- XSTest exists to
    MEASURE false refusal on genuinely safe prompts, so every row is
    assigned split="test" regardless of the manifest's split plan (there
    is nothing to steer toward here; see behaviors.py's docstring on the
    defensive-research framing) and `metadata.is_safe_control` marks it so
    the safety metric bundle (evaluators.py) can compute false-refusal
    rate specifically on this split.
    """
    name = "xstest"
    default_dataset = "walledai/XSTest"

    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        _require_reviewed_dataset_card(config, "XSTestAdapter", config.get("dataset", self.default_dataset), "CC-BY-4.0")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the optional 'datasets' dependency to load benchmark adapters.") from exc
        ds = load_dataset(config.get("dataset", self.default_dataset), split=config.get("split", "train"),
                           cache_dir=config.get("cache_dir"))
        rows = []
        for index, row in enumerate(ds):
            prompt_type = row.get("type", "")
            rows.append({
                "id": str(index), "prompt": row.get("prompt"), "category": prompt_type,
                "is_unsafe_contrast": str(prompt_type).startswith("contrast_"),
                "source": config.get("dataset", self.default_dataset),
            })
        return rows

    def normalize(self, record: Dict[str, Any], behavior: BehaviorSpec) -> ContrastiveExample:
        prompt = str(record["prompt"])
        is_unsafe = bool(record.get("is_unsafe_contrast"))
        return ContrastiveExample(
            id=str(record.get("id", hashlib.sha256(prompt.encode()).hexdigest()[:16])), prompt=prompt,
            positive="", negative="",  # not used for extraction -- evaluation-only control set
            behavior_id=behavior.id, split="test",  # always test: nothing to select on here, see class docstring
            source=str(record.get("source", self.name)), category=record.get("category"),
            metadata={"benchmark": "XSTest", "is_safe_control": not is_unsafe, "is_unsafe_contrast": is_unsafe},
        )


ADAPTERS = {
    "local_jsonl": LocalJsonlAdapter,
    "huggingface": HuggingFaceAdapter,
    "harmbench": HarmBenchAdapter,
    "abstentionbench": AbstentionBenchAdapter,
    "jsonschemabench": JSONSchemaBenchAdapter,
    "xstest": XSTestAdapter,
}


def build_adapter(name: str):
    try:
        return ADAPTERS[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown dataset adapter '{name}'. Available: {sorted(ADAPTERS)}") from exc
