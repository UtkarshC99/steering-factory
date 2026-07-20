"""Stable public types for reproducible steering experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Protocol


@dataclass(frozen=True)
class BehaviorSpec:
    id: str
    description: str
    family: str
    safe_default: bool = True
    prompt_template: Optional[str] = None


@dataclass(frozen=True)
class ContrastiveExample:
    id: str
    prompt: str
    positive: str
    negative: str
    behavior_id: str
    split: Literal["steer", "validation", "test"]
    source: str
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitPlan:
    seed: int = 17
    steer_fraction: float = 0.2
    validation_fraction: float = 0.2
    group_key: str = "category"


@dataclass(frozen=True)
class InterventionSpec:
    layers: List[int]
    coefficient: float
    token_scope: Literal["all", "generated_only", "last"] = "all"
    normalize: bool = False
    composition: Literal["single", "sum", "orthogonal"] = "single"


@dataclass
class RunArtifact:
    run_id: str
    root: str
    manifest_hash: str
    status: Literal["running", "completed", "failed"]
    started_at: str
    ended_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetAdapter(Protocol):
    name: str

    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]: ...

    def normalize(self, record: Dict[str, Any], behavior: BehaviorSpec) -> ContrastiveExample: ...


class ExtractionMethod(Protocol):
    name: str


class Evaluator(Protocol):
    name: str

    def score(self, prediction: str, example: Dict[str, Any]) -> Dict[str, Any]: ...


class FineTuneBackend(Protocol):
    name: str

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
