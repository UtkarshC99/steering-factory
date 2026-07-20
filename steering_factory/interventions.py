"""Intervention construction, compatibility checks, and vector composition."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Dict, Iterable, List

import torch

from .hooks import apply_steering
from .model_utils import LoadedModel


def normalized(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-8)


def orthogonalize(vectors: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    basis: List[torch.Tensor] = []
    for vector in vectors:
        current = vector.float().clone()
        for prior in basis:
            current = current - torch.dot(current, prior) * prior
        if current.norm() > 1e-8:
            basis.append(normalized(current))
    return basis


def compose(vectors: Iterable[torch.Tensor], mode: str = "sum") -> torch.Tensor:
    items = list(vectors)
    if not items:
        raise ValueError("At least one vector is required")
    if mode == "single":
        return items[0]
    if mode == "orthogonal":
        items = orthogonalize(items)
    return torch.stack(items).sum(dim=0)


def validate_vector_metadata(metadata: Dict[str, object], loaded: LoadedModel) -> None:
    expected = metadata.get("hidden_size")
    if expected is not None and int(expected) != loaded.hidden_size:
        raise ValueError(f"Vector hidden size {expected} is incompatible with model hidden size {loaded.hidden_size}.")


@contextmanager
def apply_intervention(
    loaded: LoadedModel,
    vector: torch.Tensor,
    layers: Iterable[int],
    coefficient: float,
    token_scope: str = "all",
):
    with ExitStack() as stack:
        for layer in layers:
            if layer < 0 or layer >= loaded.num_layers:
                raise IndexError(f"Layer {layer} outside [0, {loaded.num_layers - 1}]")
            stack.enter_context(apply_steering(loaded.layers[layer], vector, coefficient, token_scope))
        yield
