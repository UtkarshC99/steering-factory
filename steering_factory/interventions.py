"""Intervention construction, compatibility checks, and vector composition."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Dict, Iterable, List, Optional

import torch

from .hooks import apply_ablation, apply_conditional_steering, apply_steering
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
    """Additive steering across `layers` -- the original, and still
    default, intervention. See `apply_ablation_intervention` and
    `apply_conditional_intervention` below for the safety-hardening/gated
    alternatives added alongside this."""
    with ExitStack() as stack:
        for layer in layers:
            if layer < 0 or layer >= loaded.num_layers:
                raise IndexError(f"Layer {layer} outside [0, {loaded.num_layers - 1}]")
            stack.enter_context(apply_steering(loaded.layers[layer], vector, coefficient, token_scope))
        yield


@contextmanager
def apply_ablation_intervention(
    loaded: LoadedModel,
    vector: torch.Tensor,
    layers: Iterable[int],
    strength: float = 1.0,
    token_scope: str = "all",
):
    """Directional ablation ("abliteration") across `layers`: projects
    `vector`'s direction OUT of the residual stream instead of adding it
    in. The defensive-hardening demonstration for the safety pillar --
    given a direction associated with an unsafe behavior (extracted the
    same way any other steering vector is), this removes that component
    from every position's activation rather than pushing further toward
    or away from it. Bounded by construction (see AblationHook's
    docstring): `strength` can only remove what's already present, never
    add an arbitrary new push, unlike `apply_intervention`'s coefficient."""
    with ExitStack() as stack:
        for layer in layers:
            if layer < 0 or layer >= loaded.num_layers:
                raise IndexError(f"Layer {layer} outside [0, {loaded.num_layers - 1}]")
            stack.enter_context(apply_ablation(loaded.layers[layer], vector, strength, token_scope))
        yield


@contextmanager
def apply_conditional_intervention(
    loaded: LoadedModel,
    vector: torch.Tensor,
    layers: Iterable[int],
    coefficient: float,
    condition_vector: Optional[torch.Tensor],
    threshold: float,
    token_scope: str = "all",
):
    """CAST-style gated steering across `layers`: the additive push only
    fires at positions whose activation projects onto `condition_vector`
    above `threshold`. Key for cutting benign over-refusal -- an
    unconditional additive vector (`apply_intervention`) pushes every
    input toward the behavior regardless of whether it actually needs it;
    gating on a condition direction confines the push to inputs that
    trigger the condition, leaving genuinely benign inputs unaffected."""
    with ExitStack() as stack:
        for layer in layers:
            if layer < 0 or layer >= loaded.num_layers:
                raise IndexError(f"Layer {layer} outside [0, {loaded.num_layers - 1}]")
            stack.enter_context(apply_conditional_steering(
                loaded.layers[layer], vector, coefficient, condition_vector, threshold, token_scope,
            ))
        yield
