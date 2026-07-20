"""Safe, inspectable behavior registry used by recipes and synthetic data."""
from __future__ import annotations

from typing import Dict

from .experiment_types import BehaviorSpec

DEFAULT_BEHAVIORS: Dict[str, BehaviorSpec] = {
    "defensive_refusal": BehaviorSpec(
        id="defensive_refusal", family="safety",
        description="Decline unsafe requests while offering a brief, safe alternative.",
    ),
    "domain_classification": BehaviorSpec(
        id="domain_classification", family="classification",
        description="Return exactly one allowed class label for the input.",
    ),
    "appropriate_abstention": BehaviorSpec(
        id="appropriate_abstention", family="calibration",
        description="Acknowledge uncertainty when the question cannot be answered reliably.",
    ),
    "structured_output": BehaviorSpec(
        id="structured_output", family="structured_output",
        description="Return valid structured output that satisfies the supplied schema.",
    ),
}


def get_behavior(behavior_id: str, description: str | None = None) -> BehaviorSpec:
    if behavior_id in DEFAULT_BEHAVIORS:
        return DEFAULT_BEHAVIORS[behavior_id]
    if not description:
        raise KeyError(f"Unknown behavior '{behavior_id}'; provide behavior.description in the manifest.")
    return BehaviorSpec(id=behavior_id, family="custom", description=description)
