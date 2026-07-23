"""Safe, inspectable behavior registry used by recipes and synthetic data.

The safety sub-taxonomy below follows the defensive-research framing: each
entry extracts the direction associated with an unsafe/undesirable
behavior only to MEASURE how present it is (held out) and to demonstrate
hardening against it (ablation/conditional steering in interventions.py),
never to ship the direction as an operational jailbreak. `positive` in
every safety ContrastiveExample is always the SAFE completion (what we
want the model biased toward); `negative` is the unsafe one -- this
matches the sign convention `defensive_refusal` already uses, so a
positive coefficient always steers toward safety across the whole
taxonomy, not just one behavior.
"""
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
    "harmful_instruction_compliance": BehaviorSpec(
        id="harmful_instruction_compliance", family="safety",
        description=(
            "Refuse to carry out a harmful instruction (e.g. HarmBench-style behavior "
            "requests: weapons, cybercrime, illegal activity) rather than complying with it."
        ),
    ),
    "jailbreak_susceptibility": BehaviorSpec(
        id="jailbreak_susceptibility", family="safety",
        description=(
            "Maintain refusal of a harmful request even when it is wrapped in a jailbreak "
            "framing (roleplay, hypothetical, instruction override) rather than being "
            "talked into compliance by the framing alone."
        ),
    ),
    "deception_sycophancy": BehaviorSpec(
        id="deception_sycophancy", family="safety",
        description=(
            "Give an honest, accurate answer even when the prompt signals the user wants "
            "a particular (possibly wrong) answer, rather than agreeing to flatter or "
            "avoid disagreement."
        ),
    ),
    "pii_leakage": BehaviorSpec(
        id="pii_leakage", family="safety",
        description=(
            "Decline to fabricate, guess, or repeat back personal identifying information "
            "about a real individual, rather than producing a plausible-sounding answer."
        ),
    ),
}


def get_behavior(behavior_id: str, description: str | None = None) -> BehaviorSpec:
    if behavior_id in DEFAULT_BEHAVIORS:
        return DEFAULT_BEHAVIORS[behavior_id]
    if not description:
        raise KeyError(f"Unknown behavior '{behavior_id}'; provide behavior.description in the manifest.")
    return BehaviorSpec(id=behavior_id, family="custom", description=description)
