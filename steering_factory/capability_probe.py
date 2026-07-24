"""Benign general-capability regression control.

A recipe's own quality metric only measures the target behavior; it says
nothing about collateral damage elsewhere. The comparison literature's
central finding (SKOP-vs-LoRA, ReFT/LoRA/IA3 low-resource studies) is that
steering and fine-tuning arms diverge MOST on collateral utility loss, not
target-behavior efficacy -- so "steering hit X on the target task" is an
incomplete claim without also reporting what happened to ordinary,
unrelated capability under the same intervention.

CAPABILITY_PROBES is a small, fixed set of benign general-knowledge/
arithmetic/instruction-following prompts with known-good answers, held
constant across every recipe and every arm -- it is deliberately NOT
behavior-specific (no safety/classification/abstention/schema content),
so it measures "did this intervention break ordinary capability" rather
than re-measuring the target behavior itself under another name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CapabilityProbe:
    id: str
    prompt: str
    expected_substrings: tuple  # any one matching (case-insensitive) counts as correct


CAPABILITY_PROBES: List[CapabilityProbe] = [
    CapabilityProbe("arithmetic_1", "What is 12 plus 7? Answer with just the number.", ("19",)),
    CapabilityProbe("arithmetic_2", "What is 9 times 6? Answer with just the number.", ("54",)),
    CapabilityProbe("geography_1", "What is the capital of France? Answer with just the city name.", ("paris",)),
    CapabilityProbe("geography_2", "What is the capital of Japan? Answer with just the city name.", ("tokyo",)),
    CapabilityProbe("word_count", "How many letters are in the word 'cat'? Answer with just the number.", ("3", "three")),
    CapabilityProbe("sequence", "What comes next in this sequence: 2, 4, 6, 8, ? Answer with just the number.", ("10", "ten")),
    CapabilityProbe("instruction_following", "Repeat exactly the word 'banana' and nothing else.", ("banana",)),
    CapabilityProbe("basic_fact", "Is the sky blue on a clear day? Answer yes or no.", ("yes",)),
]


def score_capability_probe(prediction: str, probe: CapabilityProbe) -> float:
    lowered = prediction.lower()
    return float(any(marker in lowered for marker in probe.expected_substrings))


def capability_quality(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Mean correctness across probe rows, or None if there are none (so
    callers can distinguish "not measured" from "measured, scored 0")."""
    if not rows:
        return None
    return sum(row["correct"] for row in rows) / len(rows)


def run_capability_probe_steering(
    loaded: Any, vector: Optional[Any], layer_idx: int, coefficient: float, token_scope: str, max_new_tokens: int = 24,
) -> List[Dict[str, Any]]:
    """Runs every CAPABILITY_PROBES prompt through the steering arm's
    generation path at a given (vector, coefficient, token_scope) --
    typically called once with coefficient=0.0 (base model) and once with
    the recipe's winning coefficient (steered model) so the ratio of the
    two `capability_quality()` values is the retention score. Reuses
    `sweep.generate_with_steering_batch` (the same primitive
    `_evaluate_vector_grid` uses) rather than a separate generation path,
    so probe generations are produced identically to every other row."""
    from . import sweep
    from .model_utils import format_chat

    # format_chat, matching _evaluate_vector_grid's own fix -- the QLoRA
    # side of this same probe (run_capability_probe_qlora below) already
    # templates via evaluate_qlora_adapter/evaluate_base_model, so this was
    # the one capability-probe path still sending an un-templated prompt.
    prompts = [format_chat(loaded.tokenizer, probe.prompt) for probe in CAPABILITY_PROBES]
    results = sweep.generate_with_steering_batch(loaded, layer_idx, vector, coefficient, prompts, max_new_tokens, token_scope)
    rows = []
    for probe, result in zip(CAPABILITY_PROBES, results):
        rows.append({
            "probe_id": probe.id, "prompt": probe.prompt, "output": result.text,
            "correct": score_capability_probe(result.text, probe),
        })
    return rows


def run_capability_probe_qlora(
    model_name: str, adapter_dir: Optional[str], max_new_tokens: int = 24,
    trust_remote_code: bool = False, batch_size: int = 16,
    quantization: Optional[str] = "4bit", dtype: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Runs every CAPABILITY_PROBES prompt through the QLoRA arm's
    generation path -- `adapter_dir=None` for the base-model (before)
    measurement, or a trained adapter's directory for the after
    measurement. Reuses `finetune.evaluate_qlora_adapter`'s model-loading
    and generation code rather than a separate implementation; when
    `adapter_dir` is None, loads the plain base model instead of a PEFT
    model wrapping it.

    `quantization`/`dtype` default to the pre-existing hardcoded 4bit
    behavior and should match whatever the adapter (if any) was trained at
    -- see the identical note on `evaluate_qlora_adapter`.
    """
    from .finetune import evaluate_base_model, evaluate_qlora_adapter

    probe_examples = [{"id": probe.id, "prompt": probe.prompt, "behavior_id": "capability_probe", "split": "test"}
                       for probe in CAPABILITY_PROBES]
    if adapter_dir is None:
        result = evaluate_base_model(probe_examples, model_name, max_new_tokens=max_new_tokens,
                                      trust_remote_code=trust_remote_code, batch_size=batch_size,
                                      quantization=quantization, dtype=dtype)
    else:
        result = evaluate_qlora_adapter(probe_examples, model_name, adapter_dir, max_new_tokens=max_new_tokens,
                                         trust_remote_code=trust_remote_code, batch_size=batch_size,
                                         quantization=quantization, dtype=dtype)

    by_id = {probe.id: probe for probe in CAPABILITY_PROBES}
    rows = []
    for row in result["rows"]:
        probe = by_id[row["example_id"]]
        rows.append({"probe_id": probe.id, "prompt": probe.prompt, "output": row["output"],
                      "correct": score_capability_probe(row["output"], probe)})
    return rows


def capability_retention(before_rows: List[Dict[str, Any]], after_rows: List[Dict[str, Any]]) -> Optional[float]:
    """after / before capability quality. None if either side has no rows
    to measure, so "not measured" is never silently reported as "retained
    everything" (a 1.0 that isn't real) or "lost everything" (a 0.0 that
    isn't real)."""
    before = capability_quality(before_rows)
    after = capability_quality(after_rows)
    if before is None or after is None:
        return None
    if before == 0:
        return None  # base model already failed every probe -- ratio is meaningless
    return after / before
