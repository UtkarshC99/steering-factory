from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any

import torch

from . import metrics
from .hooks import apply_steering
from .model_utils import LoadedModel

logger = logging.getLogger(__name__)


def generate_with_steering(
    loaded: LoadedModel,
    layer_idx: int,
    vector: Optional[torch.Tensor],
    coefficient: float,
    prompt_text: str,
    max_new_tokens: int = 120,
):
    """Greedy-decodes (deterministic, so coefficients are comparable) and
    returns (generated_text, per-token logprobs, first-step next-token
    probability distribution as a numpy array)."""
    device = next(loaded.model.parameters()).device
    layer_module = loaded.layers[layer_idx]
    enc = loaded.tokenizer(prompt_text, return_tensors="pt").to(device)

    with apply_steering(layer_module, vector, coefficient):
        with torch.no_grad():
            out = loaded.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
            )

    gen_ids = out.sequences[0][enc["input_ids"].shape[1]:]
    text = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)

    token_logprobs = []
    first_step_probs = None
    for i, step_logits in enumerate(out.scores):
        logp = torch.log_softmax(step_logits.float(), dim=-1)[0]
        if i < gen_ids.shape[0]:
            token_logprobs.append(float(logp[gen_ids[i]].item()))
        if i == 0:
            first_step_probs = torch.softmax(step_logits.float(), dim=-1)[0].cpu().numpy()

    return text, token_logprobs, first_step_probs


def coefficient_sweep(
    loaded: LoadedModel,
    layer_idx: int,
    vector: torch.Tensor,
    prompt_text: str,
    coefficients: List[float],
    max_new_tokens: int = 120,
    judge=None,
    behavior_description: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs generation at coefficient=0 (baseline) plus every requested
    coefficient, computing fluency/divergence/degeneration metrics for
    each, and an optional LLM-judge behavior score. Returns the raw rows
    plus baseline perplexity for normalization."""
    coeffs = sorted(set(coefficients) | {0.0})

    baseline_text, baseline_logp, baseline_probs = generate_with_steering(
        loaded, layer_idx, vector, 0.0, prompt_text, max_new_tokens
    )
    baseline_ppl = metrics.perplexity_from_logprobs(baseline_logp)

    rows = []
    for coeff in coeffs:
        if coeff == 0.0:
            text, logp, probs = baseline_text, baseline_logp, baseline_probs
        else:
            text, logp, probs = generate_with_steering(
                loaded, layer_idx, vector, coeff, prompt_text, max_new_tokens
            )
        ppl = metrics.perplexity_from_logprobs(logp)
        js = (
            metrics.js_divergence(probs, baseline_probs)
            if probs is not None and baseline_probs is not None
            else None
        )
        row = {
            "coefficient": coeff,
            "text": text,
            "perplexity": ppl,
            "perplexity_ratio": (ppl / baseline_ppl) if baseline_ppl and baseline_ppl > 0 else None,
            "js_divergence_vs_baseline": js,
            "repetition_score": metrics.repetition_score(text),
            "distinct_2": metrics.distinct_n(text, n=2),
        }
        if judge is not None and behavior_description:
            try:
                row["behavior_score"] = judge.score_behavior(text, behavior_description)
            except Exception as e:  # judge is best-effort, never fail the sweep
                logger.warning("Judge scoring failed for coefficient %s: %s", coeff, e)
                row["behavior_score"] = None
        rows.append(row)

    return {"rows": rows, "baseline_perplexity": baseline_ppl}


def _row_score(row: Dict[str, Any]) -> float:
    if row.get("behavior_score") is not None:
        return row["behavior_score"]
    return row.get("js_divergence_vs_baseline") or 0.0


def suggest_best_coefficients(
    rows: List[Dict[str, Any]], fluency_cap_ratio: float = 1.6
):
    """Picks the coefficient (one positive, one negative) that maximizes
    behavior shift (judge score if available, else JS divergence from
    baseline) subject to a perplexity-ratio ceiling, so a degenerate but
    "high divergence" output doesn't win by default."""
    candidates = [
        r for r in rows
        if r["coefficient"] != 0
        and (r["perplexity_ratio"] is None or r["perplexity_ratio"] <= fluency_cap_ratio)
        and r["repetition_score"] < 0.5
    ]
    pos = [r for r in candidates if r["coefficient"] > 0]
    neg = [r for r in candidates if r["coefficient"] < 0]
    best_pos = max(pos, key=_row_score, default=None)
    best_neg = max(neg, key=_row_score, default=None)
    return best_pos, best_neg
