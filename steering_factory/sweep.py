from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch

from . import metrics
from .hooks import apply_steering
from .model_utils import LoadedModel

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    token_logprobs: List[float]
    first_step_probs: Optional[Any]  # numpy array or None
    batch_size: int
    batch_wall_time_s: float

    @property
    def latency_s(self) -> float:
        """Amortized per-row cost: `batch_wall_time_s / batch_size`. Under
        batch size 1 this equals the batch's own wall time exactly, so
        existing single-example callers see no change in meaning. Under
        batching it is an estimate, not a measurement of this row alone --
        `batch_wall_time_s` and `batch_size` are kept alongside it
        specifically so downstream cost analysis (e.g. the steering-vs-QLoRA
        comparison) can tell the two apart rather than silently trusting an
        amortized number as if it were measured per-row."""
        return self.batch_wall_time_s / self.batch_size if self.batch_size else self.batch_wall_time_s


def _reduce_scores(scores, gen_ids_row: torch.Tensor) -> tuple[List[float], Optional[Any]]:
    """Extracts only what any caller in this codebase actually uses from
    `generate(..., output_scores=True)`: the chosen token's logprob at each
    step, and the first step's full next-token distribution (for JS
    divergence against a baseline). `scores` is a list of (batch, vocab)
    tensors, one per decode step, for a SINGLE row already sliced out of a
    (possibly batched) generation call.

    This intentionally never returns the full per-step distribution beyond
    step 0 -- retaining all steps x full vocab was the dominant memory cost
    of a generation call before this existed (e.g. ~1.6 GB at batch 16 for a
    262k-vocab model), for data nothing downstream reads."""
    token_logprobs: List[float] = []
    first_step_probs = None
    for i, step_logits_row in enumerate(scores):
        # step_logits_row is (1, vocab) -- a single row kept 2D by the
        # caller's `step_logits[row:row+1]` slice -- so index [0] to reach
        # the (vocab,) distribution before gathering the chosen token.
        logp = torch.log_softmax(step_logits_row.float(), dim=-1)[0]
        if i < gen_ids_row.shape[0]:
            token_logprobs.append(float(logp[gen_ids_row[i]].item()))
        if i == 0:
            first_step_probs = torch.softmax(step_logits_row.float(), dim=-1)[0].cpu().numpy()
    return token_logprobs, first_step_probs


def generate_with_steering_batch(
    loaded: LoadedModel,
    layer_idx: int,
    vector: Optional[torch.Tensor],
    coefficient: float,
    prompt_texts: List[str],
    max_new_tokens: int = 120,
    token_scope: str = "all",
) -> List[GenerationResult]:
    """Batched greedy decode across `prompt_texts` at a FIXED (layer,
    vector, coefficient, token_scope) -- the axis every caller in this
    codebase already holds fixed while looping over examples, since one
    `SteeringHook` instance carries one coefficient for its entire active
    scope. Never batch across coefficients; each row in a batch gets the
    identical intervention.

    Requires left-padding (decoder-only convention for batched generation):
    the true last prompt token is index -1 for every row regardless of
    prompt length, which is also what makes "last"-scope steering correct
    under ragged lengths. `attention_mask` is passed to `generate()` so
    padded positions are excluded from attention; `SteeringHook`'s
    "generated_only" prefill/decode heuristic (seq_len > 1 => prefill) is
    unaffected by batch size since HF's cache-based generation still does
    exactly one shared prefill call for the whole batch, then one shared
    single-token-per-row call per decode step.

    Returns one `GenerationResult` per input prompt, in the same order.
    Determinism: greedy decoding is per-row independent, so results here
    are identical to calling `generate_with_steering` once per prompt at
    batch size 1 -- only wall-clock cost differs (see
    `tests/test_sweep_batching.py`)."""
    if not prompt_texts:
        return []
    device = next(loaded.model.parameters()).device
    layer_module = loaded.layers[layer_idx]

    original_padding_side = loaded.tokenizer.padding_side
    loaded.tokenizer.padding_side = "left"
    try:
        # truncation=True, max_length=1024 (matching extraction.py's own
        # prompt-length cap) is load-bearing, not cosmetic: with
        # padding=True and no length bound, a SINGLE unusually long prompt
        # in the batch (e.g. a JSONSchemaBench prompt embedding a schema up
        # to ~1.28M characters, or a HarmBench `contextual`-subset prompt)
        # pads every other row in the batch out to match it. At batch_size
        # 32 that produced a real OOM (~26 GiB requested in one
        # torch.embedding call) from a single oversized prompt, not from
        # batch_size itself -- this was a missing guard, not a memory-
        # capacity problem to solve by lowering batch_size.
        enc = loaded.tokenizer(prompt_texts, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(device)
    finally:
        loaded.tokenizer.padding_side = original_padding_side

    prompt_len = enc["input_ids"].shape[1]
    batch_size = len(prompt_texts)

    started = time.perf_counter()
    with apply_steering(layer_module, vector, coefficient, token_scope):
        with torch.no_grad():
            out = loaded.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
            )
    batch_wall_time_s = time.perf_counter() - started

    results: List[GenerationResult] = []
    for row in range(batch_size):
        gen_ids_row = out.sequences[row][prompt_len:]
        text = loaded.tokenizer.decode(gen_ids_row, skip_special_tokens=True)
        row_scores = [step_logits[row : row + 1] for step_logits in out.scores]
        token_logprobs, first_step_probs = _reduce_scores(row_scores, gen_ids_row)
        results.append(GenerationResult(
            text=text, token_logprobs=token_logprobs, first_step_probs=first_step_probs,
            batch_size=batch_size, batch_wall_time_s=batch_wall_time_s,
        ))
    return results


def generate_with_steering(
    loaded: LoadedModel,
    layer_idx: int,
    vector: Optional[torch.Tensor],
    coefficient: float,
    prompt_text: str,
    max_new_tokens: int = 120,
    token_scope: str = "all",
):
    """Greedy-decodes (deterministic, so coefficients are comparable) and
    returns (generated_text, per-token logprobs, first-step next-token
    probability distribution as a numpy array).

    `token_scope` is forwarded to the steering hook: "all" steers every
    position including the prompt prefill; "generated_only" leaves the
    prompt untouched and only pushes newly generated tokens; "last" steers
    only the final position of each forward call.

    Thin batch-of-1 wrapper around `generate_with_steering_batch`, kept for
    existing single-example callers/tests -- returns the plain 3-tuple
    instead of a `GenerationResult` (batch_size is always 1 here, so
    `latency_s` and `batch_wall_time_s` are identical)."""
    result = generate_with_steering_batch(
        loaded, layer_idx, vector, coefficient, [prompt_text], max_new_tokens, token_scope,
    )[0]
    return result.text, result.token_logprobs, result.first_step_probs


def coefficient_sweep(
    loaded: LoadedModel,
    layer_idx: int,
    vector: torch.Tensor,
    prompt_text: str,
    coefficients: List[float],
    max_new_tokens: int = 120,
    judge=None,
    behavior_description: Optional[str] = None,
    token_scope: str = "all",
) -> Dict[str, Any]:
    """Runs generation at coefficient=0 (baseline) plus every requested
    coefficient, computing fluency/divergence/degeneration metrics for
    each, and an optional LLM-judge behavior score. Returns the raw rows
    plus baseline perplexity for normalization."""
    coeffs = sorted(set(coefficients) | {0.0})

    baseline_text, baseline_logp, baseline_probs = generate_with_steering(
        loaded, layer_idx, vector, 0.0, prompt_text, max_new_tokens, token_scope
    )
    baseline_ppl = metrics.perplexity_from_logprobs(baseline_logp)

    rows = []
    for coeff in coeffs:
        if coeff == 0.0:
            text, logp, probs = baseline_text, baseline_logp, baseline_probs
        else:
            text, logp, probs = generate_with_steering(
                loaded, layer_idx, vector, coeff, prompt_text, max_new_tokens, token_scope
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
