from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import numpy as np
import torch

from .callbacks import RunCallback, _safe
from .hooks import ActivationCapture, SteeringHook
from .model_utils import LoadedModel, format_chat

logger = logging.getLogger(__name__)

Pooling = Literal["last", "mean"]


@dataclass
class ExtractionResult:
    vector: torch.Tensor          # (hidden,)
    method: str
    layer_idx: int
    pooling: Pooling
    num_pairs: int
    convergence: List[Dict] = field(default_factory=list)  # [{"n": int, "cos_to_final": float}]


def _encode(tokenizer, text: str, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    return {k: v.to(device) for k, v in enc.items()}


def _pooled_activations_batch(
    loaded: LoadedModel, layer_idx: int, texts: List[str], pooling: Pooling, max_length: int = 1024,
) -> torch.Tensor:
    """Runs one left-padded, batched forward pass over `texts` and returns
    the pooled activation at `layer_idx` for each row, shape (len(texts),
    hidden). A plain forward pass (no autoregression, no KV cache), so
    unlike generation this needs no scores/logprob handling -- just
    `ActivationCapture`'s existing attention_mask-aware pooling, which
    already returns one row per batch item correctly under left-padding.

    Left-padding (not right) matters here for the same reason as
    generation: `last_token_activation` finds each row's real last token
    via its attention mask regardless of pad side, but this project's
    target architectures (Qwen3/Gemma-3, both RoPE + attention_mask-aware,
    verified against a Llama-family stand-in in
    tests/test_extraction_batching.py) only produce identical hidden
    states to an unpadded single-row forward pass under LEFT padding --
    right-padding shifts position handling for the trailing pad tokens in
    a way that does not affect the real tokens' own hidden states, but
    left-padding is the standard decoder-only convention and is what's
    verified, so it's what's used everywhere in this codebase.

    `max_length` (default 1024) is the tokenizer truncation cap, overridable
    per-recipe (runner.py's decoding.max_length) for recipes whose prompts
    routinely approach it mid-content (structured_output_real's JSON
    schemas) -- the extracted vector should come from the same effective
    input length the eval-generation path sees, not a shorter one."""
    device = next(loaded.model.parameters()).device
    layer_module = loaded.layers[layer_idx]

    original_padding_side = loaded.tokenizer.padding_side
    loaded.tokenizer.padding_side = "left"
    try:
        enc = loaded.tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    finally:
        loaded.tokenizer.padding_side = original_padding_side
    enc = {k: v.to(device) for k, v in enc.items()}

    with ActivationCapture(layer_module) as cap:
        with torch.no_grad():
            loaded.model(**enc, logits_to_keep=1)
    if pooling == "last":
        pooled = cap.last_token_activation(enc.get("attention_mask"))
    else:
        pooled = cap.mean_activation(enc.get("attention_mask"))
    return pooled.float().cpu()


def _collect_diffs(
    loaded: LoadedModel, layer_idx: int, pairs: List[Dict], pooling: Pooling, batch_size: int = 16,
    max_length: int = 1024,
) -> torch.Tensor:
    """Computes the pooled (compliant - non_compliant) activation diff for
    every pair, batching `batch_size` pairs' worth of forward passes at a
    time (both sides together, so one chunk of N pairs is one batch of 2N
    texts) rather than one forward pass per side per pair. Pair order is
    preserved exactly, which matters for the convergence-tracking partial
    sums in mean_diff_vector/pca_vector (they read diffs[:n] and depend on
    pair order, not batch order)."""
    prompt_texts = [format_chat(loaded.tokenizer, p["prompt"]) for p in pairs]
    all_texts: List[str] = []
    for prompt_text, pair in zip(prompt_texts, pairs):
        all_texts.append(prompt_text + pair["compliant"])
        all_texts.append(prompt_text + pair["non_compliant"])

    pooled_rows: List[torch.Tensor] = []
    for start in range(0, len(all_texts), batch_size * 2):
        chunk = all_texts[start : start + batch_size * 2]
        pooled_rows.append(_pooled_activations_batch(loaded, layer_idx, chunk, pooling, max_length))
    pooled = torch.cat(pooled_rows, dim=0)  # (2 * n_pairs, hidden): [pos0, neg0, pos1, neg1, ...]

    diffs = pooled[0::2] - pooled[1::2]  # (n_pairs, hidden)
    return diffs


def mean_diff_vector(
    loaded: LoadedModel,
    layer_idx: int,
    pairs: List[Dict],
    pooling: Pooling = "last",
    track_convergence: bool = True,
    convergence_steps: Optional[List[int]] = None,
    batch_size: int = 16,
    max_length: int = 1024,
) -> ExtractionResult:
    diffs = _collect_diffs(loaded, layer_idx, pairs, pooling, batch_size, max_length)
    final_vec = diffs.mean(dim=0)

    convergence = []
    if track_convergence and len(pairs) >= 4:
        steps = convergence_steps or sorted(
            set([max(2, len(pairs) // 8), max(2, len(pairs) // 4),
                 max(2, len(pairs) // 2), len(pairs)])
        )
        final_norm = final_vec / (final_vec.norm() + 1e-8)
        for n in steps:
            n = min(n, len(pairs))
            partial = diffs[:n].mean(dim=0)
            partial_norm = partial / (partial.norm() + 1e-8)
            cos = float(torch.dot(partial_norm, final_norm))
            convergence.append({"n": n, "cos_to_final": cos})

    return ExtractionResult(
        vector=final_vec, method="mean_diff", layer_idx=layer_idx,
        pooling=pooling, num_pairs=len(pairs), convergence=convergence,
    )


def pca_vector(
    loaded: LoadedModel,
    layer_idx: int,
    pairs: List[Dict],
    pooling: Pooling = "last",
    track_convergence: bool = True,
    batch_size: int = 16,
    max_length: int = 1024,
) -> ExtractionResult:
    diffs = _collect_diffs(loaded, layer_idx, pairs, pooling, batch_size, max_length)
    centered = diffs - diffs.mean(dim=0, keepdim=True)
    # SVD on the (n_pairs, hidden) diff matrix; top right-singular vector
    # is the principal direction of the activation differences.
    u, s, vt = torch.linalg.svd(centered, full_matrices=False)
    top_pc = vt[0]

    # Orient sign so it points toward "compliant", using the mean diff as
    # a reference direction (PCA components have arbitrary sign).
    mean_diff = diffs.mean(dim=0)
    if torch.dot(top_pc, mean_diff) < 0:
        top_pc = -top_pc
    # Scale to roughly match the norm of the mean-diff vector so
    # coefficients are comparable across methods.
    scaled = top_pc * (mean_diff.norm() / (top_pc.norm() + 1e-8))

    convergence = []
    if track_convergence and len(pairs) >= 4:
        final_norm = scaled / (scaled.norm() + 1e-8)
        for n in sorted(set([max(2, len(pairs) // 4), max(2, len(pairs) // 2), len(pairs)])):
            sub = diffs[:n]
            sub_c = sub - sub.mean(dim=0, keepdim=True)
            if sub_c.shape[0] < 2:
                continue
            _, _, sub_vt = torch.linalg.svd(sub_c, full_matrices=False)
            pc = sub_vt[0]
            if torch.dot(pc, mean_diff) < 0:
                pc = -pc
            pc_norm = pc / (pc.norm() + 1e-8)
            convergence.append({"n": n, "cos_to_final": float(torch.dot(pc_norm, final_norm))})

    return ExtractionResult(
        vector=scaled, method="pca", layer_idx=layer_idx,
        pooling=pooling, num_pairs=len(pairs), convergence=convergence,
    )


def whitened_mean_diff_vector(
    loaded: LoadedModel,
    layer_idx: int,
    pairs: List[Dict],
    pooling: Pooling = "last",
    ridge: float = 1e-3,
    batch_size: int = 16,
    max_length: int = 1024,
) -> ExtractionResult:
    """Regularized Fisher/whitened contrastive direction.

    It downweights activation dimensions whose pair differences are noisy,
    offering a useful ablation between raw CAA and a learned optimizer.
    """
    diffs = _collect_diffs(loaded, layer_idx, pairs, pooling, batch_size, max_length).float()
    mean = diffs.mean(dim=0)
    centered = diffs - mean
    # Diagonal covariance is deliberate: full covariance is unstable when
    # examples << hidden dimension, the normal research setting here.
    variance = centered.pow(2).mean(dim=0).clamp_min(ridge)
    vector = mean / variance
    vector = vector * (mean.norm() / vector.norm().clamp_min(1e-8))
    return ExtractionResult(vector=vector, method="whitened_mean_diff", layer_idx=layer_idx,
                            pooling=pooling, num_pairs=len(pairs))


def _completion_logprob(loaded: LoadedModel, prompt_text: str, completion_text: str, device) -> torch.Tensor:
    """Sum log p(token_i | token_<i) for tokens belonging to `completion_text`
    only (teacher-forced), under whatever steering hook is currently active
    on the model. Returns a scalar tensor that supports backprop."""
    tokenizer = loaded.tokenizer
    # Same truncation cap as _encode/_pooled_activations_batch above: an
    # unbounded prompt+completion here isn't a batch-padding risk (this is
    # always batch size 1), but a single pathologically long pair could
    # still blow up attention memory on its own, so it gets the same
    # defensive bound as every other tokenizer call site in this module.
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    full_ids = tokenizer(prompt_text + completion_text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    prompt_len = prompt_ids.shape[1]

    full_ids = full_ids.to(device)
    out = loaded.model(input_ids=full_ids)
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq-1)

    # completion tokens are those at position >= prompt_len - 1 in the
    # shifted (seq-1) indexing
    start = max(prompt_len - 1, 0)
    completion_logp = token_logp[:, start:]
    if completion_logp.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return completion_logp.sum()


def optimized_vector(
    loaded: LoadedModel,
    layer_idx: int,
    pairs: List[Dict],
    init_vector: torch.Tensor,
    steps: int = 40,
    lr: float = 0.05,
    coefficient: float = 1.0,
    max_pairs_per_step: Optional[int] = None,
    callback: Optional[RunCallback] = None,
) -> ExtractionResult:
    """BiPO-lite: optimizes the vector by gradient descent to maximize
    logp(compliant) - logp(non_compliant) under teacher forcing, with the
    candidate vector injected at `layer_idx` via the same steering hook used
    at inference time.

    This is a simplified, single-vector version of Bi-directional
    Preference Optimization (Cao et al., 2024) -- correct in spirit, not a
    faithful reproduction. It is slow (full forward+backward per pair per
    step); keep `pairs` small (8-16) and `steps` modest (20-60) unless you
    have a lot of GPU time to spare. Cross-check the result against the
    held-out spectrum -- this method overfits small pair sets easily.

    `callback`, if given, receives one `"opt_step"` event per gradient step
    (this is the one extraction method with a real per-step loss curve --
    the closed-form methods below have nothing analogous to emit). Always
    `None` from the CLI; see `callbacks.py`.
    """
    device = next(loaded.model.parameters()).device
    layer_module = loaded.layers[layer_idx]

    vec = init_vector.clone().detach().to(device).float()
    vec.requires_grad_(True)
    optimizer = torch.optim.Adam([vec], lr=lr)

    use_pairs = pairs if not max_pairs_per_step else pairs[:max_pairs_per_step]

    history = []
    for step in range(steps):
        optimizer.zero_grad()
        step_loss = torch.tensor(0.0, device=device)
        for pair in use_pairs:
            prompt_text = format_chat(loaded.tokenizer, pair["prompt"])
            hook = SteeringHook(layer_module, vec, coefficient)
            with hook:
                pos_logp = _completion_logprob(loaded, prompt_text, pair["compliant"], device)
            with hook:
                neg_logp = _completion_logprob(loaded, prompt_text, pair["non_compliant"], device)
            step_loss = step_loss - (pos_logp - neg_logp)
        step_loss = step_loss / max(len(use_pairs), 1)
        step_loss.backward()
        grad_norm = float(vec.grad.norm().detach().cpu()) if vec.grad is not None else None
        optimizer.step()
        history.append(float(step_loss.detach().cpu()))
        logger.debug("optimized_vector step %d loss %.4f", step, history[-1])
        _safe(callback, {"arm": "extract", "event": "opt_step", "i": step + 1, "n": steps,
                          "layer_idx": layer_idx, "loss": history[-1], "grad_norm": grad_norm})

    final_vec = vec.detach().cpu()
    convergence = [{"n": i + 1, "cos_to_final": None, "loss": v} for i, v in enumerate(history)]

    return ExtractionResult(
        vector=final_vec, method="optimized", layer_idx=layer_idx,
        pooling="n/a", num_pairs=len(pairs), convergence=convergence,
    )


def extract(
    method: str,
    loaded: LoadedModel,
    layer_idx: int,
    pairs: List[Dict],
    pooling: Pooling = "last",
    callback: Optional[RunCallback] = None,
    **kwargs,
) -> ExtractionResult:
    """`callback` is accepted here for every method for a uniform call site
    in runner.py, but only `optimized` (BiPO-lite) has a real per-step loss
    curve to emit -- the closed-form methods (mean_diff/pca/whitened) are
    each a single computation with nothing analogous to a training step, so
    `callback` is silently unused for them rather than forwarded and
    rejected as an unexpected kwarg."""
    if method == "mean_diff":
        return mean_diff_vector(loaded, layer_idx, pairs, pooling, **kwargs)
    if method == "pca":
        return pca_vector(loaded, layer_idx, pairs, pooling, **kwargs)
    if method in ("whitened_mean_diff", "fisher"):
        return whitened_mean_diff_vector(loaded, layer_idx, pairs, pooling, **kwargs)
    if method == "optimized":
        # batch_size is only meaningful for the init-vector's closed-form
        # forward passes; optimized_vector's own per-pair gradient loop
        # below is NOT batched (each pair needs its own backward pass
        # under teacher forcing, which batching would change the semantics
        # of, not just the speed) and has no batch_size/max_length parameter
        # -- pop both here rather than forward them and hit an
        # unexpected-kwarg error. max_length still governs the init
        # vector's own closed-form forward passes below.
        init_batch_size = kwargs.pop("batch_size", 16)
        init_max_length = kwargs.pop("max_length", 1024)
        init = mean_diff_vector(loaded, layer_idx, pairs, pooling, track_convergence=False,
                                 batch_size=init_batch_size, max_length=init_max_length).vector
        return optimized_vector(loaded, layer_idx, pairs, init_vector=init, callback=callback, **kwargs)
    raise ValueError(f"Unknown extraction method: {method}")
