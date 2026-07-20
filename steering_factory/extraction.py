from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import numpy as np
import torch

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


def _pair_activation_diff(
    loaded: LoadedModel, layer_idx: int, pair: Dict, pooling: Pooling
) -> torch.Tensor:
    """Runs the model on (prompt + compliant) and (prompt + non_compliant)
    and returns the pooled activation difference at the given layer."""
    device = next(loaded.model.parameters()).device
    layer_module = loaded.layers[layer_idx]
    prompt_text = format_chat(loaded.tokenizer, pair["prompt"])

    diffs = []
    for key in ("compliant", "non_compliant"):
        text = prompt_text + pair[key]
        enc = _encode(loaded.tokenizer, text, device)
        with ActivationCapture(layer_module) as cap:
            with torch.no_grad():
                loaded.model(**enc)
        if pooling == "last":
            act = cap.last_token_activation(enc.get("attention_mask"))
        else:
            act = cap.mean_activation(enc.get("attention_mask"))
        diffs.append(act.squeeze(0).float().cpu())

    pos_act, neg_act = diffs
    return pos_act - neg_act


def _collect_diffs(
    loaded: LoadedModel, layer_idx: int, pairs: List[Dict], pooling: Pooling
) -> torch.Tensor:
    rows = [_pair_activation_diff(loaded, layer_idx, p, pooling) for p in pairs]
    return torch.stack(rows, dim=0)  # (n_pairs, hidden)


def mean_diff_vector(
    loaded: LoadedModel,
    layer_idx: int,
    pairs: List[Dict],
    pooling: Pooling = "last",
    track_convergence: bool = True,
    convergence_steps: Optional[List[int]] = None,
) -> ExtractionResult:
    diffs = _collect_diffs(loaded, layer_idx, pairs, pooling)
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
) -> ExtractionResult:
    diffs = _collect_diffs(loaded, layer_idx, pairs, pooling)
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
) -> ExtractionResult:
    """Regularized Fisher/whitened contrastive direction.

    It downweights activation dimensions whose pair differences are noisy,
    offering a useful ablation between raw CAA and a learned optimizer.
    """
    diffs = _collect_diffs(loaded, layer_idx, pairs, pooling).float()
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
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt_text + completion_text, return_tensors="pt").input_ids
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
        optimizer.step()
        history.append(float(step_loss.detach().cpu()))
        logger.debug("optimized_vector step %d loss %.4f", step, history[-1])

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
    **kwargs,
) -> ExtractionResult:
    if method == "mean_diff":
        return mean_diff_vector(loaded, layer_idx, pairs, pooling, **kwargs)
    if method == "pca":
        return pca_vector(loaded, layer_idx, pairs, pooling, **kwargs)
    if method in ("whitened_mean_diff", "fisher"):
        return whitened_mean_diff_vector(loaded, layer_idx, pairs, pooling, **kwargs)
    if method == "optimized":
        init = mean_diff_vector(loaded, layer_idx, pairs, pooling, track_convergence=False).vector
        return optimized_vector(loaded, layer_idx, pairs, init_vector=init, **kwargs)
    raise ValueError(f"Unknown extraction method: {method}")
