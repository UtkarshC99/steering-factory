from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def perplexity_from_logprobs(token_logprobs: Sequence[float]) -> float:
    """token_logprobs: natural-log probabilities of each generated token.
    Standard perplexity = exp(-mean(logp))."""
    arr = np.asarray(token_logprobs, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.exp(-arr.mean()))


def _normalize_dist(p: Sequence[float], eps: float = 1e-10) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    return p / p.sum()


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p = _normalize_dist(p)
    q = _normalize_dist(q)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence, symmetric and bounded in [0, ln(2)]."""
    p = _normalize_dist(p)
    q = _normalize_dist(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def repetition_score(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeats of an earlier n-gram in the
    same text. 0 = no repetition, closer to 1 = degenerate looping."""
    tokens = text.split()
    grams = _ngrams(tokens, n)
    if len(grams) == 0:
        return 0.0
    seen = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams)


def distinct_n(text: str, n: int = 2) -> float:
    """Unique n-grams / total n-grams. Higher = more lexically diverse."""
    tokens = text.split()
    grams = _ngrams(tokens, n)
    if len(grams) == 0:
        return 0.0
    return len(set(grams)) / len(grams)


def convergence_summary(convergence_points: List[dict]) -> dict:
    """Given [{'n': int, 'cos_to_final': float}, ...], returns a small
    summary used to flag suspiciously-fast or non-converging vectors."""
    pts = [p for p in convergence_points if p.get("cos_to_final") is not None]
    if not pts:
        return {"converged": None, "note": "no convergence data"}
    pts = sorted(pts, key=lambda p: p["n"])
    last = pts[-1]["cos_to_final"]
    early = pts[0]["cos_to_final"]
    if early is not None and early > 0.97:
        note = "Converges almost immediately -- check pairs aren't near-duplicates."
    elif last < 0.9:
        note = "Still moving at full dataset size -- add more, more diverse pairs."
    else:
        note = "Looks reasonably converged."
    return {"converged": last >= 0.9, "final_cos": last, "early_cos": early, "note": note}
