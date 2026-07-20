import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from steering_factory import metrics


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert metrics.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert metrics.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_opposite():
    a = np.array([1.0, 2.0, 3.0])
    assert metrics.cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_safe():
    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert metrics.cosine_similarity(a, b) == 0.0


def test_perplexity_uniform_distribution():
    # If every token had logp = log(1/V) for V tokens, perplexity == V.
    V = 50
    logp = [math.log(1.0 / V)] * 20
    ppl = metrics.perplexity_from_logprobs(logp)
    assert ppl == pytest.approx(V, rel=1e-6)


def test_perplexity_perfect_predictions():
    # logp = 0 (probability 1) every time -> perplexity 1.
    ppl = metrics.perplexity_from_logprobs([0.0] * 10)
    assert ppl == pytest.approx(1.0)


def test_perplexity_empty_is_nan():
    assert math.isnan(metrics.perplexity_from_logprobs([]))


def test_kl_divergence_identical_is_zero():
    p = [0.2, 0.3, 0.5]
    assert metrics.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_nonnegative():
    p = [0.9, 0.05, 0.05]
    q = [0.1, 0.1, 0.8]
    assert metrics.kl_divergence(p, q) > 0


def test_js_divergence_symmetric():
    p = [0.9, 0.05, 0.05]
    q = [0.1, 0.1, 0.8]
    assert metrics.js_divergence(p, q) == pytest.approx(metrics.js_divergence(q, p))


def test_js_divergence_bounded():
    p = [1.0, 0.0, 0.0]
    q = [0.0, 0.0, 1.0]
    js = metrics.js_divergence(p, q)
    assert 0 <= js <= math.log(2) + 1e-6


def test_repetition_score_no_repeats():
    text = "the quick brown fox jumps over the lazy dog today"
    score = metrics.repetition_score(text, n=3)
    assert 0.0 <= score < 0.3


def test_repetition_score_degenerate_loop():
    text = " ".join(["the cat sat"] * 10)
    score = metrics.repetition_score(text, n=3)
    assert score > 0.5


def test_distinct_n_diverse_text_high():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    assert metrics.distinct_n(text, n=2) == pytest.approx(1.0)


def test_distinct_n_repetitive_text_low():
    text = " ".join(["foo bar"] * 10)
    assert metrics.distinct_n(text, n=2) < 0.3


def test_convergence_summary_flags_fast_convergence():
    points = [{"n": 2, "cos_to_final": 0.99}, {"n": 8, "cos_to_final": 0.995}]
    summary = metrics.convergence_summary(points)
    assert "near-duplicate" in summary["note"]


def test_convergence_summary_flags_nonconvergence():
    points = [{"n": 2, "cos_to_final": 0.3}, {"n": 32, "cos_to_final": 0.5}]
    summary = metrics.convergence_summary(points)
    assert summary["converged"] is False


def test_convergence_summary_handles_empty():
    summary = metrics.convergence_summary([])
    assert summary["converged"] is None
