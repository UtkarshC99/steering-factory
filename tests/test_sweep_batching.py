"""Batching correctness for sweep.generate_with_steering_batch.

These tests use a real (tiny, randomly-initialized, no-download) GPT-2
model rather than mocking `generate()` -- steering/batching correctness
depends on real tensor shapes flowing through HF's generation loop and a
hand-written mock would only test the mock.
"""
import numpy as np
import pytest
import torch

from steering_factory.sweep import generate_with_steering, generate_with_steering_batch

from _tiny_model import build_loaded_model


@pytest.fixture(scope="module")
def loaded():
    return build_loaded_model(seed=7)


RAGGED_PROMPTS = ["w4 w5 w6", "w4 w5", "w4 w5 w6 w7 w8", "w10 w11 w12 w13"]


@pytest.mark.parametrize("token_scope", ["all", "generated_only", "last"])
def test_batched_generation_matches_sequential_for_ragged_lengths(loaded, token_scope):
    """The whole point of batching: identical output whether prompts are
    run one at a time or together in a left-padded batch, for every
    token_scope. This is what makes batching safe to turn on without
    compromising quality -- greedy decoding is per-row independent, so a
    mismatch here would mean padding or the SteeringHook's positional
    logic leaked across rows."""
    torch.manual_seed(0)
    vector = torch.randn(loaded.hidden_size)
    batched = generate_with_steering_batch(
        loaded, 0, vector, 1.5, RAGGED_PROMPTS, max_new_tokens=6, token_scope=token_scope,
    )
    assert len(batched) == len(RAGGED_PROMPTS)

    for prompt, batch_result in zip(RAGGED_PROMPTS, batched):
        text, logp, probs = generate_with_steering(
            loaded, 0, vector, 1.5, prompt, max_new_tokens=6, token_scope=token_scope,
        )
        assert text == batch_result.text
        assert len(logp) == len(batch_result.token_logprobs)
        assert logp == pytest.approx(batch_result.token_logprobs, abs=1e-4)
        assert (probs is None) == (batch_result.first_step_probs is None)
        if probs is not None:
            assert np.allclose(probs, batch_result.first_step_probs, atol=1e-4)


def test_batched_generation_matches_sequential_for_equal_length_prompts(loaded):
    # Equal-length prompts need no padding at all -- a simpler case worth
    # pinning separately from the ragged-length one above.
    torch.manual_seed(1)
    vector = torch.randn(loaded.hidden_size)
    prompts = ["w4 w5 w6", "w10 w11 w12", "w20 w21 w22"]
    batched = generate_with_steering_batch(loaded, 1, vector, -1.0, prompts, max_new_tokens=4)
    for prompt, batch_result in zip(prompts, batched):
        text, logp, _ = generate_with_steering(loaded, 1, vector, -1.0, prompt, max_new_tokens=4)
        assert text == batch_result.text
        assert logp == pytest.approx(batch_result.token_logprobs, abs=1e-4)


def test_batch_size_and_wall_time_recorded_on_every_row(loaded):
    vector = torch.randn(loaded.hidden_size)
    results = generate_with_steering_batch(loaded, 0, vector, 1.0, RAGGED_PROMPTS, max_new_tokens=3)
    assert all(r.batch_size == len(RAGGED_PROMPTS) for r in results)
    # All rows share one measured batch wall time (it's one generate() call).
    wall_times = {r.batch_wall_time_s for r in results}
    assert len(wall_times) == 1
    assert next(iter(wall_times)) > 0


def test_latency_s_is_amortized_batch_wall_time(loaded):
    vector = torch.randn(loaded.hidden_size)
    results = generate_with_steering_batch(loaded, 0, vector, 1.0, RAGGED_PROMPTS, max_new_tokens=3)
    for r in results:
        assert r.latency_s == pytest.approx(r.batch_wall_time_s / r.batch_size)


def test_single_prompt_batch_of_one_latency_equals_wall_time(loaded):
    # At batch size 1, amortized latency must equal the measured wall time
    # exactly -- this is what keeps generate_with_steering's existing
    # meaning ("this one generation's cost") unchanged for callers who
    # never opt into batching.
    vector = torch.randn(loaded.hidden_size)
    [result] = generate_with_steering_batch(loaded, 0, vector, 1.0, ["w4 w5 w6"], max_new_tokens=3)
    assert result.latency_s == result.batch_wall_time_s


def test_empty_prompt_list_returns_empty():
    loaded_model = build_loaded_model(seed=2)
    assert generate_with_steering_batch(loaded_model, 0, None, 0.0, [], max_new_tokens=3) == []


def test_tokenizer_padding_side_restored_after_batch_call(loaded):
    # generate_with_steering_batch temporarily flips padding_side to "left"
    # (required for decoder-only batched generation) -- it must not leak
    # that mutation onto the shared tokenizer for unrelated callers.
    original = loaded.tokenizer.padding_side
    generate_with_steering_batch(loaded, 0, torch.randn(loaded.hidden_size), 1.0, RAGGED_PROMPTS, max_new_tokens=2)
    assert loaded.tokenizer.padding_side == original
