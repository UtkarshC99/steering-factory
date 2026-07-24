"""Batching correctness for extraction._collect_diffs / _pooled_activations_batch.

Uses a real (tiny, no-download) Llama-architecture model -- see
tests/_tiny_model.py for why Llama and not GPT-2. These tests caught a
real, previously-latent bug in hooks.ActivationCapture.last_token_activation:
its index formula (`attention_mask.sum(dim=1) - 1`) is only correct for
right-padding. Under left-padding it pointed into the padding region
instead of the real last token, silently, because every caller before
batching existed used batch size 1 with an all-ones mask (where the two
formulas happen to coincide).
"""
import pytest
import torch

from steering_factory import extraction

from _tiny_model import build_loaded_model


@pytest.fixture(scope="module")
def loaded():
    return build_loaded_model(seed=11)


RAGGED_PAIRS = [
    {"prompt": "w4 w5", "compliant": "w10 w11 w12", "non_compliant": "w20 w21"},
    {"prompt": "w4 w5 w6", "compliant": "w10 w11", "non_compliant": "w20 w21 w22 w23"},
    {"prompt": "w4", "compliant": "w10 w11 w12 w13", "non_compliant": "w20"},
    {"prompt": "w4 w5 w6 w7", "compliant": "w10", "non_compliant": "w20 w21 w22"},
    {"prompt": "w4 w5", "compliant": "w10 w11 w12", "non_compliant": "w20 w21"},
]


@pytest.mark.parametrize("pooling", ["last", "mean"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 16])
def test_collect_diffs_matches_across_batch_sizes(loaded, pooling, batch_size):
    """The core equivalence guarantee: whatever batch_size chunks pairs
    into, the resulting diffs must be identical (within float tolerance)
    to running every pair alone at batch_size=1. This is what makes
    batching safe to enable without changing which steering vector gets
    extracted."""
    reference = extraction._collect_diffs(loaded, 0, RAGGED_PAIRS, pooling, batch_size=1)
    batched = extraction._collect_diffs(loaded, 0, RAGGED_PAIRS, pooling, batch_size=batch_size)
    assert torch.allclose(reference, batched, atol=1e-4)


def test_pair_order_preserved_under_batching(loaded):
    # Convergence tracking in mean_diff_vector/pca_vector reads diffs[:n]
    # and depends on pairs staying in their original order, not batch
    # order -- distinct pairs at distinct positions must not get shuffled.
    diffs = extraction._collect_diffs(loaded, 0, RAGGED_PAIRS, "last", batch_size=4)
    assert diffs.shape[0] == len(RAGGED_PAIRS)
    # The 1st and 5th pair are identical inputs -- their diffs must match each other.
    assert torch.allclose(diffs[0], diffs[4], atol=1e-4)


@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_pooled_activations_unaffected_by_logits_to_keep(loaded, pooling):
    """_pooled_activations_batch passes logits_to_keep=1 to the forward
    call -- a real CUDA OOM traced to this call materializing the full
    (batch, seq_len, vocab) logits tensor on every extraction forward pass
    even though only the hooked decoder layer's hidden states are ever
    read (the LM head runs strictly AFTER the hooked layer, so slicing its
    output can't affect what ActivationCapture captures). This proves that
    claim directly rather than by architectural reasoning alone: pooled
    activations must be bit-identical with and without the flag."""
    texts = ["w4 w5 w6", "w10 w11", "w4 w5 w6 w7 w8"]
    with_flag = extraction._pooled_activations_batch(loaded, 0, texts, pooling)

    # Re-run the exact same forward pass without logits_to_keep, bypassing
    # the function under test to get an independent reference.
    device = next(loaded.model.parameters()).device
    layer_module = loaded.layers[0]
    original_padding_side = loaded.tokenizer.padding_side
    loaded.tokenizer.padding_side = "left"
    try:
        enc = loaded.tokenizer(texts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    finally:
        loaded.tokenizer.padding_side = original_padding_side
    enc = {k: v.to(device) for k, v in enc.items()}

    from steering_factory.hooks import ActivationCapture
    with ActivationCapture(layer_module) as cap:
        with torch.no_grad():
            loaded.model(**enc)
    without_flag = (
        cap.last_token_activation(enc.get("attention_mask"))
        if pooling == "last" else cap.mean_activation(enc.get("attention_mask"))
    ).float().cpu()

    assert torch.equal(with_flag, without_flag)


def test_last_token_activation_correct_under_left_padding_regression():
    """Direct regression test for the bug this batching work uncovered:
    ActivationCapture.last_token_activation must find each row's true last
    real token via its attention mask, not via a real-token-count-based
    offset that only works for right-padding."""
    from steering_factory.hooks import ActivationCapture

    loaded_model = build_loaded_model(seed=11)
    tokenizer = loaded_model.tokenizer
    tokenizer.padding_side = "left"
    short_text, long_text = "w4 w5\nw10 w11 w12", "w4 w5 w6 w7\nw20 w21 w22 w23 w24 w25 w26"

    enc_batch = tokenizer([short_text, long_text], return_tensors="pt", padding=True)
    enc_single = tokenizer([short_text], return_tensors="pt")

    with ActivationCapture(loaded_model.layers[0]) as cap_batch:
        with torch.no_grad():
            loaded_model.model(**enc_batch)
    pooled_batch = cap_batch.last_token_activation(enc_batch["attention_mask"])

    with ActivationCapture(loaded_model.layers[0]) as cap_single:
        with torch.no_grad():
            loaded_model.model(**enc_single)
    pooled_single = cap_single.last_token_activation(enc_single["attention_mask"])

    assert not torch.allclose(pooled_batch[0], torch.zeros_like(pooled_batch[0]))
    assert torch.allclose(pooled_batch[0], pooled_single[0], atol=1e-4)


def test_batch_size_forwarded_from_extract_dispatcher(loaded):
    # extract()'s **kwargs forwarding must reach _collect_diffs for every
    # closed-form method without raising an unexpected-kwarg error.
    for method in ("mean_diff", "pca", "whitened_mean_diff"):
        result = extraction.extract(method, loaded, 0, RAGGED_PAIRS, batch_size=4)
        assert result.vector.shape == (loaded.hidden_size,)


def test_optimized_method_pops_batch_size_before_forwarding(loaded):
    # optimized_vector has no batch_size parameter -- extract() must pop
    # batch_size out of kwargs for the init-vector computation rather than
    # forward it into optimized_vector's own (unbatched) gradient loop.
    result = extraction.extract("optimized", loaded, 0, RAGGED_PAIRS, batch_size=4, steps=2)
    assert result.vector.shape == (loaded.hidden_size,)
