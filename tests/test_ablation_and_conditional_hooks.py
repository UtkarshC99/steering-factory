"""Tests for AblationHook and ConditionalSteeringHook (hooks.py) -- the
directional-ablation and CAST-style gated-steering intervention modes
added alongside the existing additive SteeringHook.

Exact-math tests use the deterministic _EchoLayer stand-in (same pattern
as test_hooks_token_scope.py); batching-correctness tests run the tiny
Llama fixture, since left-padded batch safety is architecture-dependent
(GPT-2-family architectures were found NOT to be batch-safe here without
extra handling -- see tests/_tiny_model.py's docstring -- so verifying
against the Llama-family stand-in this project actually targets matters,
not a formality).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from steering_factory.hooks import (
    AblationHook,
    ConditionalSteeringHook,
    apply_ablation,
    apply_conditional_steering,
)
from steering_factory.interventions import apply_ablation_intervention, apply_conditional_intervention

from _tiny_model import build_loaded_model


class _EchoLayer(nn.Module):
    def forward(self, hidden_states):
        return (hidden_states,)


# --- AblationHook: exact math ---------------------------------------------------

def test_ablation_full_strength_zeroes_projection_onto_vector():
    layer = _EchoLayer()
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0])  # unit vector along dim 0
    hidden_in = torch.tensor([[[5.0, 2.0, -3.0, 1.0]]])  # (1, 1, 4)
    with apply_ablation(layer, vector, strength=1.0, token_scope="all"):
        (out,) = layer(hidden_in)
    # Component along dim 0 removed entirely; other dims untouched.
    assert torch.allclose(out, torch.tensor([[[0.0, 2.0, -3.0, 1.0]]]), atol=1e-5)


def test_ablation_partial_strength_scales_the_removed_component():
    layer = _EchoLayer()
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden_in = torch.tensor([[[5.0, 2.0, -3.0, 1.0]]])
    with apply_ablation(layer, vector, strength=0.5, token_scope="all"):
        (out,) = layer(hidden_in)
    # Half the projected component removed: 5.0 - 0.5*5.0 = 2.5.
    assert torch.allclose(out, torch.tensor([[[2.5, 2.0, -3.0, 1.0]]]), atol=1e-5)


def test_ablation_normalizes_a_non_unit_vector_internally():
    layer = _EchoLayer()
    vector = torch.tensor([3.0, 0.0, 0.0, 0.0])  # NOT unit norm
    hidden_in = torch.tensor([[[5.0, 2.0, -3.0, 1.0]]])
    with apply_ablation(layer, vector, strength=1.0, token_scope="all"):
        (out,) = layer(hidden_in)
    assert torch.allclose(out, torch.tensor([[[0.0, 2.0, -3.0, 1.0]]]), atol=1e-5)


def test_ablation_last_scope_only_ablates_final_position():
    layer = _EchoLayer()
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden_in = torch.tensor([[[5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]])  # (1, 2, 4)
    with apply_ablation(layer, vector, strength=1.0, token_scope="last"):
        (out,) = layer(hidden_in)
    assert torch.allclose(out[:, 0, :], torch.tensor([[5.0, 0.0, 0.0, 0.0]]))  # untouched
    assert torch.allclose(out[:, 1, :], torch.zeros(1, 4), atol=1e-5)  # ablated


def test_ablation_zero_strength_or_none_vector_is_noop():
    layer = _EchoLayer()
    hidden_in = torch.tensor([[[5.0, 2.0, -3.0, 1.0]]])
    with apply_ablation(layer, None, strength=1.0, token_scope="all"):
        (out,) = layer(hidden_in.clone())
    assert torch.allclose(out, hidden_in)

    with apply_ablation(layer, torch.tensor([1.0, 0.0, 0.0, 0.0]), strength=0.0, token_scope="all"):
        (out,) = layer(hidden_in.clone())
    assert torch.allclose(out, hidden_in)


def test_ablation_hook_removes_handle_on_exit():
    layer = _EchoLayer()
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hook = AblationHook(layer, vector, strength=1.0, token_scope="all")
    with hook:
        assert hook._handle is not None
    assert hook._handle is None
    hidden_in = torch.tensor([[[5.0, 0.0, 0.0, 0.0]]])
    (out,) = layer(hidden_in.clone())
    assert torch.allclose(out, hidden_in)  # no longer ablating after exit


# --- ConditionalSteeringHook: exact math -----------------------------------------

def test_conditional_steering_fires_above_threshold():
    layer = _EchoLayer()
    vector = torch.tensor([0.0, 0.0, 1.0, 0.0])  # push along dim 2
    condition = torch.tensor([1.0, 0.0, 0.0, 0.0])  # gate on dim 0
    # projection onto condition = 5.0, well above threshold 1.0 -> fires
    hidden_in = torch.tensor([[[5.0, 0.0, 0.0, 0.0]]])
    with apply_conditional_steering(layer, vector, coefficient=2.0, condition_vector=condition, threshold=1.0, token_scope="all"):
        (out,) = layer(hidden_in)
    assert torch.allclose(out, torch.tensor([[[5.0, 0.0, 2.0, 0.0]]]), atol=1e-5)


def test_conditional_steering_does_not_fire_below_threshold():
    layer = _EchoLayer()
    vector = torch.tensor([0.0, 0.0, 1.0, 0.0])
    condition = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # projection onto condition = 0.5, below threshold 1.0 -> no-op
    hidden_in = torch.tensor([[[0.5, 0.0, 0.0, 0.0]]])
    with apply_conditional_steering(layer, vector, coefficient=2.0, condition_vector=condition, threshold=1.0, token_scope="all"):
        (out,) = layer(hidden_in)
    assert torch.allclose(out, hidden_in)


def test_conditional_steering_gates_per_position_independently():
    layer = _EchoLayer()
    vector = torch.tensor([0.0, 0.0, 1.0, 0.0])
    condition = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # position 0 projects to 5.0 (fires), position 1 projects to 0.0 (doesn't).
    hidden_in = torch.tensor([[[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    with apply_conditional_steering(layer, vector, coefficient=3.0, condition_vector=condition, threshold=1.0, token_scope="all"):
        (out,) = layer(hidden_in)
    assert torch.allclose(out[:, 0, :], torch.tensor([[5.0, 0.0, 3.0, 0.0]]), atol=1e-5)
    assert torch.allclose(out[:, 1, :], torch.tensor([[0.0, 0.0, 0.0, 0.0]]), atol=1e-5)


def test_conditional_steering_none_condition_or_zero_coefficient_is_noop():
    layer = _EchoLayer()
    vector = torch.tensor([0.0, 0.0, 1.0, 0.0])
    hidden_in = torch.tensor([[[5.0, 0.0, 0.0, 0.0]]])
    with apply_conditional_steering(layer, vector, coefficient=2.0, condition_vector=None, threshold=0.0, token_scope="all"):
        (out,) = layer(hidden_in.clone())
    assert torch.allclose(out, hidden_in)

    with apply_conditional_steering(layer, vector, coefficient=0.0, condition_vector=torch.tensor([1.0, 0, 0, 0]), threshold=0.0, token_scope="all"):
        (out,) = layer(hidden_in.clone())
    assert torch.allclose(out, hidden_in)


def test_conditional_steering_hook_removes_handle_on_exit():
    layer = _EchoLayer()
    vector = torch.tensor([0.0, 0.0, 1.0, 0.0])
    condition = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hook = ConditionalSteeringHook(layer, vector, coefficient=2.0, condition_vector=condition, threshold=0.0, token_scope="all")
    with hook:
        assert hook._handle is not None
    assert hook._handle is None
    hidden_in = torch.tensor([[[5.0, 0.0, 0.0, 0.0]]])
    (out,) = layer(hidden_in.clone())
    assert torch.allclose(out, hidden_in)  # no longer gating after exit


# --- Batching correctness (real tiny Llama fixture, left-padded) ---------------

def test_ablation_hook_batch_safe_under_left_padding():
    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    layer = loaded.layers[0]
    tokenizer = loaded.tokenizer
    texts = ["w4 w5 w6", "w4 w5 w6 w7 w8"]

    tokenizer.padding_side = "left"
    enc_batch = tokenizer(texts, return_tensors="pt", padding=True)
    hook = AblationHook(layer, vector, strength=1.0, token_scope="all")
    with hook:
        with torch.no_grad():
            out_batch = loaded.model(**enc_batch)

    enc_single = tokenizer([texts[0]], return_tensors="pt")
    hook_single = AblationHook(layer, vector, strength=1.0, token_scope="all")
    with hook_single:
        with torch.no_grad():
            out_single = loaded.model(**enc_single)

    assert torch.allclose(out_batch.logits[0, -1], out_single.logits[0, -1], atol=1e-4)


def test_conditional_steering_hook_batch_safe_under_left_padding():
    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    condition = torch.randn(loaded.hidden_size)
    layer = loaded.layers[0]
    tokenizer = loaded.tokenizer
    texts = ["w4 w5 w6", "w4 w5 w6 w7 w8"]

    tokenizer.padding_side = "left"
    enc_batch = tokenizer(texts, return_tensors="pt", padding=True)
    hook = ConditionalSteeringHook(layer, vector, coefficient=2.0, condition_vector=condition, threshold=0.0, token_scope="all")
    with hook:
        with torch.no_grad():
            out_batch = loaded.model(**enc_batch)

    enc_single = tokenizer([texts[0]], return_tensors="pt")
    hook_single = ConditionalSteeringHook(layer, vector, coefficient=2.0, condition_vector=condition, threshold=0.0, token_scope="all")
    with hook_single:
        with torch.no_grad():
            out_single = loaded.model(**enc_single)

    assert torch.allclose(out_batch.logits[0, -1], out_single.logits[0, -1], atol=1e-4)


# --- Relationship to plain additive SteeringHook (sanity anchors) --------------

def test_conditional_steering_always_fires_matches_plain_steering_hook():
    from steering_factory.hooks import SteeringHook

    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    condition = torch.randn(loaded.hidden_size)
    layer = loaded.layers[0]
    enc = loaded.tokenizer("w4 w5 w6 w7", return_tensors="pt")

    with torch.no_grad():
        baseline_logits = loaded.model(**enc).logits

    with ConditionalSteeringHook(layer, vector, coefficient=5.0, condition_vector=condition, threshold=-1e9, token_scope="all"):
        with torch.no_grad():
            always_fires_logits = loaded.model(**enc).logits
    with SteeringHook(layer, vector, coefficient=5.0, token_scope="all"):
        with torch.no_grad():
            plain_steering_logits = loaded.model(**enc).logits

    assert not torch.allclose(baseline_logits, always_fires_logits)
    assert torch.allclose(always_fires_logits, plain_steering_logits, atol=1e-4)


def test_conditional_steering_never_fires_matches_baseline():
    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    condition = torch.randn(loaded.hidden_size)
    layer = loaded.layers[0]
    enc = loaded.tokenizer("w4 w5 w6 w7", return_tensors="pt")

    with torch.no_grad():
        baseline_logits = loaded.model(**enc).logits
    with ConditionalSteeringHook(layer, vector, coefficient=5.0, condition_vector=condition, threshold=1e9, token_scope="all"):
        with torch.no_grad():
            never_fires_logits = loaded.model(**enc).logits

    assert torch.allclose(baseline_logits, never_fires_logits)


# --- interventions.py multi-layer wrappers --------------------------------------

def test_apply_ablation_intervention_applies_across_multiple_layers():
    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    enc = loaded.tokenizer("w4 w5 w6 w7", return_tensors="pt")

    with torch.no_grad():
        baseline_logits = loaded.model(**enc).logits
    with apply_ablation_intervention(loaded, vector, layers=[0, 1], strength=1.0, token_scope="all"):
        with torch.no_grad():
            ablated_logits = loaded.model(**enc).logits
    assert not torch.allclose(baseline_logits, ablated_logits)


def test_apply_ablation_intervention_rejects_out_of_range_layer():
    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    with pytest.raises(IndexError):
        with apply_ablation_intervention(loaded, vector, layers=[loaded.num_layers], strength=1.0):
            pass


def test_apply_conditional_intervention_applies_across_multiple_layers():
    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    condition = torch.randn(loaded.hidden_size)
    enc = loaded.tokenizer("w4 w5 w6 w7", return_tensors="pt")

    with torch.no_grad():
        baseline_logits = loaded.model(**enc).logits
    with apply_conditional_intervention(loaded, vector, layers=[0, 1], coefficient=3.0,
                                         condition_vector=condition, threshold=-1e9, token_scope="all"):
        with torch.no_grad():
            gated_logits = loaded.model(**enc).logits
    assert not torch.allclose(baseline_logits, gated_logits)


def test_apply_conditional_intervention_rejects_out_of_range_layer():
    loaded = build_loaded_model(seed=6)
    vector = torch.randn(loaded.hidden_size)
    condition = torch.randn(loaded.hidden_size)
    with pytest.raises(IndexError):
        with apply_conditional_intervention(loaded, vector, layers=[-1], coefficient=1.0,
                                             condition_vector=condition, threshold=0.0):
            pass
