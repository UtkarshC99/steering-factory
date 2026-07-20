import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from steering_factory.hooks import SteeringHook, apply_steering


class _EchoLayer(nn.Module):
    """Stand-in for a transformer decoder block: returns its input hidden
    states unchanged, wrapped like most HF blocks (a 1-tuple)."""

    def forward(self, hidden_states):
        return (hidden_states,)


def _run_prefill_then_decode(layer, vector, coefficient, token_scope, hidden=4, prompt_len=3, decode_steps=5):
    """Simulates the call pattern of HF's cache-based `generate()`: one
    forward call with the full prompt (prefill, seq_len == prompt_len),
    followed by `decode_steps` forward calls each with a single new
    token's hidden state (seq_len == 1)."""
    outputs = []
    with apply_steering(layer, vector, coefficient, token_scope):
        prefill_in = torch.zeros(1, prompt_len, hidden)
        (prefill_out,) = layer(prefill_in)
        outputs.append(prefill_out.clone())

        for _ in range(decode_steps):
            step_in = torch.zeros(1, 1, hidden)
            (step_out,) = layer(step_in)
            outputs.append(step_out.clone())
    return outputs


def test_all_scope_steers_every_position_including_prefill():
    layer = _EchoLayer()
    vector = torch.ones(4)
    outputs = _run_prefill_then_decode(layer, vector, coefficient=2.0, token_scope="all")

    prefill_out = outputs[0]
    assert torch.allclose(prefill_out, torch.full_like(prefill_out, 2.0))
    for decode_out in outputs[1:]:
        assert torch.allclose(decode_out, torch.full_like(decode_out, 2.0))


def test_generated_only_scope_skips_prefill_steers_decode_steps():
    layer = _EchoLayer()
    vector = torch.ones(4)
    outputs = _run_prefill_then_decode(layer, vector, coefficient=3.0, token_scope="generated_only", prompt_len=5, decode_steps=4)

    prefill_out = outputs[0]
    # Prefill (seq_len > 1) must be left untouched: still all zeros.
    assert torch.allclose(prefill_out, torch.zeros_like(prefill_out))

    for decode_out in outputs[1:]:
        assert torch.allclose(decode_out, torch.full_like(decode_out, 3.0))


def test_generated_only_scope_with_no_multitoken_call_still_steers():
    """If a caller never does a multi-token prefill (every forward call is
    single-token from the start), there is no prefill to skip -- every call
    should be treated as a decode step and steered."""
    layer = _EchoLayer()
    vector = torch.ones(4)
    with apply_steering(layer, vector, 1.5, "generated_only"):
        (out,) = layer(torch.zeros(1, 1, 4))
    assert torch.allclose(out, torch.full_like(out, 1.5))


def test_last_scope_only_steers_final_position():
    layer = _EchoLayer()
    vector = torch.ones(4)
    with apply_steering(layer, vector, 5.0, "last"):
        (out,) = layer(torch.zeros(1, 3, 4))
    assert torch.allclose(out[:, :-1, :], torch.zeros(1, 2, 4))
    assert torch.allclose(out[:, -1, :], torch.full((1, 4), 5.0))


def test_zero_coefficient_or_none_vector_is_noop():
    layer = _EchoLayer()
    with apply_steering(layer, None, 1.0, "generated_only"):
        (out,) = layer(torch.zeros(1, 2, 4))
    assert torch.allclose(out, torch.zeros(1, 2, 4))

    with apply_steering(layer, torch.ones(4), 0.0, "generated_only"):
        (out,) = layer(torch.zeros(1, 2, 4))
    assert torch.allclose(out, torch.zeros(1, 2, 4))


def test_steering_hook_removes_handle_on_exit():
    layer = _EchoLayer()
    hook = SteeringHook(layer, torch.ones(4), 1.0, "all")
    with hook:
        assert hook._handle is not None
        (out,) = layer(torch.zeros(1, 1, 4))
        assert torch.allclose(out, torch.ones(1, 1, 4))
    assert hook._handle is None
    # After exit, the hook must no longer fire.
    (out,) = layer(torch.zeros(1, 1, 4))
    assert torch.allclose(out, torch.zeros(1, 1, 4))
