from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch


def _unwrap_output(output):
    """Decoder blocks typically return a tuple (hidden_states, ...) or just
    a tensor. Normalize to (hidden_states, rest, was_tuple)."""
    if isinstance(output, tuple):
        return output[0], output[1:], True
    return output, None, False


def _rewrap_output(hidden_states, rest, was_tuple):
    if was_tuple:
        return (hidden_states,) + rest
    return hidden_states


class ActivationCapture:
    """Captures the residual-stream activation at a given layer's output.

    Use as a context manager. After exiting, `.activations` holds one
    tensor per forward call made while active, shape (batch, seq, hidden).
    """

    def __init__(self, layer_module: torch.nn.Module):
        self.layer_module = layer_module
        self.activations: List[torch.Tensor] = []
        self._handle = None

    def _hook(self, module, inputs, output):
        hidden_states, _, _ = _unwrap_output(output)
        self.activations.append(hidden_states.detach())
        return output

    def __enter__(self):
        self._handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def last_token_activation(self, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pools captured activations to one vector per forward call, taking
        the last non-padded token. Returns shape (n_calls, hidden).

        Finds the rightmost position where attention_mask == 1 per row,
        which is correct under EITHER padding side. `mask.sum(dim=1) - 1`
        (real-token count minus one) is only the right answer for
        right-padding, where real tokens start at index 0 -- under
        left-padding the last real token sits at a fixed absolute position
        near the end of the sequence regardless of how much padding
        precedes it, and real-token-count does not tell you where that is.
        This was a real, previously-latent bug: every caller used to pass
        batch size 1 with an all-ones mask, where the two formulas
        coincide, so it was never exercised until left-padded batching was
        introduced (see tests/test_extraction_batching.py)."""
        pooled = []
        for act in self.activations:
            if attention_mask is not None:
                seq_len = act.size(1)
                positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
                masked_positions = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1))
                idx = masked_positions.max(dim=1).values.clamp(min=0)
                pooled.append(act[torch.arange(act.size(0)), idx])
            else:
                pooled.append(act[:, -1, :])
        return torch.cat(pooled, dim=0) if pooled else torch.empty(0)

    def mean_activation(self, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = []
        for act in self.activations:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(act.dtype)
                summed = (act * mask).sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1)
                pooled.append(summed / count)
            else:
                pooled.append(act.mean(dim=1))
        return torch.cat(pooled, dim=0) if pooled else torch.empty(0)


TokenScope = str  # "all" | "generated_only" | "last"


class SteeringHook:
    """Adds `coefficient * vector` to the residual stream at a layer's
    output on every forward call while active.

    `token_scope` controls which sequence positions get the intervention:
      - "all" (default): every position of every forward call, including
        the prompt-prefill call under HF's cache-based `generate()`. This
        is also what extraction methods want when they run their own
        forward passes over full prompt+completion text.
      - "generated_only": skips the prefill call entirely (identified by
        the KV-cache heuristic: a forward call with more than one token in
        its input during autoregressive generation is prefill; each
        decode step calls the layer with exactly one new-token position).
        Only the single-token decode-step positions get steered, so the
        prompt itself is left unperturbed and only newly generated tokens
        are pushed toward the behavior.
      - "last": only the final position of each forward call gets steered
        (useful for one-shot last-token nudges rather than continuous
        steering across a generation).
    """

    def __init__(
        self,
        layer_module: torch.nn.Module,
        vector: torch.Tensor,
        coefficient: float,
        token_scope: TokenScope = "all",
    ):
        self.layer_module = layer_module
        self.vector = vector
        self.coefficient = coefficient
        self.token_scope = token_scope
        self._handle = None
        self._seen_multi_token_call = False

    def _hook(self, module, inputs, output):
        hidden_states, rest, was_tuple = _unwrap_output(output)
        vec = self.vector.to(hidden_states.dtype).to(hidden_states.device)
        seq_len = hidden_states.shape[1]

        if self.token_scope == "generated_only":
            # The first forward call carrying more than one position is the
            # prompt prefill; every later single-token call is a decode
            # step. If the whole generation is single-shot forward (seq_len
            # stays 1 throughout, e.g. some custom loops), every call is
            # treated as a decode step since there is no prefill to skip.
            if seq_len > 1:
                self._seen_multi_token_call = True
                return _rewrap_output(hidden_states, rest, was_tuple)
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.coefficient * vec
            return _rewrap_output(hidden_states, rest, was_tuple)

        if self.token_scope == "last":
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.coefficient * vec
            return _rewrap_output(hidden_states, rest, was_tuple)

        hidden_states = hidden_states + self.coefficient * vec
        return _rewrap_output(hidden_states, rest, was_tuple)

    def __enter__(self):
        self._handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def apply_steering(
    layer_module: torch.nn.Module,
    vector: Optional[torch.Tensor],
    coefficient: float,
    token_scope: TokenScope = "all",
):
    """No-op context manager if vector is None or coefficient == 0, so
    callers can always go through this path uniformly for the baseline."""
    if vector is None or coefficient == 0:
        yield
        return
    hook = SteeringHook(layer_module, vector, coefficient, token_scope)
    with hook:
        yield
