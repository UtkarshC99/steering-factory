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
        the last non-padded token. Returns shape (n_calls, hidden)."""
        pooled = []
        for i, act in enumerate(self.activations):
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1
                idx = lengths.clamp(min=0)
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


class SteeringHook:
    """Adds `coefficient * vector` to the residual stream at a layer's
    output on every forward call while active."""

    def __init__(self, layer_module: torch.nn.Module, vector: torch.Tensor, coefficient: float):
        self.layer_module = layer_module
        self.vector = vector
        self.coefficient = coefficient
        self._handle = None

    def _hook(self, module, inputs, output):
        hidden_states, rest, was_tuple = _unwrap_output(output)
        vec = self.vector.to(hidden_states.dtype).to(hidden_states.device)
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
def apply_steering(layer_module: torch.nn.Module, vector: Optional[torch.Tensor], coefficient: float):
    """No-op context manager if vector is None or coefficient == 0, so
    callers can always go through this path uniformly for the baseline."""
    if vector is None or coefficient == 0:
        yield
        return
    hook = SteeringHook(layer_module, vector, coefficient)
    with hook:
        yield
