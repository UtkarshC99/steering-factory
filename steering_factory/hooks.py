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


class AblationHook:
    """Projects `vector`'s direction OUT of the residual stream instead of
    adding it in -- the defensive-hardening counterpart to SteeringHook's
    additive intervention (directional ablation / "abliteration"): for
    unit vector v_hat, h -> h - strength * (h . v_hat) * v_hat.

    `strength` (0-1, default 1.0) scales how much of the projected
    component is removed: 1.0 is full ablation (the component is
    completely zeroed regardless of its sign/magnitude in that position);
    values below 1.0 partially damp it. Unlike SteeringHook's additive
    push (which can overshoot arbitrarily with a large enough coefficient),
    ablation is bounded by construction -- it can only remove what's
    already there, never add a new arbitrary-magnitude push -- which is
    exactly the property that makes it a defensive/hardening operation
    rather than another attack surface.

    `token_scope` has the same meaning as SteeringHook's ("all",
    "generated_only", "last")."""

    def __init__(
        self,
        layer_module: torch.nn.Module,
        vector: torch.Tensor,
        strength: float = 1.0,
        token_scope: TokenScope = "all",
    ):
        self.layer_module = layer_module
        self.vector = vector
        self.strength = strength
        self.token_scope = token_scope
        self._handle = None

    def _ablate(self, hidden_states: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(hidden_states, v_hat)  # (..., seq) or (..., 1) depending on shape
        return hidden_states - self.strength * projection.unsqueeze(-1) * v_hat

    def _hook(self, module, inputs, output):
        hidden_states, rest, was_tuple = _unwrap_output(output)
        v_hat = self.vector.to(hidden_states.dtype).to(hidden_states.device)
        v_hat = v_hat / v_hat.norm().clamp_min(1e-8)

        if self.token_scope in ("generated_only", "last"):
            seq_len = hidden_states.shape[1]
            if self.token_scope == "generated_only" and seq_len > 1:
                return _rewrap_output(hidden_states, rest, was_tuple)
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] = self._ablate(hidden_states[:, -1, :], v_hat)
            return _rewrap_output(hidden_states, rest, was_tuple)

        hidden_states = self._ablate(hidden_states, v_hat)
        return _rewrap_output(hidden_states, rest, was_tuple)

    def __enter__(self):
        self._handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class ConditionalSteeringHook:
    """CAST-style gated steering: adds `coefficient * vector` only at
    positions whose activation projects onto `condition_vector` above
    `threshold` -- everywhere else the intervention is a no-op for that
    position. This is the mechanism for cutting benign over-refusal: a
    plain additive SteeringHook pushes every input toward the behavior
    regardless of whether that input actually needs it; gating on a
    condition direction (e.g. "this input looks like an actual harmful
    request", learned/extracted the same way a steering vector is)
    confines the push to inputs that trigger the condition.

    `condition_vector` is normalized internally; the projection is a
    signed dot product against the RAW (non-unit) activation, following
    CAST's own formulation, since the threshold is calibrated in whatever
    scale the activations naturally live in. `vector` (the behavior
    direction) is used as-is, matching SteeringHook's own convention (its
    magnitude is meaningful -- it is what "coefficient" scales).
    """

    def __init__(
        self,
        layer_module: torch.nn.Module,
        vector: torch.Tensor,
        coefficient: float,
        condition_vector: torch.Tensor,
        threshold: float,
        token_scope: TokenScope = "all",
    ):
        self.layer_module = layer_module
        self.vector = vector
        self.coefficient = coefficient
        self.condition_vector = condition_vector
        self.threshold = threshold
        self.token_scope = token_scope
        self._handle = None

    def _gate_mask(self, hidden_states: torch.Tensor, cond_hat: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(hidden_states, cond_hat)  # (batch, seq)
        return (projection > self.threshold).unsqueeze(-1)  # (batch, seq, 1), broadcasts over hidden

    def _hook(self, module, inputs, output):
        hidden_states, rest, was_tuple = _unwrap_output(output)
        vec = self.vector.to(hidden_states.dtype).to(hidden_states.device)
        cond = self.condition_vector.to(hidden_states.dtype).to(hidden_states.device)
        cond_hat = cond / cond.norm().clamp_min(1e-8)

        if self.token_scope in ("generated_only", "last"):
            seq_len = hidden_states.shape[1]
            if self.token_scope == "generated_only" and seq_len > 1:
                return _rewrap_output(hidden_states, rest, was_tuple)
            hidden_states = hidden_states.clone()
            last = hidden_states[:, -1:, :]
            gate = self._gate_mask(last, cond_hat).to(last.dtype)
            hidden_states[:, -1:, :] = last + gate * self.coefficient * vec
            return _rewrap_output(hidden_states, rest, was_tuple)

        gate = self._gate_mask(hidden_states, cond_hat).to(hidden_states.dtype)
        hidden_states = hidden_states + gate * self.coefficient * vec
        return _rewrap_output(hidden_states, rest, was_tuple)

    def __enter__(self):
        self._handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def apply_ablation(
    layer_module: torch.nn.Module,
    vector: Optional[torch.Tensor],
    strength: float = 1.0,
    token_scope: TokenScope = "all",
):
    """No-op if vector is None or strength == 0, matching apply_steering's
    baseline-path convention."""
    if vector is None or strength == 0:
        yield
        return
    hook = AblationHook(layer_module, vector, strength, token_scope)
    with hook:
        yield


@contextmanager
def apply_conditional_steering(
    layer_module: torch.nn.Module,
    vector: Optional[torch.Tensor],
    coefficient: float,
    condition_vector: Optional[torch.Tensor],
    threshold: float,
    token_scope: TokenScope = "all",
):
    """No-op if vector or condition_vector is None or coefficient == 0."""
    if vector is None or condition_vector is None or coefficient == 0:
        yield
        return
    hook = ConditionalSteeringHook(layer_module, vector, coefficient, condition_vector, threshold, token_scope)
    with hook:
        yield
