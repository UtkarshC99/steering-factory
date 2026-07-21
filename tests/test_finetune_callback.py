"""Tests `finetune._build_trainer_callback` in isolation -- the adapter that
forwards transformers' own Trainer.on_log instrumentation into a
RunCallback. No real model/Trainer needed: on_log's signature is exercised
directly against a stand-in `state`/`control`.
"""
from types import SimpleNamespace

from steering_factory.finetune import _build_trainer_callback


class _Recorder:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_returns_none_when_no_callback():
    assert _build_trainer_callback(None, {}) is None


def test_forwards_on_log_as_train_log_event():
    recorder = _Recorder()
    forwarder = _build_trainer_callback(recorder, {"model_id": "qwen3_4b", "recipe_id": "safety_refusal"})
    assert forwarder is not None

    state = SimpleNamespace(global_step=5, max_steps=20)
    logs = {"loss": 1.234, "grad_norm": 0.5, "learning_rate": 2e-4, "epoch": 0.5}
    forwarder.on_log(args=None, state=state, control=None, logs=logs)

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event["arm"] == "qlora"
    assert event["event"] == "train_log"
    assert event["step"] == 5
    assert event["max_steps"] == 20
    assert event["loss"] == 1.234
    assert event["grad_norm"] == 0.5
    assert event["model_id"] == "qwen3_4b"
    assert event["recipe_id"] == "safety_refusal"


def test_handles_none_logs_gracefully():
    recorder = _Recorder()
    forwarder = _build_trainer_callback(recorder, {})
    state = SimpleNamespace(global_step=1, max_steps=-1)
    forwarder.on_log(args=None, state=state, control=None, logs=None)
    assert len(recorder.events) == 1
    assert recorder.events[0]["loss"] is None
    assert recorder.events[0]["max_steps"] is None  # -1 means "unbounded", normalized to None


def test_context_merges_into_every_event():
    recorder = _Recorder()
    forwarder = _build_trainer_callback(recorder, {"model_id": "m", "recipe_id": "r"})
    state = SimpleNamespace(global_step=1, max_steps=10)
    forwarder.on_log(args=None, state=state, control=None, logs={"loss": 1.0})
    forwarder.on_log(args=None, state=state, control=None, logs={"loss": 0.9})
    assert all(e["model_id"] == "m" and e["recipe_id"] == "r" for e in recorder.events)
