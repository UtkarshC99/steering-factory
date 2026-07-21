from pathlib import Path

import pytest

from steering_factory.artifacts import ArtifactStore
from steering_factory.callbacks import CompositeCallback, JsonlCallback, _safe


def _manifest(tmp_path):
    return {
        "experiment": {"name": "test"}, "artifacts": {"root": str(tmp_path)},
        "models": [{"id": "toy", "name_or_path": "toy"}],
        "recipes": [{"id": "r", "behavior": "domain_classification", "dataset": {"adapter": "local_jsonl", "path": "x"}}],
    }


class _Recorder:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


class _Raiser:
    def on_event(self, event):
        raise RuntimeError("boom")


def test_safe_is_noop_for_none_callback():
    # Must not raise -- this is what keeps callback=None on every runner
    # entrypoint a true no-op for the CLI/subprocess path.
    _safe(None, {"event": "whatever"})


def test_safe_calls_callback():
    recorder = _Recorder()
    _safe(recorder, {"event": "x", "i": 1})
    assert recorder.events == [{"event": "x", "i": 1}]


def test_safe_isolates_raising_callback(caplog):
    # A callback bug (e.g. a live plot mid-redraw) must never propagate and
    # fail the underlying experiment.
    _safe(_Raiser(), {"event": "x"})  # should not raise


def test_jsonl_callback_appends_to_progress_file(tmp_path):
    store = ArtifactStore(tmp_path, _manifest(tmp_path), "test")
    callback = JsonlCallback(store, arm="steering")
    callback.on_event({"event": "generation", "i": 1, "n": 3})
    callback.on_event({"event": "generation", "i": 2, "n": 3})

    progress_path = store.path / "progress" / "steering.jsonl"
    assert progress_path.exists()
    lines = progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["i"] == 1
    assert json.loads(lines[1])["i"] == 2


def test_jsonl_callback_survives_partial_run(tmp_path):
    # A crashed run should still leave a readable partial trace -- append
    # mode, not write-once-at-the-end.
    store = ArtifactStore(tmp_path, _manifest(tmp_path), "test")
    callback = JsonlCallback(store, arm="qlora")
    callback.on_event({"event": "train_log", "step": 1, "loss": 2.0})
    try:
        raise RuntimeError("simulated crash mid-run")
    except RuntimeError as error:
        store.finalize(error)

    progress_path = store.path / "progress" / "qlora.jsonl"
    assert progress_path.exists()
    run = (store.path / "run.json").read_text(encoding="utf-8")
    assert '"failed"' in run


def test_composite_callback_fans_out_to_all_children():
    a, b = _Recorder(), _Recorder()
    composite = CompositeCallback([a, b])
    composite.on_event({"event": "x"})
    assert a.events == [{"event": "x"}]
    assert b.events == [{"event": "x"}]


def test_composite_callback_isolates_one_raising_child():
    good, bad = _Recorder(), _Raiser()
    composite = CompositeCallback([bad, good])
    composite.on_event({"event": "x"})  # must not raise
    assert good.events == [{"event": "x"}]


def test_composite_callback_is_itself_a_valid_callback(tmp_path):
    # CompositeCallback must satisfy the same on_event(dict) contract so it
    # can be passed anywhere a single RunCallback is accepted.
    store = ArtifactStore(tmp_path, _manifest(tmp_path), "test")
    jsonl = JsonlCallback(store, arm="steering")
    recorder = _Recorder()
    composite = CompositeCallback([jsonl, recorder])
    _safe(composite, {"event": "generation", "i": 1, "n": 1})
    assert recorder.events == [{"event": "generation", "i": 1, "n": 1}]
    assert (store.path / "progress" / "steering.jsonl").exists()
