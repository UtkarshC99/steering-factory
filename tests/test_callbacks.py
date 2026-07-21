from pathlib import Path

import pytest

from steering_factory.artifacts import ArtifactStore
from steering_factory.callbacks import CompositeCallback, JsonlCallback, PrintSummaryCallback, _safe


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


def _generation_event(i, n=10, ppl=1.0, js=0.1):
    return {"event": "generation", "i": i, "n": n, "model_id": "m", "recipe_id": "r", "method": "mean_diff",
            "layer_idx": 5, "coefficient": 1.0, "token_scope": "all",
            "perplexity_ratio_vs_baseline": ppl, "js_divergence_vs_baseline": js}


def test_print_summary_callback_fires_only_every_k_events(capsys):
    callback = PrintSummaryCallback(every=3)
    for i in range(1, 3):
        callback.on_event(_generation_event(i))
    assert capsys.readouterr().out == ""  # no print yet, below threshold
    callback.on_event(_generation_event(3))
    out = capsys.readouterr().out
    assert "steering" in out
    assert "mean perplexity_ratio_vs_baseline=" in out
    assert "mean js_divergence_vs_baseline=" in out


def test_print_summary_callback_reports_only_the_most_recent_block(capsys):
    # Block 1: all ppl=1.0. Block 2: all ppl=3.0 -- the second summary must
    # reflect only block 2, not a running mean across both.
    callback = PrintSummaryCallback(every=2)
    callback.on_event(_generation_event(1, ppl=1.0))
    callback.on_event(_generation_event(2, ppl=1.0))
    capsys.readouterr()  # discard first block's output
    callback.on_event(_generation_event(3, ppl=3.0))
    callback.on_event(_generation_event(4, ppl=3.0))
    out = capsys.readouterr().out
    assert "mean perplexity_ratio_vs_baseline=3.000" in out


def test_print_summary_callback_ignores_unrelated_events():
    callback = PrintSummaryCallback(every=1)
    callback.on_event({"event": "vector", "i": 1, "n": 1})  # no handler -- must not raise
    callback.on_event({"event": "run_done"})


def test_print_summary_callback_handles_train_log(capsys):
    callback = PrintSummaryCallback(every=2)
    callback.on_event({"event": "train_log", "step": 1, "max_steps": 10, "loss": 2.0, "grad_norm": 0.5,
                        "model_id": "m", "recipe_id": "r"})
    callback.on_event({"event": "train_log", "step": 2, "max_steps": 10, "loss": 1.5, "grad_norm": 0.4,
                        "model_id": "m", "recipe_id": "r"})
    out = capsys.readouterr().out
    assert "qlora" in out
    assert "mean loss=1.750" in out


def test_print_summary_callback_handles_opt_step(capsys):
    callback = PrintSummaryCallback(every=2)
    callback.on_event({"event": "opt_step", "i": 1, "n": 5, "loss": 1.0, "grad_norm": 0.1})
    callback.on_event({"event": "opt_step", "i": 2, "n": 5, "loss": 2.0, "grad_norm": 0.2})
    out = capsys.readouterr().out
    assert "bipo-lite" in out
    assert "mean loss=1.500" in out


def test_print_summary_callback_handles_missing_metric_values(capsys):
    # perplexity/JS can be None (e.g. baseline computation failed) -- the
    # summary must render "n/a" rather than crashing on statistics.fmean([]).
    callback = PrintSummaryCallback(every=1)
    callback.on_event(_generation_event(1, ppl=None, js=None))
    out = capsys.readouterr().out
    assert "mean perplexity_ratio_vs_baseline=n/a" in out
    assert "mean js_divergence_vs_baseline=n/a" in out


def test_print_summary_callback_is_isolated_by_composite_when_it_raises(monkeypatch):
    # Even if PrintSummaryCallback somehow raised, CompositeCallback must
    # isolate it exactly like any other callback.
    def _boom(self, event):
        raise RuntimeError("boom")

    monkeypatch.setattr(PrintSummaryCallback, "on_event", _boom)
    composite = CompositeCallback([PrintSummaryCallback(every=1)])
    composite.on_event(_generation_event(1))  # must not raise
