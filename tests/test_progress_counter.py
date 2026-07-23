"""Regression test for run_steering's "generation" progress events: `n`
(the denominator) must stay a fixed, correct total for the whole model's
generation work, and `i` (the numerator) must never exceed it.

Before this existed, `n` was recomputed inside the per-recipe loop from
that recipe's OWN grid size alone, while `i` kept counting across every
recipe in the run without resetting -- so once a run moved from a
larger-grid recipe to a smaller-grid one, `i` could exceed the new (smaller,
stale) `n`, producing exactly the "2030/1080" nonsense a real run showed.
A manifest with two recipes of deliberately different grid sizes
reproduces this: the bug would show `i > n` once the second (smaller)
recipe's events start.
"""
import json

import pytest

from steering_factory import model_utils
from steering_factory.callbacks import RunCallback
from steering_factory.runner import _count_generation_units, run_steering

from _tiny_model import build_loaded_model


def _write_recipe_jsonl(path, n_per_category=20):
    records = []
    for cat in ("alpha", "beta"):
        for i in range(n_per_category):
            records.append({
                "id": f"{cat}-{i}", "prompt": f"w4 w5 w{6 + i % 10}",
                "positive": f"w10 w11 w{20 + i % 5}", "negative": f"w30 w31 w{40 + i % 5}",
                "category": cat,
            })
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _manifest_two_recipes(tmp_path, big_path, small_path):
    return {
        "experiment": {"name": "progress-counter-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [
            {
                # Larger grid: 2 layers x 2 coefficients (+implicit baseline = 3) x 1 scope.
                "id": "big_recipe", "behavior": "domain_classification",
                "dataset": {"adapter": "local_jsonl", "path": str(big_path)},
                "extraction": ["mean_diff"],
                "application": {"layers": [0, 1], "coefficients": [1.0, -1.0], "token_scopes": ["all"]},
            },
            {
                # Smaller grid: 1 layer x 1 coefficient (+implicit baseline = 2) x 1 scope.
                # This is the recipe that would expose the stale-denominator
                # bug -- its own n is much smaller than big_recipe's total,
                # but i (cumulative) is already past big_recipe's total by
                # the time these events fire.
                "id": "small_recipe", "behavior": "domain_classification",
                "dataset": {"adapter": "local_jsonl", "path": str(small_path)},
                "extraction": ["mean_diff"],
                "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
            },
        ],
        "decoding": {"max_new_tokens": 4},
    }


@pytest.fixture(autouse=True)
def _patch_load_model(monkeypatch):
    monkeypatch.setattr(model_utils, "load_model", lambda cfg: build_loaded_model(seed=5))


class _EventRecorder:
    def __init__(self):
        self.generation_events = []

    def on_event(self, event):
        if event.get("event") == "generation":
            self.generation_events.append(event)


def test_generation_progress_i_never_exceeds_n(tmp_path):
    big_path, small_path = tmp_path / "big.jsonl", tmp_path / "small.jsonl"
    _write_recipe_jsonl(big_path)
    _write_recipe_jsonl(small_path)
    manifest = _manifest_two_recipes(tmp_path, big_path, small_path)

    recorder = _EventRecorder()
    run_steering(manifest, command="test", callback=recorder)

    assert recorder.generation_events, "expected at least one generation event"
    for event in recorder.generation_events:
        assert event["i"] <= event["n"], (
            f"progress counter went past its own total: i={event['i']} n={event['n']} "
            f"(recipe={event.get('recipe_id')})"
        )


def test_generation_progress_n_is_constant_across_the_whole_model(tmp_path):
    """`n` must be the SAME fixed value for every event in a model's run --
    not recomputed smaller when a new (smaller-grid) recipe starts."""
    big_path, small_path = tmp_path / "big.jsonl", tmp_path / "small.jsonl"
    _write_recipe_jsonl(big_path)
    _write_recipe_jsonl(small_path)
    manifest = _manifest_two_recipes(tmp_path, big_path, small_path)

    recorder = _EventRecorder()
    run_steering(manifest, command="test", callback=recorder)

    ns = {event["n"] for event in recorder.generation_events}
    assert len(ns) == 1, f"n changed mid-run across recipes: {ns}"


def test_generation_progress_i_reaches_n_at_the_final_event(tmp_path):
    big_path, small_path = tmp_path / "big.jsonl", tmp_path / "small.jsonl"
    _write_recipe_jsonl(big_path)
    _write_recipe_jsonl(small_path)
    manifest = _manifest_two_recipes(tmp_path, big_path, small_path)

    recorder = _EventRecorder()
    run_steering(manifest, command="test", callback=recorder)

    last = recorder.generation_events[-1]
    assert last["i"] == last["n"]


def test_count_generation_units_matches_the_actual_event_count(tmp_path):
    """The precomputed total (_count_generation_units) must equal the
    number of generation events actually emitted -- otherwise the final
    event's i would never reach n, or i would overshoot n."""
    big_path, small_path = tmp_path / "big.jsonl", tmp_path / "small.jsonl"
    _write_recipe_jsonl(big_path)
    _write_recipe_jsonl(small_path)
    manifest = _manifest_two_recipes(tmp_path, big_path, small_path)

    recorder = _EventRecorder()
    run_steering(manifest, command="test", callback=recorder)

    loaded = build_loaded_model(seed=5)
    from steering_factory.runner import _load_normalized_records
    from steering_factory.artifacts import ArtifactStore
    store = ArtifactStore(manifest["artifacts"]["root"], manifest, "test")
    examples = _load_normalized_records(manifest, store)
    expected_total = _count_generation_units(manifest, examples, {"tiny": loaded.num_layers})

    assert len(recorder.generation_events) == expected_total
    assert recorder.generation_events[-1]["n"] == expected_total
