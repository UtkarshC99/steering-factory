"""End-to-end test of run_steering / run_evaluate against the tiny Llama
fixture, with load_model patched so no real model is downloaded.

This is the integration point every other batching test (sweep, extraction,
finetune) feeds into: _evaluate_vector_grid restructured the per-example
generation loop to batch across examples at fixed (method, layer,
coefficient, token_scope), shared between run_evaluate and run_steering.
Unit tests for the pieces don't prove the wiring between them is correct --
only running the real orchestration function does.
"""
import json

import pytest
import torch

from steering_factory import model_utils
from steering_factory.datasets import stable_split
from steering_factory.experiment_types import SplitPlan
from steering_factory.runner import run_evaluate, run_extract, run_steering

from _tiny_model import build_loaded_model


def _write_recipe_jsonl(path, n_per_category=6):
    records = []
    for cat in ("alpha", "beta"):
        for i in range(n_per_category):
            records.append({
                "id": f"{cat}-{i}",
                "prompt": f"w4 w5 w{6 + i % 10}",
                "positive": f"w10 w11 w{20 + i % 5}",
                "negative": f"w30 w31 w{40 + i % 5}",
                "category": cat,
            })
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _manifest(tmp_path, data_path, batch_size=None):
    application = {
        "layers": [0],
        "coefficients": [-1.0, 1.0],
        "token_scopes": ["all", "generated_only"],
    }
    manifest = {
        "experiment": {"name": "batching-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": [{
            "id": "toy_recipe", "behavior": "domain_classification",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": application,
        }],
        "decoding": {"max_new_tokens": 4},
    }
    if batch_size is not None:
        manifest["decoding"]["batch_size"] = batch_size
    return manifest


@pytest.fixture(autouse=True)
def _patch_load_model(monkeypatch):
    # Every model in every test manifest here resolves to the same tiny
    # Llama fixture regardless of name_or_path -- load_model is imported
    # locally (`from .model_utils import load_model`) inside each runner
    # function, so patching the module-level name is picked up at call time.
    monkeypatch.setattr(model_utils, "load_model", lambda cfg: build_loaded_model(seed=5))


def test_run_steering_end_to_end_batch_size_1_vs_16_identical_rows(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)

    store_b1 = run_steering(_manifest(tmp_path, data_path, batch_size=1), command="test")
    store_b16 = run_steering(_manifest(tmp_path, data_path, batch_size=16), command="test")

    rows_b1 = _read_jsonl(store_b1.path / "results" / "generations.jsonl")
    rows_b16 = _read_jsonl(store_b16.path / "results" / "generations.jsonl")

    assert len(rows_b1) == len(rows_b16) > 0
    key = lambda r: (r["method"], r["layer_idx"], r["token_scope"], r["coefficient"], r["example_id"])
    rows_b1_sorted = sorted(rows_b1, key=key)
    rows_b16_sorted = sorted(rows_b16, key=key)
    for r1, r16 in zip(rows_b1_sorted, rows_b16_sorted):
        assert r1["output"] == r16["output"]
        assert r1["example_id"] == r16["example_id"]
        assert r1["coefficient"] == r16["coefficient"]
        assert r1["perplexity"] == pytest.approx(r16["perplexity"], rel=1e-3)


def test_run_steering_rows_carry_batch_size_and_wall_time(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    store = run_steering(_manifest(tmp_path, data_path, batch_size=8), command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert rows
    for row in rows:
        assert "batch_size" in row and "batch_wall_time_s" in row
        assert row["latency_s"] == pytest.approx(row["batch_wall_time_s"] / row["batch_size"])


def test_run_extract_then_run_evaluate_matches_run_steering(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)

    extract_store = run_extract(_manifest(tmp_path, data_path, batch_size=4), command="test")
    evaluate_store = run_evaluate(_manifest(tmp_path, data_path, batch_size=4), extract_store.path, command="test")
    steering_store = run_steering(_manifest(tmp_path, data_path, batch_size=4), command="test")

    rows_split = _read_jsonl(evaluate_store.path / "results" / "generations.jsonl")
    rows_combined = _read_jsonl(steering_store.path / "results" / "generations.jsonl")

    key = lambda r: (r["method"], r["layer_idx"], r["token_scope"], r["coefficient"], r["example_id"])
    rows_split_sorted = sorted(rows_split, key=key)
    rows_combined_sorted = sorted(rows_combined, key=key)
    assert len(rows_split_sorted) == len(rows_combined_sorted) > 0
    for a, b in zip(rows_split_sorted, rows_combined_sorted):
        assert a["output"] == b["output"]


def test_generation_callback_events_are_per_row_not_per_batch(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)

    events = []

    class _Recorder:
        def on_event(self, event):
            events.append(event)

    manifest = _manifest(tmp_path, data_path, batch_size=16)
    store = run_steering(manifest, command="test", callback=_Recorder())
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")

    generation_events = [e for e in events if e.get("event") == "generation"]
    # One callback event per scored row, regardless of batch_size=16 -- the
    # whole point of threading progress through _evaluate_vector_grid
    # per-row rather than per-batch.
    assert len(generation_events) == len(rows)


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
