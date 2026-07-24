"""Regression tests for per-recipe decoding.max_new_tokens / decoding.max_length
overrides -- the same override pattern already proven for
decoding.batch_size (runner.py's recipe.get("decoding", {}).get(...)),
extended to the two knobs a real run needed for structured_output_real:
outputs and inputs were both being truncated (tokens_generated==96 on
nearly every row; long JSON schemas cut off mid-object at the 1024-token
input cap) at the manifest-wide defaults.
"""
import json

import pytest
import torch

from steering_factory import model_utils
from steering_factory.runner import run_steering

from _tiny_model import build_loaded_model


def _write_recipe_jsonl(path, n_per_category=6, prompt_len=3):
    records = []
    for cat in ("alpha", "beta"):
        for i in range(n_per_category):
            words = " ".join(f"w{4 + (i + j) % 50}" for j in range(prompt_len))
            records.append({
                "id": f"{cat}-{i}", "prompt": words,
                "positive": f"w10 w11 w{20 + i % 5}", "negative": f"w30 w31 w{40 + i % 5}",
                "category": cat,
            })
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _manifest(tmp_path, data_path, recipe_decoding=None):
    recipe = {
        "id": "toy_recipe", "behavior": "domain_classification",
        "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
        "extraction": ["mean_diff"],
        "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
    }
    if recipe_decoding is not None:
        recipe["decoding"] = recipe_decoding
    return {
        "experiment": {"name": "recipe-decoding-overrides-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [recipe],
        "decoding": {"max_new_tokens": 4, "batch_size": 8},
    }


@pytest.fixture(autouse=True)
def _patch_load_model(monkeypatch):
    monkeypatch.setattr(model_utils, "load_model", lambda cfg: build_loaded_model(seed=5))


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_recipe_max_new_tokens_override_is_used(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path, recipe_decoding={"max_new_tokens": 10})

    store = run_steering(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert rows
    # Greedy decode from a randomly-initialized tiny model rarely hits EOS
    # early, so tokens_generated should reflect the override (10), not the
    # manifest-wide default (4).
    assert any(r["tokens_generated"] > 4 for r in rows)


def test_recipe_max_new_tokens_falls_back_to_manifest_wide_default(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path, recipe_decoding=None)

    store = run_steering(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert rows
    assert all(r["tokens_generated"] <= 4 for r in rows)


def test_recipe_max_length_override_reaches_generation(tmp_path, monkeypatch):
    """Confirms decoding.max_length flows all the way to
    sweep.generate_with_steering_batch's tokenizer call -- spies on the
    max_prompt_length argument sweep.generate_with_steering_batch actually
    receives."""
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path, recipe_decoding={"max_length": 2048})

    from steering_factory import sweep
    seen_max_lengths = []
    original = sweep.generate_with_steering_batch

    def spy(*args, **kwargs):
        # (loaded, layer_idx, vector, coefficient, prompts, max_new_tokens,
        # token_scope, max_prompt_length) -- max_prompt_length is the 8th
        # positional arg (index 7) or a kwarg.
        if "max_prompt_length" in kwargs:
            seen_max_lengths.append(kwargs["max_prompt_length"])
        elif len(args) >= 8:
            seen_max_lengths.append(args[7])
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "generate_with_steering_batch", spy)
    # runner.py does `from . import sweep` inside the function body, so the
    # patched module attribute is what gets resolved on each call.

    run_steering(manifest, command="test")

    assert seen_max_lengths, "generate_with_steering_batch was never called"
    assert all(v == 2048 for v in seen_max_lengths)


def test_recipe_max_length_defaults_to_1024_when_unset(tmp_path, monkeypatch):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path, recipe_decoding=None)

    from steering_factory import sweep
    seen_max_lengths = []
    original = sweep.generate_with_steering_batch

    def spy(*args, **kwargs):
        if "max_prompt_length" in kwargs:
            seen_max_lengths.append(kwargs["max_prompt_length"])
        elif len(args) >= 8:
            seen_max_lengths.append(args[7])
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "generate_with_steering_batch", spy)

    run_steering(manifest, command="test")

    assert seen_max_lengths
    assert all(v == 1024 for v in seen_max_lengths)
