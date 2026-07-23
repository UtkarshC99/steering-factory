"""Tests for `dataset.on_load_error` (runner._load_normalized_records):
default hard-fail on a broken dataset adapter, opt-in skip-and-continue,
and that a skip is recorded loudly (data/<recipe>.load_error.json) rather
than silently dropping a recipe out of the run.

Uses a real, un-mocked failure path (LocalJsonlAdapter.load raising
FileNotFoundError on a missing file) rather than a fake adapter, so this
tests the actual exception-propagation behavior, not a stand-in for it.
"""
import json

import pytest

from steering_factory.runner import prepare_data

from test_n_sweep import _write_recipe_jsonl  # reuses the existing toy-fixture writer


def _manifest(tmp_path, recipes):
    return {
        "experiment": {"name": "on-load-error-test"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": recipes,
    }


def _good_recipe(recipe_id, data_path):
    return {
        "id": recipe_id, "behavior": "domain_classification",
        "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
        "extraction": ["mean_diff"],
    }


def _broken_recipe(recipe_id, tmp_path, on_load_error=None):
    dataset_cfg = {"adapter": "local_jsonl", "path": str(tmp_path / "does_not_exist.jsonl")}
    if on_load_error is not None:
        dataset_cfg["on_load_error"] = on_load_error
    return {"id": recipe_id, "behavior": "domain_classification", "dataset": dataset_cfg, "extraction": ["mean_diff"]}


def test_default_on_load_error_raises_and_aborts_the_run(tmp_path):
    manifest = _manifest(tmp_path, [_broken_recipe("broken", tmp_path)])  # on_load_error unset -> default "raise"
    with pytest.raises(FileNotFoundError):
        prepare_data(manifest, command="test")


def test_explicit_raise_behaves_the_same_as_default(tmp_path):
    manifest = _manifest(tmp_path, [_broken_recipe("broken", tmp_path, on_load_error="raise")])
    with pytest.raises(FileNotFoundError):
        prepare_data(manifest, command="test")


def test_skip_continues_past_the_broken_recipe(tmp_path):
    data_path = tmp_path / "good.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, [
        _broken_recipe("broken", tmp_path, on_load_error="skip"),
        _good_recipe("good", data_path),
    ])
    store = prepare_data(manifest, command="test")  # must not raise
    assert store.artifact.status == "completed"

    examples = []
    with open(store.path / "data" / "examples.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    # Only the good recipe's records made it through.
    assert all(e["recipe_id"] == "good" for e in examples)
    assert len(examples) > 0


def test_skip_records_the_failure_loudly(tmp_path):
    data_path = tmp_path / "good.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, [
        _broken_recipe("broken", tmp_path, on_load_error="skip"),
        _good_recipe("good", data_path),
    ])
    store = prepare_data(manifest, command="test")

    error_path = store.path / "data" / "broken.load_error.json"
    assert error_path.exists()
    payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert payload["recipe_id"] == "broken"
    assert payload["skipped"] is True
    assert "FileNotFoundError" in payload["error"] or "No such file" in payload["error"]

    # The successful recipe must NOT also have a load_error.json -- only
    # the one that actually failed.
    assert not (store.path / "data" / "good.load_error.json").exists()
    # And the successful recipe's normal provenance file is still written.
    assert (store.path / "data" / "good.provenance.json").exists()


def test_skip_with_only_broken_recipes_yields_empty_run_not_a_crash(tmp_path):
    manifest = _manifest(tmp_path, [_broken_recipe("broken", tmp_path, on_load_error="skip")])
    store = prepare_data(manifest, command="test")  # must not raise
    assert store.artifact.status == "completed"
    summary = json.loads((store.path / "data" / "summary.json").read_text(encoding="utf-8"))
    assert summary["records"] == 0


def test_unrecognized_on_load_error_value_defaults_to_raise(tmp_path):
    # Only the literal string "skip" opts in -- anything else (typos,
    # booleans, etc.) must fail closed to the safer default, not silently
    # skip on a value that wasn't actually "skip".
    manifest = _manifest(tmp_path, [_broken_recipe("broken", tmp_path, on_load_error="Skip")])
    with pytest.raises(FileNotFoundError):
        prepare_data(manifest, command="test")
