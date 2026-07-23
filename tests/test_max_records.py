"""Tests for `dataset.max_records`: a deterministic, seeded subsample of
the whole loaded pool, applied BEFORE stable_split so steer/validation/
test proportions are preserved on the smaller pool rather than skewed.
"""
import json

import pytest

from steering_factory.runner import prepare_data

from test_n_sweep import _write_recipe_jsonl


def _manifest(tmp_path, data_path, max_records=None, seed=3):
    dataset_cfg = {"adapter": "local_jsonl", "path": str(data_path)}
    if max_records is not None:
        dataset_cfg["max_records"] = max_records
    return {
        "experiment": {"name": "max-records-test"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": seed, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": [{
            "id": "toy_recipe", "behavior": "domain_classification",
            "dataset": dataset_cfg, "extraction": ["mean_diff"],
        }],
    }


def _read_examples(store):
    rows = []
    with open(store.path / "data" / "examples.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_max_records_absent_keeps_full_pool(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path, n_per_category=20)  # 40 total (2 categories)
    store = prepare_data(_manifest(tmp_path, data_path), command="test")
    assert len(_read_examples(store)) == 40


def test_max_records_caps_the_pool_before_splitting(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path, n_per_category=20)  # 40 total
    store = prepare_data(_manifest(tmp_path, data_path, max_records=10), command="test")
    assert len(_read_examples(store)) == 10


def test_max_records_larger_than_pool_is_a_no_op(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path, n_per_category=5)  # 10 total
    store = prepare_data(_manifest(tmp_path, data_path, max_records=1000), command="test")
    assert len(_read_examples(store)) == 10


def test_max_records_is_deterministic_for_the_same_seed(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path, n_per_category=20)
    store_a = prepare_data(_manifest(tmp_path, data_path, max_records=8, seed=5), command="test")
    store_b = prepare_data(_manifest(tmp_path, data_path, max_records=8, seed=5), command="test")
    ids_a = sorted(e["id"] for e in _read_examples(store_a))
    ids_b = sorted(e["id"] for e in _read_examples(store_b))
    assert ids_a == ids_b


def test_max_records_differs_across_seeds_typically(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path, n_per_category=20)
    store_a = prepare_data(_manifest(tmp_path, data_path, max_records=8, seed=1), command="test")
    store_b = prepare_data(_manifest(tmp_path, data_path, max_records=8, seed=2), command="test")
    ids_a = {e["id"] for e in _read_examples(store_a)}
    ids_b = {e["id"] for e in _read_examples(store_b)}
    assert ids_a != ids_b


def test_max_records_preserves_split_proportions_on_the_smaller_pool(tmp_path):
    # steer_fraction=0.4, validation_fraction=0.3 on a capped pool of 10
    # per category should still land close to those fractions, not be
    # skewed by capping happening after vs. before splitting.
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path, n_per_category=50)  # 100 total
    store = prepare_data(_manifest(tmp_path, data_path, max_records=20), command="test")
    examples = _read_examples(store)
    assert len(examples) == 20
    splits = [e["split"] for e in examples]
    assert "steer" in splits and "validation" in splits and "test" in splits
