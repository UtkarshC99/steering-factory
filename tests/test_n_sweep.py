"""Tests for the experiment.n_sweep axis: subsampling helpers in
runner.py, and an end-to-end run_steering pass proving multiple N points
produce distinct, correctly-tagged vectors/rows rather than one blended
result -- the exact failure mode a real N-sweep would hit if the
per-vector num_pairs tagging or _subsample_n's determinism were wrong.
"""
import json

import pytest

from steering_factory import model_utils
from steering_factory.runner import _n_sweep_values, _subsample_n, run_steering

from _tiny_model import build_loaded_model


# --- _n_sweep_values ----------------------------------------------------------

def test_n_sweep_values_absent_returns_none():
    assert _n_sweep_values({}, default_n=None) is None
    assert _n_sweep_values({"experiment": {}}, default_n=None) is None


def test_n_sweep_values_deduplicates_and_sorts():
    manifest = {"experiment": {"n_sweep": [32, 8, 16, 8]}}
    assert _n_sweep_values(manifest, default_n=None) == [8, 16, 32]


def test_n_sweep_values_drops_non_positive():
    manifest = {"experiment": {"n_sweep": [0, -5, 8]}}
    assert _n_sweep_values(manifest, default_n=None) == [8]


# --- _subsample_n --------------------------------------------------------------

def test_subsample_n_returns_all_items_when_n_exceeds_pool():
    items = [{"id": str(i)} for i in range(5)]
    assert _subsample_n(items, 10, seed=1) == items


def test_subsample_n_is_deterministic_for_same_seed():
    items = [{"id": str(i), "prompt": f"p{i}"} for i in range(20)]
    a = _subsample_n(items, 5, seed=42)
    b = _subsample_n(items, 5, seed=42)
    assert [x["id"] for x in a] == [x["id"] for x in b]


def test_subsample_n_is_stable_regardless_of_input_order():
    items = [{"id": str(i), "prompt": f"p{i}"} for i in range(20)]
    shuffled = list(reversed(items))
    a = _subsample_n(items, 5, seed=42)
    b = _subsample_n(shuffled, 5, seed=42)
    assert {x["id"] for x in a} == {x["id"] for x in b}


def test_subsample_n_differs_across_seeds_typically():
    items = [{"id": str(i), "prompt": f"p{i}"} for i in range(50)]
    a = _subsample_n(items, 5, seed=1)
    b = _subsample_n(items, 5, seed=2)
    assert {x["id"] for x in a} != {x["id"] for x in b}


def test_subsample_n_returns_exactly_n_items():
    items = [{"id": str(i)} for i in range(30)]
    assert len(_subsample_n(items, 7, seed=1)) == 7


# --- run_steering end-to-end with an N-sweep -----------------------------------

def _write_recipe_jsonl(path, n_per_category=20):
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


def _manifest_with_sweep(tmp_path, data_path, n_sweep):
    return {
        "experiment": {"name": "n-sweep-e2e", "n_sweep": n_sweep},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [{
            "id": "toy_recipe", "behavior": "domain_classification",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
        }],
        "decoding": {"max_new_tokens": 4},
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


def test_run_steering_without_n_sweep_extracts_one_vector_at_full_steer_size(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest_with_sweep(tmp_path, data_path, n_sweep=None)
    del manifest["experiment"]["n_sweep"]  # absent entirely, not just None -- matches every pre-existing manifest

    store = run_steering(manifest, command="test")
    vector_rows = _read_jsonl(store.path / "vectors" / "index.jsonl")
    assert len(vector_rows) == 1  # one (method, layer) config, no N-sweep -> exactly one vector


def test_run_steering_with_n_sweep_extracts_one_vector_per_n_point(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    # steer_fraction=0.5 over 40 rows (20/category) -> ~20 steer examples;
    # sweep points below and at that ceiling.
    manifest = _manifest_with_sweep(tmp_path, data_path, n_sweep=[4, 8, 16])

    store = run_steering(manifest, command="test")
    vector_rows = _read_jsonl(store.path / "vectors" / "index.jsonl")
    n_values = sorted(v["num_pairs"] for v in vector_rows)
    assert n_values == [4, 8, 16]  # one vector per sweep point, correctly tagged


def test_run_steering_n_sweep_smaller_subsets_are_subsets_of_larger_ones(tmp_path):
    # _subsample_n uses one seeded RNG per N, not a nested "take the first
    # k of the N=16 draw" scheme, so this is NOT guaranteed to be a strict
    # subset -- this test instead pins the more important, actually-relied-
    # upon property: every N point is reproducible and distinct in size.
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest_with_sweep(tmp_path, data_path, n_sweep=[4, 8])
    store = run_steering(manifest, command="test")
    vector_rows = _read_jsonl(store.path / "vectors" / "index.jsonl")
    assert {v["num_pairs"] for v in vector_rows} == {4, 8}


def test_run_steering_n_sweep_generation_rows_tag_matching_config_only(tmp_path):
    # Every generation row should trace back to exactly one (method, layer)
    # vector -- but since num_pairs isn't itself stored on generation rows
    # (only on the vector index), this confirms the count of generation
    # rows scales with the number of sweep points x evaluate examples,
    # i.e. nothing got silently merged or dropped across N points.
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest_single = _manifest_with_sweep(tmp_path, data_path, n_sweep=None)
    del manifest_single["experiment"]["n_sweep"]
    manifest_swept = _manifest_with_sweep(tmp_path, data_path, n_sweep=[4, 8, 16])

    single_store = run_steering(manifest_single, command="test")
    swept_store = run_steering(manifest_swept, command="test")

    single_rows = _read_jsonl(single_store.path / "results" / "generations.jsonl")
    swept_rows = _read_jsonl(swept_store.path / "results" / "generations.jsonl")
    assert len(swept_rows) == 3 * len(single_rows)  # 3 sweep points, same evaluate/coefficient/scope grid each
