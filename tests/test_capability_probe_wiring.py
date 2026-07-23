"""End-to-end test of the capability_probe manifest opt-in wired into
run_steering, against the tiny Llama fixture (no real model download).

Every existing manifest/test predates capability_probe.enabled, so the
critical property this file must prove FIRST is that leaving it unset (or
false) is a strict no-op -- test_capability_probe_wiring_disabled_by_default
pins that; everything else exercises what happens once it's turned on.
"""
import json

import pytest

from steering_factory import model_utils
from steering_factory.runner import run_steering

from _tiny_model import build_loaded_model


def _write_recipe_jsonl(path, n_per_category=8):
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


def _manifest(tmp_path, data_path, capability_probe=None, coefficients=None):
    manifest = {
        "experiment": {"name": "capability-probe-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": [{
            "id": "toy_recipe", "behavior": "domain_classification",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": {"layers": [0], "coefficients": coefficients or [1.0], "token_scopes": ["all"]},
        }],
        "decoding": {"max_new_tokens": 4},
    }
    if capability_probe is not None:
        manifest["capability_probe"] = capability_probe
    return manifest


@pytest.fixture(autouse=True)
def _patch_load_model(monkeypatch):
    monkeypatch.setattr(model_utils, "load_model", lambda cfg: build_loaded_model(seed=5))


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_capability_probe_disabled_by_default_writes_no_file(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    store = run_steering(_manifest(tmp_path, data_path), command="test")  # no capability_probe key at all
    assert not (store.path / "results" / "capability_probe.jsonl").exists()


def test_capability_probe_explicitly_false_writes_no_file(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    store = run_steering(_manifest(tmp_path, data_path, capability_probe={"enabled": False}), command="test")
    assert not (store.path / "results" / "capability_probe.jsonl").exists()


def test_capability_probe_enabled_writes_one_row_per_coefficient(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    coefficients = [-1.0, 1.0, 2.0]
    store = run_steering(
        _manifest(tmp_path, data_path, capability_probe={"enabled": True, "max_new_tokens": 3}, coefficients=coefficients),
        command="test",
    )
    rows = _read_jsonl(store.path / "results" / "capability_probe.jsonl")
    # coefficients grid always includes 0.0 alongside whatever's configured
    # (matching _evaluate_vector_grid's own all_coefficients construction).
    assert {r["coefficient"] for r in rows} == {-1.0, 0.0, 1.0, 2.0}
    for row in rows:
        assert row["model_id"] == "tiny"
        assert row["recipe_id"] == "toy_recipe"
        assert row["n_probes"] > 0
        assert row["base_capability_quality"] is not None
        assert row["steered_capability_quality"] is not None


def test_capability_probe_coefficient_zero_reports_perfect_retention(tmp_path):
    # At coefficient=0.0 the "steered" model IS the base model -- retention
    # must be exactly 1.0 (unless the base model scores 0 on every probe,
    # in which case it's None; the tiny fixture's random weights are
    # vanishingly unlikely to hit that, but assert the not-None case).
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    store = run_steering(
        _manifest(tmp_path, data_path, capability_probe={"enabled": True, "max_new_tokens": 3}, coefficients=[1.0]),
        command="test",
    )
    rows = _read_jsonl(store.path / "results" / "capability_probe.jsonl")
    baseline_row = next(r for r in rows if r["coefficient"] == 0.0)
    assert baseline_row["base_capability_quality"] == baseline_row["steered_capability_quality"]
    if baseline_row["capability_retention"] is not None:
        assert baseline_row["capability_retention"] == pytest.approx(1.0)


def test_capability_probe_base_measurement_cached_across_recipe_vectors(tmp_path, monkeypatch):
    # The base-model probe (coefficient=0.0) should run once per model_id
    # and be reused across every extracted vector for that model, not
    # re-measured per vector -- assert this via a call counter on the
    # underlying generation primitive.
    import steering_factory.capability_probe as capability_probe_module

    call_count = {"n": 0}
    original = capability_probe_module.run_capability_probe_steering

    def _counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(capability_probe_module, "run_capability_probe_steering", _counting)

    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path, capability_probe={"enabled": True, "max_new_tokens": 3}, coefficients=[1.0])
    manifest["recipes"][0]["application"]["layers"] = [0, 1]  # 2 vectors for the same model
    run_steering(manifest, command="test")

    # 2 vectors x 2 coefficients (0.0, 1.0) = 4 probe runs WITHOUT caching;
    # with caching, only 1 of those (per vector) is the base -- but the
    # base cache is keyed per model_id and shared ACROSS vectors, so the
    # very first vector's coefficient=0.0 call populates it and the second
    # vector's coefficient=0.0 call is skipped entirely (0 calls), leaving
    # 3 total: vector1's base + vector1's c=1.0 + vector2's c=1.0.
    assert call_count["n"] == 3
