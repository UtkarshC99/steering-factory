import json
from pathlib import Path

import pytest
import torch

from steering_factory.artifacts import ArtifactStore, fingerprint_records
from steering_factory.datasets import leakage_report, stable_split
from steering_factory.evaluators import abstention_score, expected_calibration_error, structured_score
from steering_factory.experiment_types import SplitPlan
from steering_factory.interventions import compose, orthogonalize
from steering_factory.manifest import apply_overrides, load_manifest, validate_manifest


def manifest(tmp_path):
    return {
        "experiment": {"name": "test"}, "artifacts": {"root": str(tmp_path)},
        "models": [{"id": "toy", "name_or_path": "toy"}],
        "recipes": [{"id": "r", "behavior": "domain_classification", "dataset": {"adapter": "local_jsonl", "path": "x"}}],
    }


def test_dotted_overrides_preserve_types(tmp_path):
    updated = apply_overrides(manifest(tmp_path), ["artifacts.root=/tmp/results", "experiment.seed=9", "new.enabled=true"])
    assert updated["experiment"]["seed"] == 9
    assert updated["new"]["enabled"] is True


def test_manifest_validation_requires_core_keys(tmp_path):
    with pytest.raises(ValueError):
        validate_manifest({"experiment": {}})
    validate_manifest(manifest(tmp_path))


def test_split_is_deterministic_and_grouped():
    records = [{"id": f"a{i}", "category": "a", "prompt": str(i)} for i in range(5)] + [{"id": f"b{i}", "category": "b", "prompt": str(i)} for i in range(5)]
    plan = SplitPlan(seed=42)
    assert stable_split(records, plan) == stable_split(records, plan)
    assert set(stable_split(records, plan).values()) == {"steer", "validation", "test"}


def test_leakage_detects_cross_split_duplicate():
    records = [{"id": "one", "prompt": "same prompt"}, {"id": "two", "prompt": " same  prompt "}]
    report = leakage_report(records, {"one": "steer", "two": "test"})
    assert report["has_leakage"]


def test_artifact_store_is_immutable_and_tracks_failure(tmp_path):
    store = ArtifactStore(tmp_path, manifest(tmp_path), "test")
    store.write_jsonl("results/rows.jsonl", [{"metric": 1.0}])
    store.save_vector("v", torch.ones(3), {"hidden_size": 3})
    store.finalize()
    run = json.loads((store.path / "run.json").read_text())
    assert run["status"] == "completed"
    assert (store.path / "vectors" / "v.pt").exists()


def test_structured_and_abstention_metrics():
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}
    assert structured_score('{"x": 1}', schema, {"x": 1})["parse_success"] == 1.0
    assert structured_score("not json", schema)["parse_success"] == 0.0
    assert abstention_score("I don't know", answerable=False)["appropriate_abstention"] == 1.0
    assert abstention_score("I don't know", answerable=True)["false_abstention"] == 1.0
    assert expected_calibration_error([0.8, 0.2], [1, 0]) >= 0


def test_vector_composition_and_orthogonalization():
    vectors = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])]
    basis = orthogonalize(vectors)
    assert len(basis) == 2
    assert torch.dot(basis[0], basis[1]).abs() < 1e-5
    assert compose(vectors, "sum").shape == torch.Size([2])
