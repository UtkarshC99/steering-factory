"""Tests for `apply_vectors_from`: a recipe with no steer split of its own
(the XSTest shape -- every row forced to split="test" by its adapter's
`normalize`, which bypasses whatever `stable_split` assigned) evaluated
against a DIFFERENT recipe's already-extracted vectors, instead of
silently contributing zero rows.

Before this existed, every vector/eval-example filter in runner.py matched
strictly on recipe_id, so an evaluation-only control recipe had no way to
receive another recipe's vectors -- confirmed as a real, not hypothetical,
gap: manifests/benchmarks.yaml's benign_over_refusal_control (XSTest)
recipe's own comment described exactly this cross-recipe measurement, but
no code path implemented it.

Uses a tiny in-test adapter (_AlwaysTestAdapter) registered into
datasets.ADAPTERS for the duration of each test, rather than the real
XSTestAdapter -- LocalJsonlAdapter.normalize reads `record.get("split")`,
but by the time normalize() runs in _load_normalized_records, `split` has
already been overwritten by stable_split's assignment (item["split"] =
assignments[...]), so a plain local_jsonl fixture CANNOT reproduce
XSTest's actual hardcoded-split property -- only an adapter whose
normalize() ignores the incoming split value the way XSTestAdapter's does
can. Isolating exactly that property, without XSTest's real schema/HF
loading, is the point of the stand-in.
"""
import json

import pytest

from steering_factory import datasets as datasets_module
from steering_factory import model_utils
from steering_factory.experiment_types import ContrastiveExample
from steering_factory.runner import run_evaluate, run_extract, run_steering

from _tiny_model import build_loaded_model


class _AlwaysTestAdapter:
    """Minimal stand-in for XSTestAdapter's one relevant property: every
    row normalizes to split="test" regardless of what stable_split
    assigned upstream."""
    name = "always_test"

    def load(self, config):
        path = config["path"]
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def normalize(self, record, behavior):
        return ContrastiveExample(
            id=record["id"], prompt=record["prompt"], positive="", negative="",
            behavior_id=behavior.id, split="test", source=self.name, category=record.get("category"),
        )


@pytest.fixture(autouse=True)
def _register_always_test_adapter(monkeypatch):
    monkeypatch.setitem(datasets_module.ADAPTERS, "always_test", _AlwaysTestAdapter)


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


def _write_control_jsonl(path, n=10):
    records = [{"id": f"control-{i}", "prompt": f"w4 w5 w{6 + i}", "category": "benign"} for i in range(n)]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _manifest(tmp_path, source_path, control_path, apply_vectors_from="source_recipe"):
    control_dataset = {"adapter": "always_test", "path": str(control_path)}
    if apply_vectors_from is not None:
        control_dataset["apply_vectors_from"] = apply_vectors_from
    return {
        "experiment": {"name": "cross-recipe-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [
            {
                "id": "source_recipe", "behavior": "harmful_instruction_compliance",
                "dataset": {"adapter": "local_jsonl", "path": str(source_path)},
                "extraction": ["mean_diff"],
                "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
            },
            {
                "id": "control_recipe", "behavior": "harmful_instruction_compliance",
                "dataset": control_dataset,
                # No `extraction`/`application` needed -- this recipe never
                # extracts its own vector.
            },
        ],
        "decoding": {"max_new_tokens": 4},
    }


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


def test_control_recipe_has_zero_steer_examples(tmp_path):
    # Sanity check on the fixture itself: confirms _AlwaysTestAdapter
    # actually reproduces the "every row lands in test" property before
    # testing anything about apply_vectors_from.
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path, apply_vectors_from=None)

    from steering_factory.artifacts import ArtifactStore
    from steering_factory.runner import _load_normalized_records
    store = ArtifactStore(manifest["artifacts"]["root"], manifest, "test")
    examples = _load_normalized_records(manifest, store)
    control_examples = [e for e in examples if e["recipe_id"] == "control_recipe"]
    assert control_examples  # loaded something
    assert all(e["split"] == "test" for e in control_examples)


def test_control_recipe_without_apply_vectors_from_contributes_nothing(tmp_path):
    # Baseline: confirms the bug this feature fixes actually reproduces --
    # without apply_vectors_from, the control recipe's rows never appear.
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path, apply_vectors_from=None)

    store = run_steering(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert all(r["recipe_id"] != "control_recipe" for r in rows)


def test_run_steering_applies_source_vectors_to_control_recipe(tmp_path):
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path)

    store = run_steering(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    control_rows = [r for r in rows if r["recipe_id"] == "control_recipe"]
    assert len(control_rows) > 0
    # Every control row's example_id must come from the control dataset,
    # not the source recipe's.
    assert all(r["example_id"].startswith("control-") for r in control_rows)
    # And the vector applied is one of source_recipe's methods.
    assert all(r["method"] == "mean_diff" for r in control_rows)


def test_cross_recipe_rows_use_the_source_recipes_application_grid(tmp_path):
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path)
    manifest["recipes"][0]["application"]["coefficients"] = [-1.0, 1.0]

    store = run_steering(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    control_rows = [r for r in rows if r["recipe_id"] == "control_recipe"]
    coeffs = {r["coefficient"] for r in control_rows}
    assert coeffs == {-1.0, 0.0, 1.0}  # source grid's coefficients (+ implicit baseline)


def test_extract_then_evaluate_also_applies_cross_recipe(tmp_path):
    # The split (extract-only, then evaluate-separately) path must behave
    # identically to the combined run_steering path -- both call the same
    # _apply_vectors_cross_recipe helper.
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path)

    extract_store = run_extract(manifest, command="test")
    evaluate_store = run_evaluate(manifest, extract_store.path, command="test")
    rows = _read_jsonl(evaluate_store.path / "results" / "generations.jsonl")
    control_rows = [r for r in rows if r["recipe_id"] == "control_recipe"]
    assert len(control_rows) > 0


def test_apply_vectors_from_with_no_matching_source_vectors_is_a_noop(tmp_path):
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path, apply_vectors_from="nonexistent_recipe")

    store = run_steering(manifest, command="test")  # must not raise
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert all(r["recipe_id"] != "control_recipe" for r in rows)


def test_safety_summary_includes_cross_recipe_rows_for_false_refusal(tmp_path):
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path)

    store = run_steering(manifest, command="test")
    summary_path = store.path / "results" / "safety_summary.json"
    assert summary_path.exists()
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert any(r["recipe_id"] == "control_recipe" and "safe_refusal" in r for r in rows)
