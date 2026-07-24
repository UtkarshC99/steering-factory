"""Tests for `apply_adapter_from`: the QLoRA counterpart to
`apply_vectors_from` (test_apply_vectors_cross_recipe.py). A recipe with no
train pool of its own (the XSTest shape -- every row forced to
split="test") gets evaluated against a DIFFERENT recipe's already-trained
QLoRA adapter, instead of contributing zero QLoRA rows.

Before this existed, run_qlora's main loop skipped any recipe with no
train_records (`if not train_records: continue`), so benign_over_refusal_control
had a steering arm (via apply_vectors_from) but NO QLoRA arm at all -- there
was no way to check whether a QLoRA safety adapter also over-refuses benign
prompts. A real run's QLoRA safety adapter collapsed to the identical
canned refusal string on 372/372 harmful-recipe rows, making this exactly
the check needed to tell "overfit to refusal" apart from "genuinely
calibrated."

Monkeypatches steering_factory.finetune.train_qlora/evaluate_qlora_adapter
directly (run_qlora imports them fresh, by name, from .finetune inside its
own function body), so this needs no real peft/trl/GPU -- mirrors
test_qlora_recipe_batch_size.py's stubbing approach. Uses the same
_AlwaysTestAdapter stand-in as test_apply_vectors_cross_recipe.py to
reproduce XSTest's "zero steer examples" property without its real
schema/HF loading.
"""
import json

import pytest

from steering_factory import datasets as datasets_module
from steering_factory import finetune as finetune_module
from steering_factory.experiment_types import ContrastiveExample
from steering_factory.runner import run_qlora


class _AlwaysTestAdapter:
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


def _write_recipe_jsonl(path, n_per_category=6):
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


def _write_control_jsonl(path, n=10):
    records = [{"id": f"control-{i}", "prompt": f"w4 w5 w{6 + i}", "category": "benign"} for i in range(n)]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _manifest(tmp_path, source_path, control_path, apply_adapter_from="source_recipe"):
    control_dataset = {"adapter": "always_test", "path": str(control_path)}
    if apply_adapter_from is not None:
        control_dataset["apply_adapter_from"] = apply_adapter_from
    return {
        "experiment": {"name": "cross-recipe-adapter-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [
            {
                "id": "source_recipe", "behavior": "harmful_instruction_compliance",
                "dataset": {"adapter": "local_jsonl", "path": str(source_path)},
            },
            {
                "id": "control_recipe", "behavior": "harmful_instruction_compliance",
                "dataset": control_dataset,
                # No train pool -- this recipe never trains its own adapter.
            },
        ],
        "decoding": {"max_new_tokens": 4},
        "finetune": {"backend": "qlora", "target_modules": ["q_proj"], "max_steps": 1},
    }


@pytest.fixture
def _stub_finetune(monkeypatch):
    calls = {"trained": [], "evaluated": []}

    def fake_train_qlora(records, config, callback=None, callback_context=None):
        calls["trained"].append(dict(callback_context or {}))
        return {"adapter_dir": "unused-adapter-dir", "train_loss": 0.0, "global_step": 1,
                "wall_time_s": 0.01, "adapter_size_bytes": 0, "log_history": []}

    def fake_evaluate_qlora_adapter(examples, model_name, adapter_dir, max_new_tokens=96,
                                     trust_remote_code=False, batch_size=16,
                                     quantization="4bit", dtype=None):
        calls["evaluated"].append({"adapter_dir": adapter_dir, "example_ids": [e["id"] for e in examples]})
        rows = [{"example_id": e["id"], "behavior_id": e["behavior_id"], "split": e["split"],
                  "category": e.get("category"), "prompt": e["prompt"], "output": "stub-output",
                  "latency_s": 0.01, "batch_size": batch_size, "batch_wall_time_s": 0.01,
                  "tokens_generated": 1} for e in examples]
        return {"rows": rows, "wall_time_s": 0.01}

    monkeypatch.setattr(finetune_module, "train_qlora", fake_train_qlora)
    monkeypatch.setattr(finetune_module, "evaluate_qlora_adapter", fake_evaluate_qlora_adapter)
    return calls


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_control_recipe_without_apply_adapter_from_contributes_no_qlora_rows(tmp_path, _stub_finetune):
    # Baseline: confirms the pre-existing gap actually reproduces --
    # without apply_adapter_from, control_recipe (zero train examples)
    # never appears in the QLoRA arm's generations at all.
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path, apply_adapter_from=None)

    store = run_qlora(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert all(r["recipe_id"] != "control_recipe" for r in rows)


def test_run_qlora_evaluates_source_adapter_against_control_recipe(tmp_path, _stub_finetune):
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path)

    store = run_qlora(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    control_rows = [r for r in rows if r["recipe_id"] == "control_recipe"]
    assert len(control_rows) > 0
    # Every control row's example_id must come from the control dataset.
    assert all(r["example_id"].startswith("control-") for r in control_rows)
    # Only one adapter was ever trained (source_recipe's); it was reused,
    # not retrained, for the control evaluation.
    assert len(_stub_finetune["trained"]) == 1


def test_cross_recipe_adapter_eval_reuses_the_trained_adapter_dir(tmp_path, _stub_finetune):
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path)

    run_qlora(manifest, command="test")

    # Two evaluate_qlora_adapter calls: one for source_recipe's own eval
    # examples, one for control_recipe's (cross-recipe) -- both against the
    # SAME adapter_dir, since only one adapter was trained.
    assert len(_stub_finetune["evaluated"]) == 2
    adapter_dirs = {call["adapter_dir"] for call in _stub_finetune["evaluated"]}
    assert adapter_dirs == {"unused-adapter-dir"}
    control_eval_ids = next(c["example_ids"] for c in _stub_finetune["evaluated"]
                             if any(i.startswith("control-") for i in c["example_ids"]))
    assert all(i.startswith("control-") for i in control_eval_ids)


def test_apply_adapter_from_with_no_matching_source_adapter_is_a_noop(tmp_path, _stub_finetune):
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path, apply_adapter_from="nonexistent_recipe")

    store = run_qlora(manifest, command="test")  # must not raise
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    assert all(r["recipe_id"] != "control_recipe" for r in rows)


def test_cross_recipe_rows_tagged_with_matching_num_train_records(tmp_path, _stub_finetune):
    # Cross-recipe rows must carry the SOURCE adapter's num_train_records
    # (not e.g. 0 or the control recipe's own example count), so
    # comparison.py's max-N pinning still matches them to the right
    # source-side training point.
    source_path, control_path = tmp_path / "source.jsonl", tmp_path / "control.jsonl"
    _write_recipe_jsonl(source_path)
    _write_control_jsonl(control_path)
    manifest = _manifest(tmp_path, source_path, control_path)

    store = run_qlora(manifest, command="test")
    rows = _read_jsonl(store.path / "results" / "generations.jsonl")
    source_rows = [r for r in rows if r["recipe_id"] == "source_recipe"]
    control_rows = [r for r in rows if r["recipe_id"] == "control_recipe"]
    assert source_rows and control_rows
    source_n = {r["num_train_records"] for r in source_rows}
    control_n = {r["num_train_records"] for r in control_rows}
    assert control_n == source_n
