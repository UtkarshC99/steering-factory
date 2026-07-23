"""Regression test: run_qlora's adapter-eval generation must honor a
per-recipe decoding.batch_size override, matching run_steering/run_evaluate's
identical override -- otherwise a recipe with much longer prompts than the
manifest's other recipes (e.g. structured_output_real's JSON-schema prompts)
runs QLoRA eval at the full manifest-wide batch size, exposed to the same
variable-size-prompt VRAM spike that produced a real OOM on the steering arm
before ITS override was added. finetune.evaluate_qlora_adapter's own
generation loop (_generate_batched_rows) uses the identical padding=True
batching pattern as the steering arm's sweep.generate_with_steering_batch.

Monkeypatches steering_factory.finetune.train_qlora/evaluate_qlora_adapter
directly (run_qlora imports them fresh, by name, from .finetune inside its
own function body) so this test needs no real peft/trl/GPU -- it only
checks what batch_size run_qlora PASSES to evaluate_qlora_adapter, not
evaluate_qlora_adapter's own internals (covered by test_finetune_batching.py
and test_finetune_quantization.py).
"""
import json

import pytest

from steering_factory import finetune as finetune_module
from steering_factory.runner import run_qlora


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


def _manifest(tmp_path, big_path, small_path, small_recipe_batch_size=None):
    small_dataset_recipe = {
        "id": "small_recipe", "behavior": "domain_classification",
        "dataset": {"adapter": "local_jsonl", "path": str(small_path)},
    }
    if small_recipe_batch_size is not None:
        small_dataset_recipe["decoding"] = {"batch_size": small_recipe_batch_size}
    return {
        "experiment": {"name": "qlora-recipe-batch-size-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [
            {
                "id": "big_recipe", "behavior": "domain_classification",
                "dataset": {"adapter": "local_jsonl", "path": str(big_path)},
            },
            small_dataset_recipe,
        ],
        "decoding": {"max_new_tokens": 4, "batch_size": 32},
        "finetune": {"backend": "qlora", "target_modules": ["q_proj"], "max_steps": 1},
    }


@pytest.fixture
def _stub_finetune(monkeypatch):
    calls = {"eval_batch_sizes": []}

    def fake_train_qlora(records, config, callback=None, callback_context=None):
        return {"adapter_dir": "unused-adapter-dir", "train_loss": 0.0, "global_step": 1,
                "wall_time_s": 0.01, "adapter_size_bytes": 0, "log_history": []}

    def fake_evaluate_qlora_adapter(examples, model_name, adapter_dir, max_new_tokens=96,
                                     trust_remote_code=False, batch_size=16,
                                     quantization="4bit", dtype=None):
        calls["eval_batch_sizes"].append(batch_size)
        rows = [{"example_id": e["id"], "behavior_id": e["behavior_id"], "split": e["split"],
                  "category": e.get("category"), "prompt": e["prompt"], "output": "stub-output",
                  "latency_s": 0.01, "batch_size": batch_size, "batch_wall_time_s": 0.01,
                  "tokens_generated": 1} for e in examples]
        return {"rows": rows, "wall_time_s": 0.01}

    monkeypatch.setattr(finetune_module, "train_qlora", fake_train_qlora)
    monkeypatch.setattr(finetune_module, "evaluate_qlora_adapter", fake_evaluate_qlora_adapter)
    return calls


def test_qlora_eval_uses_recipe_override_when_set(tmp_path, _stub_finetune):
    big_path, small_path = tmp_path / "big.jsonl", tmp_path / "small.jsonl"
    _write_recipe_jsonl(big_path)
    _write_recipe_jsonl(small_path)
    manifest = _manifest(tmp_path, big_path, small_path, small_recipe_batch_size=8)

    run_qlora(manifest, command="test")

    # big_recipe has no override -> falls back to manifest-wide 32.
    # small_recipe overrides to 8.
    assert set(_stub_finetune["eval_batch_sizes"]) == {32, 8}


def test_qlora_eval_falls_back_to_manifest_wide_batch_size_when_unset(tmp_path, _stub_finetune):
    big_path, small_path = tmp_path / "big.jsonl", tmp_path / "small.jsonl"
    _write_recipe_jsonl(big_path)
    _write_recipe_jsonl(small_path)
    manifest = _manifest(tmp_path, big_path, small_path, small_recipe_batch_size=None)

    run_qlora(manifest, command="test")

    # No recipe sets an override -> every call uses the manifest-wide 32,
    # preserving pre-existing behavior for every manifest written before
    # this override existed.
    assert _stub_finetune["eval_batch_sizes"] == [32, 32]
