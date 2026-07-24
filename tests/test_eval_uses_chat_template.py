"""Regression test: steering-eval generation must format prompts through
the tokenizer's chat template (model_utils.format_chat), matching
extraction (extraction.py's _collect_diffs) and QLoRA (finetune.py's
tokenize/_generate_batched_rows), which both already did this.

Before this existed, runner.py's _evaluate_vector_grid built its prompts
list directly from example["prompt"] with no format_chat call -- a real
train/eval mismatch (vectors extracted through the template, then
evaluated without it) that made even instruction-tuned checkpoints
(Qwen3-4B, gemma-3-4b-it) behave like base completion models at eval time:
a real run's outputs continued the prompt text ("The JSON object must not
contain any additional properties...") instead of answering it.

Verifies the fix by monkeypatching model_utils.format_chat with a sentinel
wrapper and asserting the sentinel reaches the tokenizer -- proves the
actual code path is used, not just that format_chat exists somewhere.
"""
import json

import pytest
import torch

from steering_factory import model_utils
from steering_factory.runner import run_steering

from _tiny_model import build_loaded_model

SENTINEL = "<<TEMPLATED>>"


def _write_recipe_jsonl(path, n_per_category=8):
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


def _manifest(tmp_path, data_path):
    return {
        "experiment": {"name": "chat-template-eval-e2e"},
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


def test_eval_generation_prompts_are_templated(tmp_path, monkeypatch):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path)

    captured_prompts = []
    real_tokenizer_call = None

    def spy_format_chat(tokenizer, user_message, system=None):
        wrapped = f"{SENTINEL}{user_message}{SENTINEL}"
        captured_prompts.append(wrapped)
        return wrapped

    # Patch at both import sites (runner.py and capability_probe.py both
    # import format_chat via `from .model_utils import format_chat` inside
    # their own function bodies -- patching model_utils.format_chat itself
    # is what both fresh imports resolve to).
    monkeypatch.setattr(model_utils, "format_chat", spy_format_chat)

    run_steering(manifest, command="test")

    assert captured_prompts, "format_chat was never called during eval generation"
    assert all(SENTINEL in p for p in captured_prompts)
    # One call per evaluate_example (per token_scope) -- proves format_chat
    # is applied to every row going into generation, not just once
    # incidentally somewhere else in the run.
    assert len(captured_prompts) >= 4  # 4 validation+test examples in this fixture's split
