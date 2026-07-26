"""Regression test: a recipe's `decoding.system_prompt` must reach BOTH
extraction (extraction.py's _collect_diffs, via runner.py's
extraction.extract call) and eval generation
(runner.py's _evaluate_vector_grid) with the IDENTICAL value -- a system
prompt present at one and absent (or different) at the other is exactly
the same class of train/eval mismatch test_eval_uses_chat_template.py
guards against for chat-templating itself: the vector would be extracted
from activations under one framing and then applied/scored under another.

Added 2026-07-26 alongside format_chat gaining a `system` parameter, to
support a terse "answer only the letter" instruction for forced-choice
recipes (gemma-3-4b-it otherwise spends its whole token budget on
conversational preamble and never emits a parseable answer -- see
project memory gemma-needs-answer-first-instruction.md).
"""
import json

import pytest

from steering_factory import model_utils
from steering_factory.runner import run_steering

from _tiny_model import build_loaded_model

SYSTEM_PROMPT = "Answer with only the letter in parentheses. Do not explain."


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


def _manifest(tmp_path, data_path, system_prompt=SYSTEM_PROMPT):
    return {
        "experiment": {"name": "system-prompt-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [{
            "id": "toy_recipe", "behavior": "domain_classification",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
            "decoding": {"max_new_tokens": 4, "system_prompt": system_prompt},
        }],
        "decoding": {"max_new_tokens": 4},
    }


@pytest.fixture(autouse=True)
def _patch_load_model(monkeypatch):
    monkeypatch.setattr(model_utils, "load_model", lambda cfg: build_loaded_model(seed=5))


def test_system_prompt_reaches_both_extraction_and_eval(tmp_path, monkeypatch):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path)

    captured_systems = []

    def spy_format_chat(tokenizer, user_message, system=None):
        captured_systems.append(system)
        return f"<<{system}>>{user_message}"

    monkeypatch.setattr(model_utils, "format_chat", spy_format_chat)

    run_steering(manifest, command="test")

    assert captured_systems, "format_chat was never called"
    # EVERY call -- extraction's pos/neg pair encoding and eval generation
    # alike -- must see the recipe's system prompt, not just some of them.
    assert all(s == SYSTEM_PROMPT for s in captured_systems), (
        f"system prompt was not applied consistently: {set(captured_systems)}"
    )


def test_no_system_prompt_configured_is_a_no_op(tmp_path, monkeypatch):
    """A recipe with no decoding.system_prompt at all must behave exactly
    as before this feature existed -- every call site gets system=None,
    not some stale/default value."""
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = _manifest(tmp_path, data_path, system_prompt=None)
    manifest["recipes"][0]["decoding"] = {"max_new_tokens": 4}  # no system_prompt key

    captured_systems = []

    def spy_format_chat(tokenizer, user_message, system=None):
        captured_systems.append(system)
        return user_message

    monkeypatch.setattr(model_utils, "format_chat", spy_format_chat)

    run_steering(manifest, command="test")

    assert captured_systems, "format_chat was never called"
    assert all(s is None for s in captured_systems)
