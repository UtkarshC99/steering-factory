"""Tests for the LLM-judge refusal scorer, the safety metric bundle
(evaluators.py), and their wiring into runner._behavior_score /
run_steering / run_qlora.
"""
import json

import pytest

from steering_factory import model_utils
from steering_factory.evaluators import (
    REFUSAL_JUDGE_BEHAVIOR_DESCRIPTION,
    refusal_score_judge,
    refusal_score_substring,
    safety_metric_bundle,
)
from steering_factory.runner import _behavior_score, run_steering

from _tiny_model import build_loaded_model


# --- refusal_score_substring (baseline, unchanged behavior) --------------------

def test_refusal_score_substring_detects_known_markers():
    assert refusal_score_substring("I can't help with that.")["safe_refusal"] == 1.0
    assert refusal_score_substring("Sure, here's how.")["safe_refusal"] == 0.0


def test_refusal_score_substring_fields_are_complementary():
    result = refusal_score_substring("I cannot assist with that request.")
    assert result["safe_refusal"] + result["unsafe_compliance"] == 1.0


# --- refusal_score_judge --------------------------------------------------------

class _StubJudge:
    def __init__(self, score, model="stub-judge-v1"):
        self._score = score
        self.model = model

    def score_behavior(self, text, behavior_description):
        return self._score


class _DecliningJudge:
    model = "declining-judge"

    def score_behavior(self, text, behavior_description):
        return None  # simulates a failed/unavailable judge call


def test_refusal_score_judge_thresholds_continuous_score_to_binary():
    safe = refusal_score_judge("some text", _StubJudge(0.9))
    unsafe = refusal_score_judge("some text", _StubJudge(0.1))
    assert safe["safe_refusal"] == 1.0
    assert unsafe["safe_refusal"] == 0.0


def test_refusal_score_judge_records_provenance():
    result = refusal_score_judge("some text", _StubJudge(0.8))
    assert result["judge_available"] is True
    assert result["judge_model"] == "stub-judge-v1"
    assert result["judge_rubric_hash"] is not None
    assert result["judge_raw_score"] == 0.8


def test_refusal_score_judge_uses_the_expected_behavior_description_by_default():
    captured = {}

    class _CapturingJudge:
        model = "capture"

        def score_behavior(self, text, behavior_description):
            captured["behavior_description"] = behavior_description
            return 0.5

    refusal_score_judge("text", _CapturingJudge())
    assert captured["behavior_description"] == REFUSAL_JUDGE_BEHAVIOR_DESCRIPTION


def test_refusal_score_judge_falls_back_to_substring_when_judge_declines():
    result = refusal_score_judge("I can't help with that.", _DecliningJudge())
    assert result["judge_available"] is False
    assert result["judge_model"] is None
    assert result["judge_raw_score"] is None
    assert result["safe_refusal"] == 1.0  # substring fallback still finds the marker


def test_refusal_score_judge_none_judge_falls_back_to_substring():
    result = refusal_score_judge("I can't help with that.", judge=None)
    assert result["judge_available"] is False
    assert result["safe_refusal"] == 1.0


def test_refusal_score_judge_threshold_is_configurable():
    borderline = refusal_score_judge("text", _StubJudge(0.6), threshold=0.7)
    assert borderline["safe_refusal"] == 0.0  # 0.6 < 0.7
    borderline_pass = refusal_score_judge("text", _StubJudge(0.6), threshold=0.5)
    assert borderline_pass["safe_refusal"] == 1.0


# --- _behavior_score wiring ------------------------------------------------------

def test_behavior_score_uses_substring_when_no_judge_given():
    example = {"behavior_id": "harmful_instruction_compliance", "metadata": {}}
    result = _behavior_score("I can't help with that.", example)
    assert result["safe_refusal"] == 1.0
    assert "judge_available" not in result  # substring path has no judge fields at all


def test_behavior_score_uses_judge_when_given():
    example = {"behavior_id": "harmful_instruction_compliance", "metadata": {}}
    result = _behavior_score("some text", example, judge=_StubJudge(0.9))
    assert result["judge_available"] is True
    assert result["safe_refusal"] == 1.0


@pytest.mark.parametrize("behavior_id", [
    "defensive_refusal", "harmful_instruction_compliance", "jailbreak_susceptibility",
    "deception_sycophancy", "pii_leakage",
])
def test_behavior_score_covers_every_safety_behavior(behavior_id):
    example = {"behavior_id": behavior_id, "metadata": {}}
    result = _behavior_score("I can't help with that.", example)
    assert result["safe_refusal"] == 1.0


def test_behavior_score_unrelated_family_ignores_judge():
    # A judge object being present must not change scoring for families
    # that don't use refusal_score_* at all.
    example = {"behavior_id": "domain_classification", "positive": "cats", "metadata": {}}
    result = _behavior_score("cats", example, judge=_StubJudge(0.1))
    assert result == {"exact_match": 1.0, "valid_label": 1.0}
    assert "judge_available" not in result


# --- safety_metric_bundle --------------------------------------------------------

def test_safety_metric_bundle_reports_rates_and_per_category():
    rows = [
        {"safe_refusal": 1.0, "unsafe_compliance": 0.0, "category": "cybercrime"},
        {"safe_refusal": 0.0, "unsafe_compliance": 1.0, "category": "cybercrime"},
        {"safe_refusal": 1.0, "unsafe_compliance": 0.0, "category": "chemical"},
    ]
    bundle = safety_metric_bundle(rows)
    assert bundle["n"] == 3
    assert bundle["safe_refusal_rate"] == pytest.approx(2 / 3)
    assert bundle["unsafe_compliance_rate"] == pytest.approx(1 / 3)
    assert bundle["per_category"]["cybercrime"]["safe_refusal_rate"] == pytest.approx(0.5)
    assert bundle["per_category"]["chemical"]["safe_refusal_rate"] == pytest.approx(1.0)


def test_safety_metric_bundle_false_refusal_only_counts_is_safe_control_true():
    rows = [
        {"safe_refusal": 1.0, "unsafe_compliance": 0.0, "category": "safe_1", "is_safe_control": True},   # false refusal
        {"safe_refusal": 0.0, "unsafe_compliance": 1.0, "category": "safe_1", "is_safe_control": True},   # correctly answered
        {"safe_refusal": 1.0, "unsafe_compliance": 0.0, "category": "contrast_1", "is_safe_control": False},  # correct refusal
    ]
    bundle = safety_metric_bundle(rows)
    assert bundle["false_refusal_rate_on_benign_controls"] == pytest.approx(0.5)
    assert bundle["n_benign_controls"] == 2
    assert bundle["unsafe_contrast_safe_refusal_rate"] == pytest.approx(1.0)
    assert bundle["n_unsafe_contrast"] == 1


def test_safety_metric_bundle_non_xstest_rows_are_not_counted_as_benign_controls():
    # HarmBench-style categories (no is_safe_control field at all) must
    # NOT be silently treated as benign controls -- this was a real bug
    # caught during development (an earlier version inferred "benign" from
    # category not starting with "contrast_", which misclassified every
    # non-XSTest safety recipe's rows).
    rows = [
        {"safe_refusal": 1.0, "unsafe_compliance": 0.0, "category": "cybercrime"},
        {"safe_refusal": 0.0, "unsafe_compliance": 1.0, "category": "chemical"},
    ]
    bundle = safety_metric_bundle(rows)
    assert bundle["n_benign_controls"] == 0
    assert bundle["false_refusal_rate_on_benign_controls"] is None


def test_safety_metric_bundle_harmless_task_regression_from_capability_rows():
    capability_rows = [
        {"capability_retention": 1.0}, {"capability_retention": 0.5}, {"capability_retention": None},
    ]
    bundle = safety_metric_bundle([], capability_rows=capability_rows)
    assert bundle["harmless_task_regression"] == pytest.approx(0.75)  # mean of [1.0, 0.5], None excluded


def test_safety_metric_bundle_harmless_task_regression_none_when_not_measured():
    bundle = safety_metric_bundle([{"safe_refusal": 1.0, "unsafe_compliance": 0.0}])
    assert bundle["harmless_task_regression"] is None


def test_safety_metric_bundle_empty_rows():
    bundle = safety_metric_bundle([])
    assert bundle["n"] == 0
    assert bundle["safe_refusal_rate"] is None
    assert bundle["per_category"] == {}


# --- end-to-end: judge wired into run_steering (tiny fixture) ------------------

def _write_recipe_jsonl(path):
    records = []
    for cat in ("alpha", "beta"):
        for i in range(8):
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


@pytest.fixture(autouse=True)
def _patch_load_model(monkeypatch):
    monkeypatch.setattr(model_utils, "load_model", lambda cfg: build_loaded_model(seed=5))


def test_run_steering_with_judge_records_judge_fields_on_safety_rows(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = {
        "experiment": {"name": "judge-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": [{
            "id": "toy_safety", "behavior": "harmful_instruction_compliance",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
        }],
        "decoding": {"max_new_tokens": 4},
    }

    store = run_steering(manifest, command="test", judge=_StubJudge(0.9))
    rows = []
    with open(store.path / "results" / "generations.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    assert rows
    for row in rows:
        assert row["judge_available"] is True
        assert row["judge_model"] == "stub-judge-v1"


def test_run_steering_writes_safety_summary_for_safety_recipe(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = {
        "experiment": {"name": "safety-summary-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": [{
            "id": "toy_safety", "behavior": "harmful_instruction_compliance",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
        }],
        "decoding": {"max_new_tokens": 4},
    }
    store = run_steering(manifest, command="test")
    summary_path = store.path / "results" / "safety_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "safe_refusal_rate" in summary
    assert "per_category" in summary


def test_run_steering_omits_safety_summary_for_non_safety_recipe(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = {
        "experiment": {"name": "non-safety-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": [{
            "id": "toy_classification", "behavior": "domain_classification",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
        }],
        "decoding": {"max_new_tokens": 4},
    }
    store = run_steering(manifest, command="test")
    assert not (store.path / "results" / "safety_summary.json").exists()


def test_run_steering_without_judge_uses_substring_scoring(tmp_path):
    data_path = tmp_path / "toy.jsonl"
    _write_recipe_jsonl(data_path)
    manifest = {
        "experiment": {"name": "no-judge-e2e"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.4, "validation_fraction": 0.3, "group_key": "category"},
        "recipes": [{
            "id": "toy_safety", "behavior": "harmful_instruction_compliance",
            "dataset": {"adapter": "local_jsonl", "path": str(data_path)},
            "extraction": ["mean_diff"],
            "application": {"layers": [0], "coefficients": [1.0], "token_scopes": ["all"]},
        }],
        "decoding": {"max_new_tokens": 4},
    }
    store = run_steering(manifest, command="test")  # no judge -- default CLI-equivalent path
    rows = []
    with open(store.path / "results" / "generations.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    assert rows
    for row in rows:
        assert "judge_available" not in row
        assert "safe_refusal" in row
