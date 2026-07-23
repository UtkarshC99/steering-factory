"""Tests for the benign general-capability regression control.

Pure-logic pieces (scoring, retention math) are tested directly. The
generation path (`run_capability_probe_steering`) runs against the same
tiny Llama fixture the batching/N-sweep tests use -- reusing
sweep.generate_with_steering_batch means there is nothing
capability-probe-specific to break there beyond correct wiring, which is
what this proves.
"""
import pytest
import torch

from steering_factory.capability_probe import (
    CAPABILITY_PROBES,
    CapabilityProbe,
    capability_quality,
    capability_retention,
    run_capability_probe_steering,
    score_capability_probe,
)

from _tiny_model import build_loaded_model


# --- score_capability_probe / capability_quality -------------------------------

def test_score_capability_probe_matches_case_insensitively():
    probe = CapabilityProbe("x", "prompt", ("paris",))
    assert score_capability_probe("The answer is Paris.", probe) == 1.0
    assert score_capability_probe("PARIS", probe) == 1.0
    assert score_capability_probe("London", probe) == 0.0


def test_score_capability_probe_matches_any_of_multiple_expected():
    probe = CapabilityProbe("x", "prompt", ("3", "three"))
    assert score_capability_probe("the answer is three", probe) == 1.0
    assert score_capability_probe("3", probe) == 1.0
    assert score_capability_probe("four", probe) == 0.0


def test_capability_quality_is_mean_correctness():
    rows = [{"correct": 1.0}, {"correct": 0.0}, {"correct": 1.0}, {"correct": 1.0}]
    assert capability_quality(rows) == pytest.approx(0.75)


def test_capability_quality_returns_none_for_empty_rows():
    assert capability_quality([]) is None


def test_capability_probes_all_have_at_least_one_expected_substring():
    # Sanity check on the fixed probe set itself -- a probe with no
    # expected_substrings would always score 0 and silently corrupt every
    # retention ratio that includes it.
    for probe in CAPABILITY_PROBES:
        assert len(probe.expected_substrings) >= 1


# --- capability_retention --------------------------------------------------------

def test_capability_retention_is_after_over_before():
    before = [{"correct": 1.0}] * 8
    after = [{"correct": 0.5}] * 8  # 4 of 8 correct
    assert capability_retention(before, after) == pytest.approx(0.5)


def test_capability_retention_none_when_before_is_empty():
    assert capability_retention([], [{"correct": 1.0}]) is None


def test_capability_retention_none_when_after_is_empty():
    assert capability_retention([{"correct": 1.0}], []) is None


def test_capability_retention_none_when_base_model_already_scores_zero():
    # 0/0 division is meaningless, not "0.0 = total collateral loss" and
    # not "1.0 = perfectly retained" -- both would misreport a base model
    # that already couldn't do the probes at all.
    before = [{"correct": 0.0}] * 8
    after = [{"correct": 0.0}] * 8
    assert capability_retention(before, after) is None


def test_capability_retention_can_exceed_one():
    # Steering could plausibly IMPROVE general capability incidentally --
    # the ratio should not be artificially capped at 1.0.
    before = [{"correct": 0.5}] * 8
    after = [{"correct": 1.0}] * 8
    assert capability_retention(before, after) == pytest.approx(2.0)


# --- run_capability_probe_steering (real generation, tiny fixture) --------------

def test_run_capability_probe_steering_returns_one_row_per_probe():
    loaded = build_loaded_model(seed=9)
    vector = torch.randn(loaded.hidden_size)
    rows = run_capability_probe_steering(loaded, vector, layer_idx=0, coefficient=0.0, token_scope="all", max_new_tokens=3)
    assert len(rows) == len(CAPABILITY_PROBES)
    assert {row["probe_id"] for row in rows} == {p.id for p in CAPABILITY_PROBES}
    for row in rows:
        assert "output" in row and "correct" in row


def test_run_capability_probe_steering_zero_coefficient_is_deterministic_baseline():
    # Calling twice at coefficient=0.0 (the "before" measurement) must
    # produce identical text -- greedy decoding, no steering applied.
    loaded = build_loaded_model(seed=9)
    vector = torch.randn(loaded.hidden_size)
    rows_a = run_capability_probe_steering(loaded, vector, layer_idx=0, coefficient=0.0, token_scope="all", max_new_tokens=3)
    rows_b = run_capability_probe_steering(loaded, vector, layer_idx=0, coefficient=0.0, token_scope="all", max_new_tokens=3)
    assert [r["output"] for r in rows_a] == [r["output"] for r in rows_b]
