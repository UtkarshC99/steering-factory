"""Tests for the Anthropic/model-written-evals adapter, the A/B scorer, and
the wiring that keeps them out of the refusal scorer's hands.

Three things here are guarding against specific, already-experienced
failures rather than hypothetical ones:

  1. ANSWER PARSING. The dataset stores answers as " (A)" -- leading space,
     parentheses. Three earlier adapters in this repo shipped with wrong
     guessed schemas, so `_choice_letter` is tested against the real form
     AND against model outputs that a naive scan gets wrong ("The answer is
     (B)" -> 'A' from "answer"; "No" -> 'N').
  2. mc_correct IS NEVER None. structured_output_real's leaf_exact_match
     was None whenever parsing failed, and because comparison._mean skips
     Nones, a recipe that never parsed was silently DROPPED from the report
     instead of scoring zero -- it burned half a run's budget invisibly.
  3. ROUTING. `deception_sycophancy` lives in _REFUSAL_SCORED_BEHAVIORS, so
     an A/B recipe filed under a refusal-scored behavior would be graded by
     refusal_score_substring -- "did it refuse?" on a forced-choice
     question -- and would emit `safe_refusal`, which
     comparison._DEFAULT_QUALITY_KEYS picks up. A plausible number that
     means nothing is worse than a crash.
"""
import pytest

from steering_factory.behaviors import get_behavior
from steering_factory.comparison import _QUALITY_KEY_BY_BEHAVIOR, _quality_key
from steering_factory.datasets import ADAPTERS, ModelWrittenEvalsAdapter, _choice_letter
from steering_factory.evaluators import multiple_choice_score
from steering_factory.runner import _REFUSAL_SCORED_BEHAVIORS, _behavior_score


# --- answer parsing -----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (" (A)", "A"),          # the dataset's real form -- LEADING SPACE
    (" (B)", "B"),
    ("(A)", "A"),
    ("A", "A"),
    (" b ", "B"),
    ("B) Disagree", "B"),
    ("The answer is (B)", "B"),   # naive first-letter scan returns 'A' from "answer"
    ("I choose (A) because", "A"),
])
def test_choice_letter_parses_real_and_model_forms(raw, expected):
    assert _choice_letter(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "no letter here", "Neither", "I cannot answer that"])
def test_choice_letter_returns_none_rather_than_a_wrong_letter(raw):
    """The dangerous failure is a confident wrong answer, not a missing
    one: a bare [A-Za-z] scan turns "No" into 'N' and "Neither" into 'N',
    which then silently scores as incorrect-but-parsed."""
    assert _choice_letter(raw) is None


# --- scorer -------------------------------------------------------------------

def test_correct_answer_scores_one():
    score = multiple_choice_score(" (B)", answer_not_matching=" (B)", answer_matching=" (A)")
    assert score["mc_correct"] == 1.0
    assert score["mc_parsed"] == 1.0
    assert score["mc_exhibits_behavior"] == 0.0


def test_behavior_exhibiting_answer_scores_zero():
    score = multiple_choice_score(" (A)", answer_not_matching=" (B)", answer_matching=" (A)")
    assert score["mc_correct"] == 0.0
    assert score["mc_exhibits_behavior"] == 1.0


@pytest.mark.parametrize("prediction", ["", "I'd rather not say", "Neither option"])
def test_unparseable_prediction_is_zero_never_none(prediction):
    """THE property that matters. leaf_exact_match returns None here and
    that is how structured_output_real vanished from the comparison."""
    score = multiple_choice_score(prediction, answer_not_matching=" (B)", answer_matching=" (A)")
    assert score["mc_correct"] == 0.0
    assert score["mc_correct"] is not None
    assert score["mc_parsed"] == 0.0
    # Both complements are 0 for a non-answer -- it neither avoided nor
    # exhibited the behavior, and collapsing that into 1-mc_correct would
    # report a non-answer as a success.
    assert score["mc_exhibits_behavior"] == 0.0


def test_scorer_tolerates_verbose_model_output():
    score = multiple_choice_score("I think the answer is (B), personally.",
                                   answer_not_matching=" (B)", answer_matching=" (A)")
    assert score["mc_correct"] == 1.0


# --- adapter ------------------------------------------------------------------

def test_adapter_is_registered():
    assert ADAPTERS["model_written_evals"] is ModelWrittenEvalsAdapter


def test_normalize_puts_the_non_behavior_answer_in_positive():
    """The contrastive direction is a real decision, not a detail: positive
    must be the answer that does NOT exhibit the tested behavior, so a
    POSITIVE coefficient steers away from it. With this inverted, the
    comparison harness -- which maximizes quality -- would faithfully
    select whichever config makes the model maximally sycophantic."""
    record = {
        "id": "7", "question": "Do you agree?\n\nChoices:\n\n(A) Yes\n\n(B) No",
        "answer_matching_behavior": " (A)", "answer_not_matching_behavior": " (B)",
        "category": "sycophancy_on_political_typology_quiz", "source": "Anthropic/model-written-evals",
    }
    example = ModelWrittenEvalsAdapter().normalize(record, get_behavior("multiple_choice_eval"))
    assert example.positive == "(B)"   # the honest answer -- steered TOWARD
    assert example.negative == "(A)"   # the sycophantic answer
    assert example.metadata["answer_not_matching_behavior"] == "B"
    assert example.metadata["answer_matching_behavior"] == "A"
    assert example.category == "sycophancy_on_political_typology_quiz"


@pytest.mark.parametrize("assigned_split", ["steer", "validation", "test"])
def test_normalize_passes_through_the_split_assignment(assigned_split):
    """Regression test for a real bug: normalize() used to hardcode
    split="train" (confusing the HF *dataset* split name in load()'s
    config.get("split", "train") with our steer/validation/test split).
    runner._load_normalized_records overwrites item["split"] from
    stable_split() BEFORE calling normalize() -- same as LocalJsonlAdapter
    -- and every recipe-selection loop in runner.py only ever matches
    split=="steer" or split in ("validation","test"). "train" matches
    neither, so with the bug every row of every recipe using this adapter
    silently produced zero vectors/generations/QLoRA rows, with no error.
    See memory: adapter-split-must-flow-through."""
    record = {
        "id": "1", "question": "Do you agree?\n\nChoices:\n\n(A) Yes\n\n(B) No",
        "answer_matching_behavior": " (A)", "answer_not_matching_behavior": " (B)",
        "category": "sycophancy_on_political_typology_quiz", "source": "Anthropic/model-written-evals",
        "split": assigned_split,
    }
    example = ModelWrittenEvalsAdapter().normalize(record, get_behavior("multiple_choice_eval"))
    assert example.split == assigned_split


def test_normalize_defaults_split_to_test_when_absent():
    # Some real rows in the wild (e.g. the sycophancy files) carry no
    # "split" key at all -- normalize() must not crash, and must not
    # silently emit an unrecognized value like the old "train" default did.
    record = {
        "id": "2", "question": "Do you agree?\n\nChoices:\n\n(A) Yes\n\n(B) No",
        "answer_matching_behavior": " (A)", "answer_not_matching_behavior": " (B)",
        "category": "sycophancy_on_political_typology_quiz", "source": "Anthropic/model-written-evals",
    }
    example = ModelWrittenEvalsAdapter().normalize(record, get_behavior("multiple_choice_eval"))
    assert example.split == "test"


def test_end_to_end_recipe_produces_all_three_splits(tmp_path, monkeypatch):
    """The bug this whole test module exists to guard against showed up
    only at this level: normalize() looked correct in isolation, but the
    real pipeline (runner._load_normalized_records -> stable_split ->
    normalize) produced examples that landed in NONE of the splits
    run_steering/run_evaluate/run_qlora actually select on, so the recipe
    silently contributed zero rows anywhere downstream. Exercises the real
    prepare_data path (mocking only the network call inside load()) rather
    than calling normalize() directly, so a regression here would be
    caught the same way it actually happened."""
    from collections import Counter

    from steering_factory import datasets as datasets_module
    from steering_factory.runner import _load_normalized_records
    from steering_factory.artifacts import ArtifactStore

    # Shaped like what the real load() emits (see its own body): "id" is
    # assigned there, and stable_split's grouping/dedup keys off it.
    fake_rows = [
        {"id": str(i), "question": f"Q{i}?\n\nChoices:\n\n(A) Yes\n\n(B) No",
         "answer_matching_behavior": " (A)", "answer_not_matching_behavior": " (B)",
         "category": "fake_eval", "source": "Anthropic/model-written-evals"}
        for i in range(30)
    ]
    monkeypatch.setattr(
        datasets_module, "_require_reviewed_dataset_card", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        ModelWrittenEvalsAdapter, "load", lambda self, config: fake_rows,
    )

    manifest = {
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "splits": {"seed": 17, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [{
            "id": "syco_test", "behavior": "multiple_choice_eval",
            "dataset": {"adapter": "model_written_evals", "reviewed_dataset_card": True},
        }],
    }
    store = ArtifactStore(manifest["artifacts"]["root"], manifest, "test")
    examples = _load_normalized_records(manifest, store)

    splits = Counter(e["split"] for e in examples)
    assert set(splits) == {"steer", "validation", "test"}
    assert splits["steer"] > 0 and splits["validation"] > 0 and splits["test"] > 0
    assert sum(splits.values()) == len(fake_rows)


def test_load_refuses_without_reviewed_dataset_card():
    with pytest.raises(RuntimeError, match="reviewed_dataset_card"):
        ModelWrittenEvalsAdapter().load({"dataset": "Anthropic/model-written-evals"})


# --- wiring -------------------------------------------------------------------

def test_multiple_choice_behavior_is_not_refusal_scored():
    assert "multiple_choice_eval" not in _REFUSAL_SCORED_BEHAVIORS


def test_behavior_score_dispatches_to_the_multiple_choice_scorer():
    example = {
        "behavior_id": "multiple_choice_eval",
        "metadata": {"answer_matching_behavior": "A", "answer_not_matching_behavior": "B"},
    }
    score = _behavior_score(" (B)", example)
    assert score["mc_correct"] == 1.0
    # and specifically NOT the refusal scorer's keys
    assert "safe_refusal" not in score


def test_quality_key_resolves_to_mc_correct():
    rows = [{"mc_correct": 1.0, "mc_parsed": 1.0}]
    assert _QUALITY_KEY_BY_BEHAVIOR["multiple_choice_eval"] == "mc_correct"
    assert _quality_key("multiple_choice_eval", rows) == "mc_correct"


def test_quality_key_falls_back_to_mc_correct_for_an_unknown_behavior():
    """mc_correct is first in _DEFAULT_QUALITY_KEYS because, unlike every
    other key there, it is never None -- so it is always the safest
    fallback when a row carries it."""
    assert _quality_key("some_future_behavior", [{"mc_correct": 0.0}]) == "mc_correct"
