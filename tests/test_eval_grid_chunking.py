"""Regression tests for `_evaluate_vector_grid` actually honoring
`batch_size`.

`batch_size` was accepted by this function, threaded through every caller,
and tuned per-recipe in the manifests -- but never used to chunk. `prompts`
went to `sweep.generate_with_steering_batch` whole, so the real batch was
always the ENTIRE eval pool. Proven before the fix by the rows' own
`batch_size` field: requesting 2 over 8 examples recorded 8.

That surfaced as a real CUDA OOM once `max_records` rose 100 -> 150, which
grew the eval pool (and therefore the actual batch) from 60 to 90 -- ~2.8x
the configured 32 -- and meant none of the per-recipe
`decoding.batch_size` overrides affected this arm at all. Extraction
(`extraction._collect_diffs`) and the QLoRA eval loop
(`finetune._generate_batched_rows`) both chunked correctly; this was the
one generation path that didn't.

The equivalence tests here are the important ones: chunking must not
change a single generated token, only how many rows share one forward
pass. Greedy decoding is per-row independent, so any difference would mean
padding or steering state leaking across chunk boundaries.
"""
import pytest
import torch

from steering_factory.runner import _evaluate_vector_grid

from _tiny_model import build_loaded_model


@pytest.fixture(scope="module")
def loaded():
    return build_loaded_model(seed=3)


def _examples(n=8):
    return [
        {"id": f"e{i}", "behavior_id": "domain_classification", "split": "test",
         "category": "a", "prompt": f"w4 w5 w{6 + i}", "positive": "w10", "negative": "w20"}
        for i in range(n)
    ]


def _run(loaded, batch_size, examples=None, coefficients=(1.0,), token_scopes=("all",)):
    torch.manual_seed(0)
    vector = torch.zeros(loaded.hidden_size)  # deterministic, non-degenerate path
    return _evaluate_vector_grid(
        loaded, vector, 0, "mean_diff", "r1", "m1", "tiny/tiny",
        examples if examples is not None else _examples(),
        list(coefficients), list(token_scopes),
        3, batch_size, None, {"done": 0, "total": 999}, None,
    )


@pytest.mark.parametrize("batch_size,expected", [(2, {2}), (4, {4}), (8, {8})])
def test_recorded_batch_size_matches_requested(loaded, batch_size, expected):
    rows = _run(loaded, batch_size)
    assert {r["batch_size"] for r in rows} == expected


def test_ragged_final_chunk_recorded_correctly(loaded):
    # 8 examples at batch_size=3 -> chunks of 3, 3, 2.
    rows = _run(loaded, 3)
    assert {r["batch_size"] for r in rows} == {3, 2}


def test_batch_size_larger_than_pool_is_capped_at_pool_size(loaded):
    rows = _run(loaded, 100)
    assert {r["batch_size"] for r in rows} == {8}


def test_batch_size_zero_or_none_falls_back_to_whole_pool(loaded):
    # Guards the `batch_size or len(prompts)` fallback -- a manifest with
    # no decoding.batch_size must not divide by zero or emit no rows.
    for bad in (0, None):
        rows = _run(loaded, bad)
        assert {r["batch_size"] for r in rows} == {8}


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 8])
def test_chunking_does_not_change_generated_output(loaded, batch_size):
    """The core guarantee: chunk size changes only how many rows share a
    forward pass, never a generated token."""
    reference = _run(loaded, 8)  # single batch
    chunked = _run(loaded, batch_size)
    assert [r["output"] for r in chunked] == [r["output"] for r in reference]
    assert [r["example_id"] for r in chunked] == [r["example_id"] for r in reference]
    assert [r["coefficient"] for r in chunked] == [r["coefficient"] for r in reference]


def test_chunking_preserves_rows_across_multiple_coefficients(loaded):
    # The coefficient sweep is the axis that multiplies generate() calls;
    # chunking must compose with it without dropping or reordering rows.
    coefficients = (-1.0, 0.0, 1.0)
    reference = _run(loaded, 8, coefficients=coefficients)
    chunked = _run(loaded, 3, coefficients=coefficients)
    assert len(chunked) == len(reference) == len(_examples()) * len(coefficients)
    assert [(r["example_id"], r["coefficient"]) for r in chunked] == \
           [(r["example_id"], r["coefficient"]) for r in reference]
    assert [r["output"] for r in chunked] == [r["output"] for r in reference]


def test_progress_counter_counts_every_row_once_under_chunking(loaded):
    progress = {"done": 0, "total": 999}
    torch.manual_seed(0)
    vector = torch.zeros(loaded.hidden_size)
    rows = _evaluate_vector_grid(
        loaded, vector, 0, "mean_diff", "r1", "m1", "tiny/tiny",
        _examples(), [1.0], ["all"], 3, 3, None, progress, None,
    )
    assert progress["done"] == len(rows)
