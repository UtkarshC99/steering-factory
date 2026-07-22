"""Batching correctness for finetune._generate_batched_rows -- the loop
`evaluate_qlora_adapter` uses to generate on validation/test examples with
a trained adapter.

Tested against a plain (unquantized, no PEFT) tiny Llama model rather than
a real quantized+adapter-wrapped one: 4-bit quantization needs a real CUDA
GPU + bitsandbytes, which this environment doesn't have, but the
batching/left-padding/score-reduction logic under test here has nothing to
do with quantization or PEFT -- it operates on whatever `model.generate`
returns, identically either way. See tests/_tiny_model.py for why Llama
(not GPT-2) is the right architecture family to test against.
"""
import pytest

from steering_factory.finetune import _generate_batched_rows

from _tiny_model import build_loaded_model


@pytest.fixture(scope="module")
def loaded():
    return build_loaded_model(seed=17)


RAGGED_EXAMPLES = [
    {"id": "e1", "behavior_id": "domain_classification", "split": "test", "category": "a", "prompt": "w4 w5"},
    {"id": "e2", "behavior_id": "domain_classification", "split": "test", "category": "a", "prompt": "w4 w5 w6 w7"},
    {"id": "e3", "behavior_id": "domain_classification", "split": "test", "category": "b", "prompt": "w4"},
    {"id": "e4", "behavior_id": "domain_classification", "split": "validation", "category": "b", "prompt": "w4 w5 w6 w7 w8 w9"},
]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 16])
def test_batched_rows_match_sequential_output(loaded, batch_size):
    reference = _generate_batched_rows(loaded.model, loaded.tokenizer, RAGGED_EXAMPLES, max_new_tokens=4, batch_size=1)
    batched = _generate_batched_rows(loaded.model, loaded.tokenizer, RAGGED_EXAMPLES, max_new_tokens=4, batch_size=batch_size)
    assert [r["output"] for r in reference] == [r["output"] for r in batched]
    assert [r["tokens_generated"] for r in reference] == [r["tokens_generated"] for r in batched]


def test_row_order_and_ids_preserved_under_batching(loaded):
    rows = _generate_batched_rows(loaded.model, loaded.tokenizer, RAGGED_EXAMPLES, max_new_tokens=3, batch_size=2)
    assert [r["example_id"] for r in rows] == [e["id"] for e in RAGGED_EXAMPLES]


def test_batch_size_and_wall_time_recorded_per_row(loaded):
    rows = _generate_batched_rows(loaded.model, loaded.tokenizer, RAGGED_EXAMPLES, max_new_tokens=3, batch_size=3)
    # First chunk is 3 examples, second chunk (the remainder) is 1.
    assert rows[0]["batch_size"] == 3
    assert rows[1]["batch_size"] == 3
    assert rows[2]["batch_size"] == 3
    assert rows[3]["batch_size"] == 1


def test_latency_s_is_amortized_batch_wall_time(loaded):
    rows = _generate_batched_rows(loaded.model, loaded.tokenizer, RAGGED_EXAMPLES, max_new_tokens=3, batch_size=4)
    for row in rows:
        assert row["latency_s"] == pytest.approx(row["batch_wall_time_s"] / row["batch_size"])


def test_batch_size_one_latency_equals_measured_wall_time(loaded):
    # Preserves the pre-batching meaning of latency_s exactly when nobody
    # opts into batching (batch_size=1, the manifest's implicit old default).
    rows = _generate_batched_rows(loaded.model, loaded.tokenizer, RAGGED_EXAMPLES, max_new_tokens=3, batch_size=1)
    for row in rows:
        assert row["latency_s"] == row["batch_wall_time_s"]


def test_tokenizer_padding_side_restored(loaded):
    original = loaded.tokenizer.padding_side
    _generate_batched_rows(loaded.model, loaded.tokenizer, RAGGED_EXAMPLES, max_new_tokens=2, batch_size=4)
    assert loaded.tokenizer.padding_side == original


def test_empty_examples_returns_empty_rows(loaded):
    assert _generate_batched_rows(loaded.model, loaded.tokenizer, [], max_new_tokens=3, batch_size=4) == []
