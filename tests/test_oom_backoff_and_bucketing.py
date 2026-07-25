"""Tests for the two mechanisms that make generation self-stabilizing
instead of dependent on a perfectly-tuned static batch_size:

  - sweep.length_bucketed_chunks -- groups similar-length prompts so a
    chunk's memory cost tracks real content instead of its longest member.
  - sweep.run_with_oom_backoff -- halves and retries on CUDA OOM, so
    batch_size is a starting point and a hard chunk costs throughput
    rather than killing a multi-hour run.

Both are pure cost/scheduling mechanisms: they change WHICH rows share a
forward pass, never a generated token. The equivalence assertions are the
load-bearing ones.

The backoff tests raise torch.cuda.OutOfMemoryError directly rather than
actually exhausting a GPU -- the retry/split/ordering logic is plain
control flow and is exactly what needs pinning; genuinely OOMing a real
card is neither reproducible nor available in this environment.
"""
import pytest
import torch

from steering_factory.sweep import length_bucketed_chunks, run_with_oom_backoff

from _tiny_model import build_loaded_model


@pytest.fixture(scope="module")
def tokenizer():
    return build_loaded_model(seed=5).tokenizer


# --- length_bucketed_chunks ---------------------------------------------------

def _texts(lengths):
    return [" ".join(f"w{4 + (i % 40)}" for i in range(n)) for n in lengths]


def test_chunks_cover_every_index_exactly_once(tokenizer):
    texts = _texts([1, 9, 2, 7, 3, 8, 4, 6])
    chunks = length_bucketed_chunks(tokenizer, texts, 3)
    flat = [i for chunk in chunks for i in chunk]
    assert sorted(flat) == list(range(len(texts)))
    assert len(flat) == len(set(flat))  # no duplicates


def test_chunks_respect_the_size_cap(tokenizer):
    texts = _texts([1, 9, 2, 7, 3, 8, 4, 6])
    for size in (1, 2, 3, 5, 8):
        chunks = length_bucketed_chunks(tokenizer, texts, size)
        assert all(len(c) <= size for c in chunks)


def test_chunks_group_similar_lengths(tokenizer):
    """The real invariant is sorted-contiguity: chunks partition the
    prompts in non-decreasing length order, so each chunk holds
    neighbours. It is NOT "every chunk spans a small absolute range" --
    with lengths [1,2,3,18,19,20] the middle chunk legitimately straddles
    the 3->18 gap because the data itself has a gap there. What matters
    for padding waste is that no chunk mixes a prompt with one far from it
    in the sorted order while a closer neighbour was available."""
    lengths = [1, 20, 2, 19, 3, 18]
    texts = _texts(lengths)
    chunks = length_bucketed_chunks(tokenizer, texts, 2)

    flat_lengths = [lengths[i] for chunk in chunks for i in chunk]
    assert flat_lengths == sorted(lengths)

    # And the shortest/longest genuinely end up at opposite ends.
    assert min(lengths) in [lengths[i] for i in chunks[0]]
    assert max(lengths) in [lengths[i] for i in chunks[-1]]


def test_bucketing_reduces_padding_versus_naive_order(tokenizer):
    """Concrete statement of the benefit: padded width summed over chunks
    (what actually drives KV-cache/attention cost) must be no worse than
    chunking in the prompts' original order, and strictly better when
    lengths are interleaved -- which is the JSONSchemaBench shape that
    made a static batch_size unpredictable."""
    lengths = [1, 20, 2, 19, 3, 18]
    texts = _texts(lengths)
    size = 2

    def padded_cost(index_chunks):
        # every row in a chunk is padded out to the chunk's longest row
        return sum(max(lengths[i] for i in chunk) * len(chunk) for chunk in index_chunks)

    naive = [list(range(s, min(s + size, len(texts)))) for s in range(0, len(texts), size)]
    bucketed = length_bucketed_chunks(tokenizer, texts, size)

    assert padded_cost(bucketed) < padded_cost(naive)


def test_empty_input_returns_no_chunks(tokenizer):
    assert length_bucketed_chunks(tokenizer, [], 4) == []


def test_zero_or_none_chunk_size_falls_back_to_one_chunk(tokenizer):
    texts = _texts([1, 5, 3])
    for bad in (0, None):
        chunks = length_bucketed_chunks(tokenizer, texts, bad)
        assert len(chunks) == 1
        assert sorted(chunks[0]) == [0, 1, 2]


# --- run_with_oom_backoff -----------------------------------------------------

def test_returns_results_in_input_order_without_oom():
    calls = []

    def gen(batch):
        calls.append(list(batch))
        return [x * 10 for x in batch]

    assert run_with_oom_backoff(gen, [1, 2, 3, 4]) == [10, 20, 30, 40]
    assert calls == [[1, 2, 3, 4]]  # no split needed


def test_halves_on_oom_and_preserves_order():
    seen = []

    def gen(batch):
        seen.append(len(batch))
        if len(batch) > 2:          # anything bigger than 2 "doesn't fit"
            raise torch.cuda.OutOfMemoryError("simulated")
        return [x * 10 for x in batch]

    result = run_with_oom_backoff(gen, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == [10, 20, 30, 40, 50, 60, 70, 80]   # ORDER preserved
    assert 8 in seen and max(s for s in seen if s <= 2) == 2


def test_splits_repeatedly_until_it_fits():
    def gen(batch):
        if len(batch) > 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return [batch[0]]

    assert run_with_oom_backoff(gen, list(range(8))) == list(range(8))


def test_reraises_when_even_a_single_row_does_not_fit():
    def gen(batch):
        raise torch.cuda.OutOfMemoryError("simulated")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_with_oom_backoff(gen, [1, 2])


def test_non_oom_errors_are_not_swallowed():
    # A bug in generation must surface immediately, not be retried
    # forever at smaller batch sizes.
    def gen(batch):
        raise ValueError("a real bug")

    with pytest.raises(ValueError, match="a real bug"):
        run_with_oom_backoff(gen, [1, 2, 3, 4])


def test_empty_input_returns_empty():
    assert run_with_oom_backoff(lambda b: [], []) == []


def test_on_backoff_callback_is_notified():
    events = []

    def gen(batch):
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return list(batch)

    run_with_oom_backoff(gen, [1, 2, 3, 4], on_backoff=lambda old, new: events.append((old, new)))
    assert events and events[0] == (4, 2)


def test_a_raising_on_backoff_callback_never_breaks_the_run():
    def gen(batch):
        if len(batch) > 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return list(batch)

    def bad_callback(old, new):
        raise RuntimeError("callback bug")

    assert run_with_oom_backoff(gen, [1, 2], on_backoff=bad_callback) == [1, 2]
