"""Verifies `extraction.extract`'s callback dispatch without loading a real
model: `callback` must reach `optimized_vector` (the one method with a real
per-step loss) but must NOT be forwarded to the closed-form methods, which
don't accept it and would TypeError if it were passed through blindly.
"""
from unittest.mock import patch

from steering_factory import extraction


def test_callback_not_forwarded_to_mean_diff():
    with patch.object(extraction, "mean_diff_vector") as mocked:
        mocked.return_value = "result"
        result = extraction.extract("mean_diff", loaded=object(), layer_idx=0, pairs=[], callback=object())
        assert result == "result"
        _, kwargs = mocked.call_args
        assert "callback" not in kwargs


def test_callback_not_forwarded_to_pca():
    with patch.object(extraction, "pca_vector") as mocked:
        mocked.return_value = "result"
        extraction.extract("pca", loaded=object(), layer_idx=0, pairs=[], callback=object())
        _, kwargs = mocked.call_args
        assert "callback" not in kwargs


def test_callback_not_forwarded_to_whitened_mean_diff():
    with patch.object(extraction, "whitened_mean_diff_vector") as mocked:
        mocked.return_value = "result"
        extraction.extract("whitened_mean_diff", loaded=object(), layer_idx=0, pairs=[], callback=object())
        _, kwargs = mocked.call_args
        assert "callback" not in kwargs


def test_callback_forwarded_to_optimized():
    sentinel = object()
    with patch.object(extraction, "mean_diff_vector") as mean_diff_mock, \
         patch.object(extraction, "optimized_vector") as optimized_mock:
        mean_diff_mock.return_value.vector = "init-vector"
        optimized_mock.return_value = "result"
        result = extraction.extract("optimized", loaded=object(), layer_idx=0, pairs=[], callback=sentinel)
        assert result == "result"
        _, kwargs = optimized_mock.call_args
        assert kwargs["callback"] is sentinel


def test_extract_with_no_callback_still_works_for_every_method():
    for method, target in [
        ("mean_diff", "mean_diff_vector"),
        ("pca", "pca_vector"),
        ("whitened_mean_diff", "whitened_mean_diff_vector"),
    ]:
        with patch.object(extraction, target) as mocked:
            mocked.return_value = "result"
            assert extraction.extract(method, loaded=object(), layer_idx=0, pairs=[]) == "result"
