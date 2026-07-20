import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from steering_factory.config import load_config, AppConfig
from steering_factory.storage import VectorStore, _slugify
from steering_factory.examples_store import ExamplesStore


def test_load_config_with_no_file_returns_defaults():
    cfg = load_config(path="/nonexistent/path/config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.target_model.dtype in ("bfloat16", "float16", "float32")
    assert cfg.generator.backend in ("anthropic", "openai_compat", "local_hf")


def test_load_config_from_example_yaml():
    example_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "config.example.yaml")
    cfg = load_config(path=example_path)
    assert cfg.defaults.num_pairs_per_topic == 24
    assert cfg.generator.backend == "anthropic"


def test_slugify_basic():
    assert _slugify("Terse, No-Fluff Technical Answers!") == "terse_no_fluff_technical_answers"


def test_vector_store_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(tmp)
        vec = torch.randn(16)
        record = store.save(
            topic="test topic", model_name="fake-model", layer_idx=5,
            method="mean_diff", pooling="last", num_pairs=10, vector=vec,
            best_positive_coefficient=2.0, best_negative_coefficient=-1.5,
        )
        loaded = store.load_vector(record.id)
        assert torch.allclose(loaded, vec)

        listed = store.list()
        assert len(listed) == 1
        assert listed[0]["topic"] == "test topic"
        assert listed[0]["best_positive_coefficient"] == 2.0


def test_vector_store_delete():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(tmp)
        vec = torch.randn(8)
        record = store.save(
            topic="to delete", model_name="m", layer_idx=0, method="mean_diff",
            pooling="last", num_pairs=4, vector=vec,
        )
        assert store.get(record.id) is not None
        store.delete(record.id)
        assert store.get(record.id) is None
        assert not os.path.exists(os.path.join(tmp, record.vector_file))


def test_vector_store_topic_filter():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(tmp)
        store.save(topic="refusal of legal advice", model_name="m", layer_idx=0,
                    method="mean_diff", pooling="last", num_pairs=4, vector=torch.randn(4))
        store.save(topic="terse tone", model_name="m", layer_idx=0,
                    method="mean_diff", pooling="last", num_pairs=4, vector=torch.randn(4))
        results = store.list(topic_filter="legal")
        assert len(results) == 1
        assert "legal" in results[0]["topic"]


def test_examples_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = ExamplesStore(tmp)
        pairs = [{"prompt": "p1", "compliant": "c1", "non_compliant": "n1"}]
        store.save("my topic", pairs, persona="curt", generator_backend="anthropic")
        loaded = store.load("my topic")
        assert loaded["pairs"] == pairs
        assert loaded["persona"] == "curt"
        assert "my topic" in store.list_topics()
