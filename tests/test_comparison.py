import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from steering_factory.comparison import best_steering_config, build_comparison, data_efficiency_curve, qlora_quality
from steering_factory.comparison_report import render_comparison_markdown, write_comparison_report
from steering_factory.runner import _is_qlora_run, _is_steering_run, compare


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _steering_rows(model_id="m1", recipe_id="r1", behavior_id="domain_classification"):
    rows = []
    # Two configs on validation: method A layer 5 c=1 is better than method B layer 8 c=2.
    for i in range(4):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"val-a-{i}", "exact_match": 1.0 if i < 3 else 0.0,
                     "latency_s": 0.1, "tokens_generated": 20})
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "method": "pca", "layer_idx": 8, "coefficient": 2.0,
                     "token_scope": "all", "example_id": f"val-b-{i}", "exact_match": 0.0,
                     "latency_s": 0.1, "tokens_generated": 20})
    # Held-out test rows only for the winning config (mean_diff/L5/c=1).
    for i in range(3):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"test-a-{i}", "exact_match": 1.0,
                     "latency_s": 0.1, "tokens_generated": 20})
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "pca", "layer_idx": 8, "coefficient": 2.0,
                     "token_scope": "all", "example_id": f"test-b-{i}", "exact_match": 0.0,
                     "latency_s": 0.1, "tokens_generated": 20})
    return rows


def _qlora_rows(model_id="m1", recipe_id="r1", behavior_id="domain_classification", test_quality=0.5):
    rows = []
    for i in range(4):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "example_id": f"qval-{i}", "exact_match": 1.0,
                     "latency_s": 0.2, "tokens_generated": 20})
    for i in range(3):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "example_id": f"qtest-{i}", "exact_match": test_quality,
                     "latency_s": 0.2, "tokens_generated": 20})
    return rows


def _build_steering_run(root: Path):
    _write_jsonl(root / "results" / "generations.jsonl", _steering_rows())
    _write_jsonl(root / "vectors" / "index.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 5, "num_pairs": 16,
         "vector_path": str(root / "vectors" / "v1.pt")},
        {"model_id": "m1", "recipe_id": "r1", "method": "pca", "layer_idx": 8, "num_pairs": 16,
         "vector_path": str(root / "vectors" / "v2.pt")},
    ])
    (root / "vectors" / "v1.pt").write_bytes(b"0" * 100)
    (root / "vectors" / "v2.pt").write_bytes(b"0" * 100)
    _write_json(root / "telemetry.json", {"wall_time_s": 12.5, "gpu_peak_allocated_bytes": 1024, "gpu_peak_reserved_bytes": 2048})
    _write_json(root / "run.json", {"run_id": "steer-1", "status": "completed", "manifest_hash": "abc"})


def _build_qlora_run(root: Path, test_quality=0.5):
    _write_jsonl(root / "results" / "generations.jsonl", _qlora_rows(test_quality=test_quality))
    _write_json(root / "results" / "qlora.json", [
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 200, "wall_time_s": 90.0,
         "eval_wall_time_s": 5.0, "adapter_size_bytes": 4096, "adapter_dir": str(root / "qlora" / "m1" / "r1"),
         "train_loss": 0.3},
    ])
    _write_json(root / "run.json", {"run_id": "qlora-1", "status": "completed", "manifest_hash": "abc"})


def test_best_steering_config_selects_on_validation_reports_held_out_test(tmp_path):
    rows = _steering_rows()
    result = best_steering_config(rows, "domain_classification")
    assert result is not None
    assert result["method"] == "mean_diff"
    assert result["layer_idx"] == 5
    assert result["validation_quality"] == 0.75
    assert result["test_quality"] == 1.0  # only the winning config's held-out rows count


def test_best_steering_config_returns_none_without_validation_rows():
    rows = [r for r in _steering_rows() if r["split"] != "validation"]
    assert best_steering_config(rows, "domain_classification") is None


def test_qlora_quality_reports_validation_and_test():
    rows = _qlora_rows(test_quality=0.66)
    result = qlora_quality(rows, "domain_classification")
    assert result["validation_quality"] == 1.0
    assert result["test_quality"] == pytest.approx(0.66)


def test_build_comparison_joins_on_model_and_recipe(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)

    comparison = build_comparison(steer_dir, qlora_dir)
    assert len(comparison["comparisons"]) == 1
    entry = comparison["comparisons"][0]
    assert entry["model_id"] == "m1"
    assert entry["recipe_id"] == "r1"
    assert entry["steering"]["test_quality"] == 1.0
    assert entry["qlora"]["test_quality"] == 0.5
    assert entry["steering"]["labeled_examples"] == 16
    assert entry["qlora"]["labeled_examples"] == 200
    assert entry["steering"]["wall_time_s"] == 12.5
    assert entry["qlora"]["train_wall_time_s"] == 90.0


def test_build_comparison_empty_when_no_matching_model_recipe(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir)
    # Overwrite qlora rows with a different recipe_id so nothing joins.
    _write_jsonl(qlora_dir / "results" / "generations.jsonl",
                 [{**r, "recipe_id": "other_recipe"} for r in _qlora_rows()])

    comparison = build_comparison(steer_dir, qlora_dir)
    assert comparison["comparisons"] == []


def test_render_comparison_markdown_declares_a_winner(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)
    comparison = build_comparison(steer_dir, qlora_dir)

    markdown = render_comparison_markdown(comparison)
    assert "steering" in markdown
    assert "m1" in markdown and "r1" in markdown


def test_write_comparison_report_writes_json_and_markdown(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir)
    comparison = build_comparison(steer_dir, qlora_dir)

    output_root = tmp_path / "comparisons"
    markdown_path = write_comparison_report(comparison, output_root)
    assert markdown_path.exists()
    assert (output_root / "report.json").exists()


def test_data_efficiency_curve_reports_points_per_arm(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir)

    curve = data_efficiency_curve(steer_dir, qlora_dir, "m1", "r1")
    assert curve["quality_key"] is not None
    assert len(curve["steering"]) >= 1
    assert len(curve["qlora"]) == 1
    assert curve["qlora"][0]["labeled_examples"] == 200


def test_compare_detects_steering_and_qlora_runs_and_writes_matched_report(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)

    assert _is_steering_run(steer_dir) and not _is_qlora_run(steer_dir)
    assert _is_qlora_run(qlora_dir)

    output_root = tmp_path / "out"
    result_path = compare([steer_dir, qlora_dir], output_root)
    assert result_path.name == "report.md"
    payload = json.loads((output_root / "report.json").read_text())
    assert len(payload["comparisons"]) == 1


def test_compare_falls_back_to_metadata_index_for_two_steering_runs(tmp_path):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    _build_steering_run(a)
    _build_steering_run(b)

    output_root = tmp_path / "out"
    result_path = compare([a, b], output_root)
    assert result_path.name == "comparison.json"
    payload = json.loads(result_path.read_text())
    assert len(payload["runs"]) == 2
