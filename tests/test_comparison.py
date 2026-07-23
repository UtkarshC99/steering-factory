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


# N_PER_SPLIT clears comparison.DEFAULT_MIN_SPLIT_SIZE (20) so the main
# join/report tests exercise a "real" (non-excluded) comparison by default.
# Dedicated small-n fixtures below test the exclusion path itself.
N_PER_SPLIT = 24


def _steering_rows(model_id="m1", recipe_id="r1", behavior_id="domain_classification", n=N_PER_SPLIT, include_baseline=True):
    rows = []
    # Two steered configs on validation: method A layer 5 c=1 is better than method B layer 8 c=2.
    for i in range(n):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"val-a-{i}", "exact_match": 1.0 if i < n - 1 else 0.0,
                     "latency_s": 0.1, "tokens_generated": 20})
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "method": "pca", "layer_idx": 8, "coefficient": 2.0,
                     "token_scope": "all", "example_id": f"val-b-{i}", "exact_match": 0.0,
                     "latency_s": 0.1, "tokens_generated": 20})
    # Held-out test rows only for the winning config (mean_diff/L5/c=1).
    for i in range(n):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"test-a-{i}", "exact_match": 1.0,
                     "latency_s": 0.1, "tokens_generated": 20})
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "pca", "layer_idx": 8, "coefficient": 2.0,
                     "token_scope": "all", "example_id": f"test-b-{i}", "exact_match": 0.0,
                     "latency_s": 0.1, "tokens_generated": 20})
    if include_baseline:
        # The unsteered baseline (coefficient=0.0): scores worse than the
        # winning steered config on both splits, so beat_baseline is True
        # by default -- see test_best_steering_config_excludes_baseline_*
        # below for the case where the "winner" does NOT beat it.
        for i in range(n):
            rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                         "split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 0.0,
                         "token_scope": "all", "example_id": f"val-base-{i}", "exact_match": 0.0,
                         "latency_s": 0.1, "tokens_generated": 20})
            rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                         "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 0.0,
                         "token_scope": "all", "example_id": f"test-base-{i}", "exact_match": 0.0,
                         "latency_s": 0.1, "tokens_generated": 20})
    return rows


def _qlora_rows(model_id="m1", recipe_id="r1", behavior_id="domain_classification", test_quality=0.5, n=N_PER_SPLIT, num_train_records=200):
    # num_train_records is tagged on every row (matching real run_qlora
    # output -- see runner.py's eval_rows construction) because
    # data_efficiency_curve/build_comparison filter QLoRA rows by it to
    # distinguish multiple N-sweep adapters' eval rows apart.
    rows = []
    for i in range(n):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "example_id": f"qval-{i}", "exact_match": 1.0,
                     "latency_s": 0.2, "tokens_generated": 20, "num_train_records": num_train_records})
    for i in range(n):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "example_id": f"qtest-{i}", "exact_match": test_quality,
                     "latency_s": 0.2, "tokens_generated": 20, "num_train_records": num_train_records})
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
    assert result["validation_quality"] == pytest.approx((N_PER_SPLIT - 1) / N_PER_SPLIT)
    assert result["test_quality"] == 1.0  # only the winning config's held-out rows count
    assert result["baseline_quality"] == 0.0  # the c=0.0 rows included by default
    assert result["beat_baseline"] is True


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
    assert comparison["excluded"] == []
    assert len(comparison["comparisons"]) == 1
    entry = comparison["comparisons"][0]
    assert entry["model_id"] == "m1"
    assert entry["recipe_id"] == "r1"
    assert entry["steering"]["test_quality"] == 1.0
    assert entry["steering"]["beat_baseline"] is True
    assert entry["qlora"]["test_quality"] == 0.5
    assert entry["steering"]["labeled_examples"] == 16
    assert entry["qlora"]["labeled_examples"] == 200
    # Full sweep telemetry (12.5s over 2 configs) amortized to the selected config's share.
    assert entry["steering"]["selected_config_extraction_cost_s"] == pytest.approx(12.5 / 2)
    assert entry["steering"]["full_sweep_wall_time_s"] == 12.5
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


# --- Part 1a: n-floor guard -------------------------------------------------

def test_best_steering_config_excludes_coefficient_zero_from_selection():
    # Every real steered config scores 0; only the unsteered baseline (c=0.0)
    # scores 1 -- this is exactly what happened on the real Qwen3 run.
    # best_steering_config must NOT report c=0.0 as "the winning config".
    rows = []
    for i in range(5):
        rows.append({"split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"val-{i}", "exact_match": 0.0})
        rows.append({"split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 0.0,
                     "token_scope": "all", "example_id": f"val-base-{i}", "exact_match": 1.0})
        rows.append({"split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"test-{i}", "exact_match": 0.0})
        rows.append({"split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 0.0,
                     "token_scope": "all", "example_id": f"test-base-{i}", "exact_match": 1.0})
    result = best_steering_config(rows, "domain_classification")
    assert result is not None
    assert result["coefficient"] == 1.0  # never the baseline, even though it "won"
    assert result["test_quality"] == 0.0
    assert result["baseline_quality"] == 1.0
    assert result["beat_baseline"] is False


def test_build_comparison_excludes_entries_below_min_split_size(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    # n=1 per split -- exactly the shape of the real degenerate run.
    _write_jsonl(steer_dir / "results" / "generations.jsonl", _steering_rows(n=1, include_baseline=False))
    _write_jsonl(steer_dir / "vectors" / "index.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 5, "num_pairs": 1,
         "vector_path": str(steer_dir / "vectors" / "v1.pt")},
        {"model_id": "m1", "recipe_id": "r1", "method": "pca", "layer_idx": 8, "num_pairs": 1,
         "vector_path": str(steer_dir / "vectors" / "v2.pt")},
    ])
    (steer_dir / "vectors" / "v1.pt").write_bytes(b"0")
    (steer_dir / "vectors" / "v2.pt").write_bytes(b"0")
    _write_json(steer_dir / "telemetry.json", {"wall_time_s": 1.0})
    _write_jsonl(qlora_dir / "results" / "generations.jsonl", _qlora_rows(n=1))
    _write_json(qlora_dir / "results" / "qlora.json", [
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 1, "wall_time_s": 1.0,
         "adapter_size_bytes": 10, "adapter_dir": "x"},
    ])

    comparison = build_comparison(steer_dir, qlora_dir)  # default min_split_size=20
    assert comparison["comparisons"] == []
    assert len(comparison["excluded"]) == 1
    excluded = comparison["excluded"][0]
    assert excluded["reason"] == "insufficient_data"
    assert excluded["n_test_steering"] == 1

    # The same data, with the floor disabled, DOES produce a (noisy) entry --
    # confirms the floor is what's gating it, not some other join failure.
    unfiltered = build_comparison(steer_dir, qlora_dir, min_split_size=0)
    assert len(unfiltered["comparisons"]) == 1


def test_render_comparison_markdown_separates_excluded_from_winner_table():
    comparison = {
        "steering_run": "s", "qlora_run": "q", "min_split_size": 20,
        "comparisons": [], "excluded": [{
            "model_id": "m1", "recipe_id": "r1", "reason": "insufficient_data",
            "n_test_steering": 1, "n_validation_steering": 1, "n_test_qlora": 1, "n_validation_qlora": 1,
        }],
    }
    markdown = render_comparison_markdown(comparison)
    assert "Excluded (insufficient data)" in markdown
    assert "winner" not in markdown.lower().split("excluded")[0] or "Matched quality" not in markdown


def test_render_comparison_markdown_reports_inconclusive_when_steering_loses_to_baseline():
    entry_rows = _steering_rows()
    # Force the winning steered config to lose to baseline by zeroing its test rows.
    for row in entry_rows:
        if row["method"] == "mean_diff" and row["coefficient"] == 1.0 and row["split"] == "test":
            row["exact_match"] = 0.0
    steer_quality = best_steering_config(entry_rows, "domain_classification")
    assert steer_quality["beat_baseline"] is False

    comparison = {
        "steering_run": "s", "qlora_run": "q", "min_split_size": 20,
        "comparisons": [{
            "model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
            "steering": {**steer_quality, "selected_config_extraction_cost_s": 1.0, "artifact_bytes": 1,
                          "labeled_examples": 1, "per_request_ms_per_token": 1.0},
            "qlora": {"test_quality": 0.9, "wall_time_s": 1.0, "artifact_bytes": 1,
                      "labeled_examples": 1, "per_request_ms_per_token": 1.0, "n_test": 24},
        }],
        "excluded": [],
    }
    markdown = render_comparison_markdown(comparison)
    assert "inconclusive" in markdown.lower()


def test_degenerate_real_run_produces_zero_winner_rows(tmp_path):
    """Acceptance test: reconstructs the actual shape of the degenerate
    real-GPU comparison report (gemma3_4b + qwen3_4b, safety_refusal,
    n_test=n_validation=1, Qwen3's validation-selected config is the
    coefficient=0.0 baseline). Re-run through the fixed code, this must
    produce zero rows in `comparisons` -- everything routes to `excluded`."""
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"

    def make_pair_rows(model_id):
        rows = []
        # 12 configs evaluated (matches num_configs_evaluated: 12 in the
        # real report), each with exactly 1 validation + 1 test row.
        configs = [("mean_diff", 12, c, "all") for c in (-2, -1, -0.5, 0.0, 0.5, 1, 2)]
        configs += [("pca", 12, 1.0, "all"), ("whitened_mean_diff", 12, 1.0, "all"),
                    ("optimized", 12, 1.0, "all"), ("mean_diff", 6, 1.0, "all"), ("mean_diff", 18, 1.0, "all")]
        for i, (method, layer_idx, coeff, scope) in enumerate(configs):
            # Only the c=0.0 baseline "succeeds" for qwen3 -- the real bug.
            quality = 1.0 if (model_id == "qwen3_4b" and coeff == 0.0) else 0.0
            rows.append({"model_id": model_id, "recipe_id": "safety_refusal", "behavior_id": "defensive_refusal",
                         "split": "validation", "method": method, "layer_idx": layer_idx, "coefficient": coeff,
                         "token_scope": scope, "example_id": f"{model_id}-val-{i}", "safe_refusal": quality,
                         "latency_s": 90.0, "tokens_generated": 96})
            rows.append({"model_id": model_id, "recipe_id": "safety_refusal", "behavior_id": "defensive_refusal",
                         "split": "test", "method": method, "layer_idx": layer_idx, "coefficient": coeff,
                         "token_scope": scope, "example_id": f"{model_id}-test-{i}", "safe_refusal": 0.0,
                         "latency_s": 90.0, "tokens_generated": 96})
        return rows

    steer_rows, vector_rows = [], []
    for model_id in ("gemma3_4b", "qwen3_4b"):
        steer_rows += make_pair_rows(model_id)
        for i, (method, layer_idx) in enumerate([("mean_diff", 12), ("pca", 12), ("whitened_mean_diff", 12),
                                                  ("optimized", 12), ("mean_diff", 6), ("mean_diff", 18)]):
            path = steer_dir / "vectors" / f"{model_id}-{i}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"0" * 1000)
            vector_rows.append({"model_id": model_id, "recipe_id": "safety_refusal", "method": method,
                                 "layer_idx": layer_idx, "num_pairs": 1, "vector_path": str(path)})
    _write_jsonl(steer_dir / "results" / "generations.jsonl", steer_rows)
    _write_jsonl(steer_dir / "vectors" / "index.jsonl", vector_rows)
    _write_json(steer_dir / "telemetry.json", {"wall_time_s": 7247.44})

    qlora_rows, qlora_results = [], []
    for model_id in ("gemma3_4b", "qwen3_4b"):
        qlora_rows.append({"model_id": model_id, "recipe_id": "safety_refusal", "behavior_id": "defensive_refusal",
                            "split": "validation", "example_id": f"{model_id}-qval", "safe_refusal": 1.0,
                            "latency_s": 150.0, "tokens_generated": 96})
        qlora_rows.append({"model_id": model_id, "recipe_id": "safety_refusal", "behavior_id": "defensive_refusal",
                            "split": "test", "example_id": f"{model_id}-qtest", "safe_refusal": 1.0,
                            "latency_s": 150.0, "tokens_generated": 96})
        qlora_results.append({"model_id": model_id, "recipe_id": "safety_refusal", "num_train_records": 1,
                               "wall_time_s": 150.0, "eval_wall_time_s": 30.0, "adapter_size_bytes": 70_000_000,
                               "adapter_dir": "x"})
    _write_jsonl(qlora_dir / "results" / "generations.jsonl", qlora_rows)
    _write_json(qlora_dir / "results" / "qlora.json", qlora_results)

    comparison = build_comparison(steer_dir, qlora_dir)  # default floor, exactly as `compare` would call it
    assert comparison["comparisons"] == [], (
        "the degenerate n=1 real-run shape must produce zero winner rows under the honesty fix"
    )
    assert len(comparison["excluded"]) == 2
    for excluded in comparison["excluded"]:
        assert excluded["n_test_steering"] == 1
        assert excluded["n_test_qlora"] == 1

    markdown = render_comparison_markdown(comparison)
    assert "Matched quality" not in markdown  # no winner section rendered at all
    assert "Excluded (insufficient data)" in markdown


# --- N-sweep: build_comparison must pin to max-N, not pool across N -----------

def _multi_n_steering_run(root: Path):
    """Two vectors for the same (method=mean_diff, layer=5) at different
    N -- N=8 (worse quality) and N=24 (better, above the min-split floor)
    -- exactly what an experiment.n_sweep run produces. build_comparison's
    winner table must use ONLY the N=24 rows/vector, not pool both N's
    generation rows together as if they were one config."""
    small_n_rows, large_n_rows = [], []
    for i in range(N_PER_SPLIT):
        # N=8 vector: worse quality (simulates less labeled data -> worse steering).
        small_n_rows.append({"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
                              "split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                              "token_scope": "all", "example_id": f"n8-val-{i}", "exact_match": 0.0})
        small_n_rows.append({"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
                              "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                              "token_scope": "all", "example_id": f"n8-test-{i}", "exact_match": 0.0})
        # N=24 vector: better quality.
        large_n_rows.append({"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
                              "split": "validation", "method": "mean_diff", "layer_idx": 8, "coefficient": 1.0,
                              "token_scope": "all", "example_id": f"n24-val-{i}", "exact_match": 1.0})
        large_n_rows.append({"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
                              "split": "test", "method": "mean_diff", "layer_idx": 8, "coefficient": 1.0,
                              "token_scope": "all", "example_id": f"n24-test-{i}", "exact_match": 1.0})
    _write_jsonl(root / "results" / "generations.jsonl", small_n_rows + large_n_rows)
    _write_jsonl(root / "vectors" / "index.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 5, "num_pairs": 8,
         "vector_path": str(root / "vectors" / "n8.pt")},
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 8, "num_pairs": 24,
         "vector_path": str(root / "vectors" / "n24.pt")},
    ])
    (root / "vectors" / "n8.pt").write_bytes(b"0" * 50)
    (root / "vectors" / "n24.pt").write_bytes(b"0" * 100)
    _write_json(root / "telemetry.json", {"wall_time_s": 20.0})
    _write_json(root / "run.json", {"run_id": "steer-nsweep", "status": "completed", "manifest_hash": "abc"})


def _multi_n_qlora_run(root: Path):
    small_n_rows = _qlora_rows(test_quality=0.3, num_train_records=8)
    large_n_rows = _qlora_rows(test_quality=0.6, num_train_records=200)
    for row in small_n_rows:
        row["example_id"] = "n8-" + row["example_id"]
    for row in large_n_rows:
        row["example_id"] = "n200-" + row["example_id"]
    _write_jsonl(root / "results" / "generations.jsonl", small_n_rows + large_n_rows)
    _write_json(root / "results" / "qlora.json", [
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 8, "wall_time_s": 10.0,
         "eval_wall_time_s": 1.0, "adapter_size_bytes": 100, "adapter_dir": "x8"},
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 200, "wall_time_s": 90.0,
         "eval_wall_time_s": 5.0, "adapter_size_bytes": 4096, "adapter_dir": "x200"},
    ])
    _write_json(root / "run.json", {"run_id": "qlora-nsweep", "status": "completed", "manifest_hash": "abc"})


def test_build_comparison_pins_winner_table_to_max_n_not_pooled_across_n(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _multi_n_steering_run(steer_dir)
    _multi_n_qlora_run(qlora_dir)

    comparison = build_comparison(steer_dir, qlora_dir)
    assert len(comparison["comparisons"]) == 1
    entry = comparison["comparisons"][0]

    # Must reflect the N=24 (better) config, not a blend of N=8 and N=24 --
    # pooling both would give exact_match=0.5 (worse than either alone,
    # correctness-wise the wrong number entirely) instead of 1.0.
    assert entry["steering"]["test_quality"] == 1.0
    assert entry["steering"]["layer_idx"] == 8
    assert entry["steering"]["labeled_examples"] == 24
    assert entry["qlora"]["test_quality"] == pytest.approx(0.6)
    assert entry["qlora"]["labeled_examples"] == 200

    # config_grid still reports only the max-N vector's own configs (L8),
    # not the N=8 vector's L5 config mixed in as if comparable.
    assert {row["layer_idx"] for row in entry["config_grid"]} == {8}


def test_data_efficiency_curve_reports_both_n_points_for_each_arm(tmp_path):
    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _multi_n_steering_run(steer_dir)
    _multi_n_qlora_run(qlora_dir)

    curve = data_efficiency_curve(steer_dir, qlora_dir, "m1", "r1")
    steer_ns = sorted(p["labeled_examples"] for p in curve["steering"])
    qlora_ns = sorted(p["labeled_examples"] for p in curve["qlora"])
    assert steer_ns == [8, 24]
    assert qlora_ns == [8, 200]
    # Each qlora point's quality must reflect only ITS OWN adapter's eval
    # rows, not rows from the other N -- this is the num_train_records
    # filter fix; without it, both points would report the same
    # (incorrectly pooled) quality.
    by_n = {p["labeled_examples"]: p["quality"] for p in curve["qlora"]}
    assert by_n[8] == pytest.approx(0.3)
    assert by_n[200] == pytest.approx(0.6)
