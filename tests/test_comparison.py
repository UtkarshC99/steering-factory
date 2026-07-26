import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from steering_factory.comparison import (
    HeldOutViolation,
    assert_disjoint_selection_and_test_rows,
    best_steering_config,
    build_comparison,
    data_efficiency_curve,
    qlora_quality,
)
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


def _read_generations(root: Path):
    path = root / "results" / "generations.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def _add_fluency_fields(rows, ppl_ratio=1.0, repetition=0.05):
    return [{**r, "perplexity_ratio_vs_baseline": ppl_ratio, "repetition_score": repetition} for r in rows]


def test_best_steering_config_rejects_a_degenerate_winner_for_fluency():
    """Regression test for a real run (2026-07-26): Qwen3-4B's SELECTED
    winning config had perplexity_ratio_vs_baseline=41.99 -- reported
    quality bought largely with broken text, because best_steering_config
    had no fluency gate at all. The winning config (mean_diff/L5/c=1) here
    is given a catastrophic perplexity ratio; the loser (pca/L8/c=2) is
    fluent, so the fluent loser must be selected instead."""
    rows = _steering_rows()
    for row in rows:
        if row["method"] == "mean_diff" and row["coefficient"] == 1.0:
            row["perplexity_ratio_vs_baseline"] = 42.0
            row["repetition_score"] = 0.1
        elif row["method"] == "pca" and row["coefficient"] == 2.0:
            row["perplexity_ratio_vs_baseline"] = 1.0
            row["repetition_score"] = 0.02

    result = best_steering_config(rows, "domain_classification")
    assert result is not None
    assert result["method"] == "pca"
    assert result["layer_idx"] == 8
    assert any(r["method"] == "mean_diff" for r in result["rejected_for_fluency"])


def test_best_steering_config_rejects_via_repetition_even_when_perplexity_passes():
    """A fluent-looking perplexity ratio can still be a repetition loop --
    a real run's Qwen3-4B c=-2.0 had perplexity_ratio=1.02 (near-normal)
    but repetition_score=0.63. Both checks must gate independently."""
    rows = _steering_rows()
    for row in rows:
        if row["method"] == "mean_diff" and row["coefficient"] == 1.0:
            row["perplexity_ratio_vs_baseline"] = 1.01   # passes perplexity
            row["repetition_score"] = 0.65                # fails repetition
        elif row["method"] == "pca" and row["coefficient"] == 2.0:
            row["perplexity_ratio_vs_baseline"] = 1.0
            row["repetition_score"] = 0.02

    result = best_steering_config(rows, "domain_classification")
    assert result["method"] == "pca"


def test_best_steering_config_returns_none_when_every_config_is_degenerate():
    rows = _add_fluency_fields(_steering_rows(), ppl_ratio=100.0, repetition=0.9)
    assert best_steering_config(rows, "domain_classification") is None


def test_best_steering_config_missing_fluency_fields_does_not_gate():
    # A behavior with no perplexity/repetition diagnostics (e.g.
    # multiple_choice_eval) must not be penalized for lacking them.
    rows = _steering_rows()
    result = best_steering_config(rows, "domain_classification")
    assert result is not None
    assert result["rejected_for_fluency"] == []


def test_best_steering_config_reports_distinct_output_ratio():
    rows = _steering_rows()
    for i, row in enumerate(r for r in rows if r["split"] == "test" and r["method"] == "mean_diff"):
        row["output"] = "same canned output"  # collapsed on purpose
    result = best_steering_config(rows, "domain_classification")
    assert result["distinct_output_ratio"] == pytest.approx(1.0 / N_PER_SPLIT)


def test_qlora_quality_reports_distinct_output_ratio():
    """Regression test for a real run (2026-07-26): a QLoRA adapter
    collapsed to the same canned refusal on 358/364 rows while still
    scoring safe_refusal=1.0 -- distinct_output_ratio is what makes that
    visible."""
    rows = _qlora_rows(test_quality=0.5)
    for row in rows:
        if row["split"] == "test":
            row["output"] = "I can't help with that."   # collapsed on purpose
    result = qlora_quality(rows, "domain_classification")
    assert result["distinct_output_ratio"] == pytest.approx(1.0 / N_PER_SPLIT)


def test_qlora_quality_distinct_output_ratio_is_none_without_output_field():
    # _qlora_rows' base fixture carries no "output" field at all --
    # distinct_output_ratio must be None (unmeasured), not 0.0 or 1.0.
    rows = _qlora_rows(test_quality=0.5)
    result = qlora_quality(rows, "domain_classification")
    assert result["distinct_output_ratio"] is None


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
    # This fixture's vectors carry no extraction_wall_time_s (predates that
    # instrumentation), so _steering_cost falls back to the legacy
    # whole-sweep-amortized figure: full sweep telemetry (12.5s over 2
    # configs) amortized to the selected config's share. See
    # test_steering_cost_prefers_the_selected_vectors_own_extraction_time
    # for the new, precise per-vector path.
    assert entry["steering"]["one_time_cost_s"] == pytest.approx(12.5 / 2)
    assert entry["steering"]["full_sweep_wall_time_s"] == 12.5
    assert entry["qlora"]["train_wall_time_s"] == 90.0
    # QLoRA's one-time cost is now TRAIN time only (2026-07-26), not
    # train+eval -- eval generation time (5.0s in this fixture) is
    # deliberately excluded; see the identical test below.
    assert entry["qlora"]["one_time_cost_s"] == pytest.approx(90.0)


def test_steering_cost_prefers_the_selected_vectors_own_extraction_time(tmp_path):
    """Regression test for Tier 2: one_time_cost_s must be the SELECTED
    vector's own measured extraction_wall_time_s, not the whole run's wall
    time amortized evenly across every config -- amortizing evenly assumes
    every extraction is equally expensive, which real runs show is false
    (different methods/layers/N cost differently)."""
    steer_dir, qlora_dir = tmp_path / "steer_run", tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)

    # Give the two vectors DIFFERENT extraction times -- the winning config
    # (mean_diff/L5) took 3.0s; the loser (pca/L8) took 40.0s. If cost were
    # still amortized evenly (12.5/2=6.25) or used the loser's time, this
    # test would catch it.
    _write_jsonl(steer_dir / "vectors" / "index.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 5, "num_pairs": 16,
         "vector_path": str(steer_dir / "vectors" / "v1.pt"), "extraction_wall_time_s": 3.0},
        {"model_id": "m1", "recipe_id": "r1", "method": "pca", "layer_idx": 8, "num_pairs": 16,
         "vector_path": str(steer_dir / "vectors" / "v2.pt"), "extraction_wall_time_s": 40.0},
    ])

    comparison = build_comparison(steer_dir, qlora_dir)
    entry = comparison["comparisons"][0]
    assert entry["steering"]["one_time_cost_s"] == pytest.approx(3.0)


def test_qlora_one_time_cost_excludes_eval_generation_time(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer_run", tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)
    _write_json(qlora_dir / "results" / "qlora.json", [
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 200, "wall_time_s": 90.0,
         "eval_wall_time_s": 500.0,  # deliberately huge, to prove it is NOT summed in
         "adapter_size_bytes": 4096, "adapter_dir": str(qlora_dir / "qlora" / "m1" / "r1"), "train_loss": 0.3},
    ])
    comparison = build_comparison(steer_dir, qlora_dir)
    entry = comparison["comparisons"][0]
    assert entry["qlora"]["one_time_cost_s"] == pytest.approx(90.0)
    assert entry["qlora"]["eval_wall_time_s"] == pytest.approx(500.0)  # kept on the row, just not folded in


def test_cost_by_n_reports_every_n_sweep_point_separately(tmp_path):
    """Regression test for the explicit ask: 'clearly delineate avg time
    for x1 samples | for x2 samples | ...'. Two N-sweep points (N=10 and
    N=40) with DIFFERENT costs must both appear, not just the max-N
    headline point."""
    steer_dir, qlora_dir = tmp_path / "steer_run", tmp_path / "qlora_run"

    _write_jsonl(steer_dir / "results" / "generations.jsonl", _steering_rows())
    _write_jsonl(steer_dir / "vectors" / "index.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 5, "num_pairs": 10,
         "vector_path": str(steer_dir / "vectors" / "v1_n10.pt"), "extraction_wall_time_s": 1.0},
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 5, "num_pairs": 40,
         "vector_path": str(steer_dir / "vectors" / "v1_n40.pt"), "extraction_wall_time_s": 5.0},
        {"model_id": "m1", "recipe_id": "r1", "method": "pca", "layer_idx": 8, "num_pairs": 40,
         "vector_path": str(steer_dir / "vectors" / "v2_n40.pt"), "extraction_wall_time_s": 7.0},
    ])
    for p in ("v1_n10.pt", "v1_n40.pt", "v2_n40.pt"):
        (steer_dir / "vectors" / p).write_bytes(b"0" * 10)
    _write_json(steer_dir / "telemetry.json", {"wall_time_s": 13.0})
    _write_json(steer_dir / "run.json", {"run_id": "s", "status": "completed", "manifest_hash": "abc"})

    _write_jsonl(qlora_dir / "results" / "generations.jsonl", _qlora_rows(test_quality=0.5, num_train_records=40))
    _write_json(qlora_dir / "results" / "qlora.json", [
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 10, "wall_time_s": 20.0,
         "eval_wall_time_s": 1.0, "adapter_size_bytes": 1, "adapter_dir": "x"},
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 40, "wall_time_s": 60.0,
         "eval_wall_time_s": 1.0, "adapter_size_bytes": 1, "adapter_dir": "y"},
    ])
    _write_json(qlora_dir / "run.json", {"run_id": "q", "status": "completed", "manifest_hash": "abc"})

    comparison = build_comparison(steer_dir, qlora_dir)
    entry = comparison["comparisons"][0]
    by_n = {row["n"]: row for row in entry["cost_by_n"]}
    assert set(by_n) == {10, 40}
    assert by_n[10]["steering_one_time_cost_s"] == pytest.approx(1.0)
    assert by_n[10]["steering_num_configs"] == 1
    assert by_n[10]["qlora_one_time_cost_s"] == pytest.approx(20.0)
    # N=40 has TWO configs (mean_diff + pca) -- averaged, not summed.
    assert by_n[40]["steering_one_time_cost_s"] == pytest.approx((5.0 + 7.0) / 2)
    assert by_n[40]["steering_num_configs"] == 2
    assert by_n[40]["qlora_one_time_cost_s"] == pytest.approx(60.0)


def test_run_qlora_writes_telemetry(tmp_path, monkeypatch):
    """run_qlora previously wrote no telemetry.json at all, unlike
    run_steering. Stubs finetune.train_qlora/evaluate_qlora_adapter (no
    real peft/trl/GPU needed) to confirm the file now exists with a real
    wall_time_s."""
    from steering_factory import finetune as finetune_module
    from steering_factory.runner import run_qlora

    def fake_train_qlora(records, config, callback=None, callback_context=None):
        return {"adapter_dir": "unused", "train_loss": 0.0, "global_step": 1,
                "wall_time_s": 0.01, "adapter_size_bytes": 0, "log_history": []}

    def fake_evaluate_qlora_adapter(examples, model_name, adapter_dir, max_new_tokens=96,
                                     trust_remote_code=False, batch_size=16,
                                     quantization="4bit", dtype=None, max_length=1024):
        rows = [{"example_id": e["id"], "behavior_id": e["behavior_id"], "split": e["split"],
                  "category": e.get("category"), "prompt": e["prompt"], "output": "x",
                  "latency_s": 0.01, "batch_size": 1, "batch_wall_time_s": 0.01, "tokens_generated": 1}
                 for e in examples]
        return {"rows": rows, "wall_time_s": 0.01}

    monkeypatch.setattr(finetune_module, "train_qlora", fake_train_qlora)
    monkeypatch.setattr(finetune_module, "evaluate_qlora_adapter", fake_evaluate_qlora_adapter)

    records = [{"id": f"e{i}", "prompt": f"p{i}", "positive": "pos", "negative": "neg", "category": "a"}
               for i in range(10)]
    manifest_path = tmp_path / "data.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")

    manifest = {
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "models": [{"id": "tiny", "name_or_path": "tiny/tiny-llama"}],
        "splits": {"seed": 3, "steer_fraction": 0.5, "validation_fraction": 0.25, "group_key": "category"},
        "recipes": [{"id": "r1", "behavior": "domain_classification",
                     "dataset": {"adapter": "local_jsonl", "path": str(manifest_path)}}],
        "finetune": {"backend": "qlora", "target_modules": ["q_proj"], "max_steps": 1},
    }
    store = run_qlora(manifest, command="test")
    telemetry_path = store.path / "telemetry.json"
    assert telemetry_path.exists()
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert telemetry["wall_time_s"] > 0
    assert "gpu_peak_allocated_bytes" in telemetry


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


def test_winner_is_inconclusive_when_leader_falsely_refuses_benign_prompts():
    """Regression test for the actual 2026-07-26 run: QLoRA was named the
    winner (safe_refusal=1.0 vs steering's 0.29-0.45) while falsely
    refusing 68-96% of benign XSTest prompts. The number that would have
    caught this (false_refusal_rate_on_benign_controls) was computed into
    safety_summary.json but never reached the winner decision -- this
    pins that it now does."""
    comparison = {
        "steering_run": "s", "qlora_run": "q", "min_split_size": 20,
        "comparisons": [{
            "model_id": "m1", "recipe_id": "r1", "behavior_id": "harmful_instruction_compliance",
            "steering": {"test_quality": 0.29, "beat_baseline": True, "distinct_output_ratio": 0.6,
                          "benign_control": {"false_refusal_rate_on_benign_controls": 0.003}},
            "qlora": {"test_quality": 1.0, "distinct_output_ratio": 0.02,
                      "benign_control": {"false_refusal_rate_on_benign_controls": 0.69}},
        }],
        "excluded": [],
    }
    markdown = render_comparison_markdown(comparison)
    assert "inconclusive" in markdown.lower()
    assert "qlora" in markdown.lower()  # names WHICH arm is degenerate
    assert "benign" in markdown.lower()


def test_winner_is_named_when_neither_arm_is_degenerate():
    comparison = {
        "steering_run": "s", "qlora_run": "q", "min_split_size": 20,
        "comparisons": [{
            "model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
            "steering": {"test_quality": 0.29, "beat_baseline": True, "distinct_output_ratio": 0.9,
                          "benign_control": {"false_refusal_rate_on_benign_controls": 0.05}},
            "qlora": {"test_quality": 1.0, "distinct_output_ratio": 0.8,
                      "benign_control": {"false_refusal_rate_on_benign_controls": 0.10}},
        }],
        "excluded": [],
    }
    markdown = render_comparison_markdown(comparison)
    assert "| qlora |" in markdown  # named plainly, not "inconclusive"


def test_winner_inconclusive_on_collapsed_distinct_output_even_without_benign_data():
    # A recipe with no matched control set (no apply_vectors_from) has no
    # benign_control data at all -- distinct_output_ratio alone must still
    # be able to veto naming a degenerate winner.
    comparison = {
        "steering_run": "s", "qlora_run": "q", "min_split_size": 20,
        "comparisons": [{
            "model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
            "steering": {"test_quality": 0.5, "beat_baseline": True, "distinct_output_ratio": 0.8},
            "qlora": {"test_quality": 1.0, "distinct_output_ratio": 0.016},
        }],
        "excluded": [],
    }
    markdown = render_comparison_markdown(comparison)
    assert "inconclusive" in markdown.lower()
    assert "collapsed" in markdown.lower()


def test_benign_control_section_only_appears_when_data_exists():
    comparison_without = {
        "steering_run": "s", "qlora_run": "q", "min_split_size": 20,
        "comparisons": [{
            "model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification",
            "steering": {"test_quality": 0.5, "beat_baseline": True},
            "qlora": {"test_quality": 0.6},
        }],
        "excluded": [],
    }
    assert "## Safety / degeneracy controls" not in render_comparison_markdown(comparison_without)


def test_control_recipe_ids_finds_recipes_pointing_at_the_source():
    from steering_factory.comparison import _control_recipe_ids

    manifest = {"recipes": [
        {"id": "harmful_instruction_compliance"},
        {"id": "benign_over_refusal_control",
         "dataset": {"apply_vectors_from": "harmful_instruction_compliance",
                      "apply_adapter_from": "harmful_instruction_compliance"}},
        {"id": "unrelated_recipe", "dataset": {}},
    ]}
    assert _control_recipe_ids(manifest, "harmful_instruction_compliance") == ["benign_over_refusal_control"]
    assert _control_recipe_ids(manifest, "unrelated_recipe") == []
    assert _control_recipe_ids({}, "anything") == []


def test_benign_control_stats_scopes_by_model_and_control_recipe():
    from steering_factory.comparison import _benign_control_stats

    rows = [
        {"model_id": "m1", "recipe_id": "control", "is_safe_control": True, "safe_refusal": 1.0, "output": "a"},
        {"model_id": "m1", "recipe_id": "control", "is_safe_control": True, "safe_refusal": 0.0, "output": "b"},
        {"model_id": "m1", "recipe_id": "control", "is_safe_control": False, "safe_refusal": 1.0, "output": "c"},
        {"model_id": "m2", "recipe_id": "control", "is_safe_control": True, "safe_refusal": 1.0, "output": "d"},  # different model
        {"model_id": "m1", "recipe_id": "unrelated", "is_safe_control": True, "safe_refusal": 1.0, "output": "e"},  # different recipe
    ]
    stats = _benign_control_stats(rows, "m1", ["control"])
    assert stats["false_refusal_rate_on_benign_controls"] == pytest.approx(0.5)
    assert stats["n_benign_controls"] == 2
    assert stats["unsafe_contrast_safe_refusal_rate"] == pytest.approx(1.0)
    assert stats["n_unsafe_contrast"] == 1


def test_benign_control_stats_none_when_no_control_recipes():
    from steering_factory.comparison import _benign_control_stats
    assert _benign_control_stats([{"model_id": "m1"}], "m1", []) is None


def test_build_comparison_attaches_benign_control_via_resolved_manifest(tmp_path):
    """End-to-end: build_comparison must read resolved_manifest.yaml from
    the steering run to discover the apply_vectors_from/apply_adapter_from
    link, then attach BOTH arms' benign-control stats to the comparison
    entry -- this is the actual mechanism that was missing on the real
    2026-07-26 run."""
    import yaml

    steer_dir = tmp_path / "steer_run"
    qlora_dir = tmp_path / "qlora_run"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)

    (steer_dir / "resolved_manifest.yaml").write_text(yaml.safe_dump({
        "recipes": [
            {"id": "r1"},
            {"id": "benign_control", "dataset": {"apply_vectors_from": "r1", "apply_adapter_from": "r1"}},
        ],
    }), encoding="utf-8")

    steer_rows = _read_generations(steer_dir)
    steer_rows += [
        {"model_id": "m1", "recipe_id": "benign_control", "is_safe_control": True, "safe_refusal": 0.02, "output": f"s{i}"}
        for i in range(10)
    ]
    _write_jsonl(steer_dir / "results" / "generations.jsonl", steer_rows)

    qlora_rows = _read_generations(qlora_dir)
    qlora_rows += [
        {"model_id": "m1", "recipe_id": "benign_control", "is_safe_control": True, "safe_refusal": 0.9, "output": "same"}
        for _ in range(10)
    ]
    _write_jsonl(qlora_dir / "results" / "generations.jsonl", qlora_rows)

    comparison = build_comparison(steer_dir, qlora_dir)
    entry = comparison["comparisons"][0]
    assert entry["steering"]["benign_control"]["false_refusal_rate_on_benign_controls"] == pytest.approx(0.02)
    assert entry["qlora"]["benign_control"]["false_refusal_rate_on_benign_controls"] == pytest.approx(0.9)


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


# --- Held-out vulnerability protocol guard --------------------------------------

def test_assert_disjoint_selection_and_test_rows_passes_when_disjoint():
    selection = [{"example_id": "a"}, {"example_id": "b"}]
    test = [{"example_id": "c"}, {"example_id": "d"}]
    assert_disjoint_selection_and_test_rows(selection, test)  # must not raise


def test_assert_disjoint_selection_and_test_rows_raises_on_overlap():
    selection = [{"example_id": "a"}, {"example_id": "b"}]
    test = [{"example_id": "b"}, {"example_id": "c"}]  # "b" leaked into both
    with pytest.raises(HeldOutViolation, match="b"):
        assert_disjoint_selection_and_test_rows(selection, test)


def test_assert_disjoint_selection_and_test_rows_ignores_missing_example_ids():
    # Rows with no example_id at all shouldn't spuriously trigger the guard.
    selection = [{"example_id": None}, {}]
    test = [{"example_id": None}, {}]
    assert_disjoint_selection_and_test_rows(selection, test)  # must not raise


def test_best_steering_config_raises_if_validation_and_test_rows_share_example_ids():
    # Constructs a deliberately-leaked fixture (same example_id appearing in
    # both splits for the winning config) to prove the guard fires from
    # inside the real call path, not just when called directly.
    rows = [
        {"split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
         "token_scope": "all", "example_id": "leaked", "exact_match": 1.0},
        {"split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
         "token_scope": "all", "example_id": "leaked", "exact_match": 1.0},  # same id, wrong split
    ]
    with pytest.raises(HeldOutViolation):
        best_steering_config(rows, "domain_classification")


def test_qlora_quality_raises_if_validation_and_test_rows_share_example_ids():
    rows = [
        {"split": "validation", "example_id": "leaked", "exact_match": 1.0},
        {"split": "test", "example_id": "leaked", "exact_match": 1.0},
    ]
    with pytest.raises(HeldOutViolation):
        qlora_quality(rows, "domain_classification")
