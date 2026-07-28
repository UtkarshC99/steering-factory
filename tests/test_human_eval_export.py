"""Tests for human_eval_export.py: the paired steering/QLoRA annotation
package built from two finalized run directories.

Fixture style matches test_comparison.py's own (write generations.jsonl/
vectors/index.jsonl/qlora.json directly to a tmp_path run dir) rather than
running real run_steering/run_qlora end to end.

REBUILT 2026-07-26 alongside the module itself: records now carry the FULL
set of swept steering configs per example (not a 4-slot pinned subset),
plus a manifest-summary header and click-to-select annotation fields.
"""
import csv
import json
from pathlib import Path

import pytest
import yaml

from steering_factory.human_eval_export import (
    ANNOTATION_COLUMNS,
    _manifest_summary,
    build_human_eval_records,
    write_human_eval_package,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


N_PER_SPLIT = 24  # clears comparison.DEFAULT_MIN_SPLIT_SIZE=20, though
                  # build_human_eval_records itself has no such floor --
                  # kept for parity with test_comparison.py's fixtures.


def _steering_rows(model_id="m1", recipe_id="r1", behavior_id="domain_classification", n=N_PER_SPLIT):
    """Three configs swept: mean_diff/L5/c=1 (winner, best on validation),
    mean_diff/L5/c=0 (baseline, same method/layer as winner), mean_diff/
    L5/c=-1 (negative, same method/layer). All three appear on TEST for
    every example -- the full-grid assertion below checks all three show
    up, not just the winner."""
    rows = []
    for i in range(n):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"val-a-{i}", "exact_match": 1.0 if i < n - 1 else 0.0,
                     "category": "cat-a", "prompt": f"prompt-a-{i}", "output": f"winner-output-{i}"})
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "validation", "method": "pca", "layer_idx": 8, "coefficient": 2.0,
                     "token_scope": "all", "example_id": f"val-b-{i}", "exact_match": 0.0,
                     "category": "cat-a", "prompt": f"prompt-b-{i}", "output": f"loser-output-{i}"})
    for i in range(n):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0,
                     "token_scope": "all", "example_id": f"test-{i}", "exact_match": 1.0,
                     "category": "cat-a", "prompt": f"test-prompt-{i}", "output": f"winner-test-output-{i}",
                     "benchmark": "TestBench", "is_safe_control": False})
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 0.0,
                     "token_scope": "all", "example_id": f"test-{i}", "exact_match": 0.0,
                     "category": "cat-a", "prompt": f"test-prompt-{i}", "output": f"baseline-test-output-{i}",
                     "benchmark": "TestBench", "is_safe_control": False})
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": -1.0,
                     "token_scope": "all", "example_id": f"test-{i}", "exact_match": 0.0,
                     "category": "cat-a", "prompt": f"test-prompt-{i}", "output": f"negative-test-output-{i}",
                     "benchmark": "TestBench", "is_safe_control": False})
    return rows


def _qlora_rows(model_id="m1", recipe_id="r1", behavior_id="domain_classification", n=N_PER_SPLIT, num_train_records=200):
    rows = []
    for i in range(n):
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "example_id": f"test-{i}", "exact_match": 0.5,
                     "num_train_records": num_train_records, "output": f"qlora-test-output-{i}"})
    return rows


def _build_steering_run(root: Path, **kwargs):
    _write_jsonl(root / "results" / "generations.jsonl", _steering_rows(**kwargs))
    _write_json(root / "run.json", {"run_id": "steer-1", "status": "completed", "manifest_hash": "abc"})


def _build_qlora_run(root: Path, **kwargs):
    _write_jsonl(root / "results" / "generations.jsonl", _qlora_rows(**kwargs))
    _write_json(root / "results" / "qlora.json", [
        {"model_id": kwargs.get("model_id", "m1"), "recipe_id": kwargs.get("recipe_id", "r1"),
         "num_train_records": kwargs.get("num_train_records", 200)},
    ])
    _write_json(root / "run.json", {"run_id": "qlora-1", "status": "completed", "manifest_hash": "abc"})


@pytest.fixture
def run_dirs(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir)
    return steer_dir, qlora_dir


def test_one_record_per_test_example(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    assert len(records) == N_PER_SPLIT
    assert {r["example_id"] for r in records} == {f"test-{i}" for i in range(N_PER_SPLIT)}


def test_record_carries_all_three_swept_configs(run_dirs):
    """The core fix: a real run had 13 swept coefficients but the old
    exporter only surfaced 3 pinned slots. Every config actually generated
    for an example must appear in steering_configs."""
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        coeffs = sorted(cfg["coefficient"] for cfg in record["steering_configs"])
        assert coeffs == [-1.0, 0.0, 1.0]


def test_winner_config_is_flagged(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        winners = [cfg for cfg in record["steering_configs"] if cfg["is_winner_config"]]
        assert len(winners) == 1
        assert winners[0]["coefficient"] == 1.0


def test_baseline_config_is_flagged(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        baselines = [cfg for cfg in record["steering_configs"] if cfg["is_baseline"]]
        assert len(baselines) == 1
        assert baselines[0]["coefficient"] == 0.0


def test_config_outputs_match_their_own_rows(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        by_coeff = {cfg["coefficient"]: cfg["output"] for cfg in record["steering_configs"]}
        assert by_coeff[1.0].startswith("winner-test-output-")
        assert by_coeff[0.0].startswith("baseline-test-output-")
        assert by_coeff[-1.0].startswith("negative-test-output-")


def test_qlora_output_present(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        assert record["qlora"]["output"].startswith("qlora-test-output-")


def test_context_fields_carried_from_rows(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        assert record["model_id"] == "m1"
        assert record["recipe_id"] == "r1"
        assert record["behavior_id"] == "domain_classification"
        assert record["category"] == "cat-a"
        assert record["benchmark"] == "TestBench"
        assert record["is_safe_control"] is False
        assert record["prompt"].startswith("test-prompt-")


def test_annotation_fields_present_and_empty(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        for col in ANNOTATION_COLUMNS:
            assert record[col] == ""


def test_scores_carried_alongside_outputs(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        by_coeff = {cfg["coefficient"]: cfg["scores"] for cfg in record["steering_configs"]}
        assert by_coeff[1.0]["exact_match"] == 1.0
        assert by_coeff[0.0]["exact_match"] == 0.0
        assert record["qlora"]["scores"]["exact_match"] == 0.5


def test_only_test_and_validation_splits_by_default(run_dirs):
    # validation rows for THIS recipe have no matching qlora rows in the
    # fixture, so only test-split records are produced -- matches the old
    # behavior, still gated by the (example_id, split) join against qlora.
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    assert all(r["split"] == "test" for r in records)


def test_no_overlap_between_arms_produces_no_records(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, recipe_id="different_recipe")
    records = build_human_eval_records(steer_dir, qlora_dir)
    assert records == []


def test_qlora_pinned_to_max_num_train_records(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    _build_steering_run(steer_dir)
    low_n_rows = _qlora_rows(num_train_records=50)
    high_n_rows = [dict(r, output=r["output"] + "-HIGHN") for r in _qlora_rows(num_train_records=200)]
    _write_jsonl(qlora_dir / "results" / "generations.jsonl", low_n_rows + high_n_rows)
    _write_json(qlora_dir / "results" / "qlora.json", [
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 50},
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 200},
    ])
    _write_json(qlora_dir / "run.json", {"run_id": "qlora-1", "status": "completed", "manifest_hash": "abc"})

    records = build_human_eval_records(steer_dir, qlora_dir)
    assert records
    assert all(r["qlora"]["output"].endswith("-HIGHN") for r in records)


def test_no_validation_data_still_produces_test_records(tmp_path):
    """Unlike the old exporter (which used best_steering_config to select
    a single winner and skipped the whole pair if that returned None), the
    new one shows every config regardless of whether a winner could be
    determined -- winner_config is just None/absent in that case."""
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    test_only_rows = [r for r in _steering_rows() if r["split"] == "test"]
    _write_jsonl(steer_dir / "results" / "generations.jsonl", test_only_rows)
    _build_qlora_run(qlora_dir)
    records = build_human_eval_records(steer_dir, qlora_dir)
    assert records
    for record in records:
        assert not any(cfg["is_winner_config"] for cfg in record["steering_configs"])


# --- manifest summary --------------------------------------------------------

def test_manifest_summary_reads_resolved_manifest(tmp_path):
    steer_dir = tmp_path / "steer"
    steer_dir.mkdir(parents=True)
    manifest = {
        "experiment": {"name": "test-experiment", "seed": 17, "n_sweep": [10, 40]},
        "models": [{"id": "m1", "name_or_path": "org/Model-1", "dtype": "bfloat16", "quantization": "4bit"}],
        "splits": {"steer_fraction": 0.4, "validation_fraction": 0.2, "group_key": "category"},
        "decoding": {"max_new_tokens": 256},
    }
    (steer_dir / "resolved_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    summary = _manifest_summary(steer_dir)
    assert summary["experiment_name"] == "test-experiment"
    assert summary["seed"] == 17
    assert summary["n_sweep"] == [10, 40]
    assert summary["models"][0]["id"] == "m1"
    assert summary["max_new_tokens"] == 256


def test_manifest_summary_empty_without_resolved_manifest(tmp_path):
    steer_dir = tmp_path / "steer"
    steer_dir.mkdir(parents=True)
    assert _manifest_summary(steer_dir) == {}


def test_manifest_summary_accepts_a_string_path(tmp_path):
    # Regression: _manifest_summary did `steer_dir / "resolved_manifest.yaml"`
    # without coercing to Path first, unlike every other function in this
    # module -- crashed with a plain str argument (the shape build_comparison/
    # the CLI actually pass run paths in as).
    steer_dir = tmp_path / "steer"
    steer_dir.mkdir(parents=True)
    manifest = {"experiment": {"name": "str-path-test"}}
    (steer_dir / "resolved_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    summary = _manifest_summary(str(steer_dir))
    assert summary["experiment_name"] == "str-path-test"


# --- package writing ----------------------------------------------------------

def test_write_human_eval_package_creates_expected_files(run_dirs, tmp_path):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    output_root = tmp_path / "output"
    package_dir = write_human_eval_package(records, output_root)

    assert package_dir == output_root / "human_eval"
    assert (package_dir / "pairs.jsonl").exists()
    assert (package_dir / "pairs.csv").exists()
    assert (package_dir / "review.html").exists()
    assert (package_dir / "README.md").exists()


def test_jsonl_round_trips(run_dirs, tmp_path):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")

    with (package_dir / "pairs.jsonl").open(encoding="utf-8") as handle:
        loaded = [json.loads(line) for line in handle if line.strip()]
    assert len(loaded) == len(records)
    assert loaded[0]["example_id"] == records[0]["example_id"]
    assert len(loaded[0]["steering_configs"]) == 3


def test_csv_has_one_column_set_per_config(run_dirs, tmp_path):
    """The CSV must expand to all swept coefficients (config_0_*,
    config_1_*, config_2_*), not the old 4-pinned-slot shape."""
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")

    with (package_dir / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert len(rows) == len(records)
    assert "config_0_output" in fieldnames
    assert "config_1_output" in fieldnames
    assert "config_2_output" in fieldnames
    assert "config_0_coefficient" in fieldnames
    assert "qlora_output" in fieldnames
    for col in ANNOTATION_COLUMNS:
        assert col in fieldnames
        assert all(row[col] == "" for row in rows)


def test_csv_round_trips_config_outputs(run_dirs, tmp_path):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")

    with (package_dir / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    all_outputs = " ".join(v for row in rows for k, v in row.items() if k.endswith("_output"))
    assert "winner-test-output" in all_outputs
    assert "baseline-test-output" in all_outputs
    assert "negative-test-output" in all_outputs
    assert "qlora-test-output" in all_outputs


def test_html_is_self_contained(run_dirs, tmp_path):
    """No external requests: no http(s):// references, no <script src=,
    no <link> stylesheet imports -- the page must be openable from a plain
    file:// path or fully offline."""
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")

    html_text = (package_dir / "review.html").read_text(encoding="utf-8")
    assert len(html_text) > 0
    assert "http://" not in html_text
    assert "https://" not in html_text
    assert "<script src=" not in html_text
    assert "<link " not in html_text
    assert "winner-test-output" in html_text
    assert "baseline-test-output" in html_text
    assert "negative-test-output" in html_text
    assert "qlora-test-output" in html_text
    assert "Download annotations as CSV" in html_text


def test_html_contains_manifest_header_data(run_dirs, tmp_path):
    steer_dir, qlora_dir = run_dirs
    manifest = {
        "experiment": {"name": "my-experiment", "seed": 17},
        "models": [{"id": "m1", "name_or_path": "org/Model-1", "dtype": "bfloat16", "quantization": "4bit"}],
        "splits": {"steer_fraction": 0.4, "validation_fraction": 0.2},
        "decoding": {"max_new_tokens": 256},
    }
    (steer_dir / "resolved_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    records = build_human_eval_records(steer_dir, qlora_dir)
    summary = _manifest_summary(steer_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output", summary)

    html_text = (package_dir / "review.html").read_text(encoding="utf-8")
    assert "my-experiment" in html_text
    assert "org/Model-1" in html_text


def test_html_escapes_output_text(tmp_path):
    """A prompt/output containing HTML-special characters must not break
    the page structure or allow injection."""
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    rows = _steering_rows(n=1)
    for row in rows:
        if row["example_id"] == "test-0" and row["split"] == "test" and row["coefficient"] == 1.0:
            row["output"] = "<script>alert('x')</script> & \"quotes\""
    _write_jsonl(steer_dir / "results" / "generations.jsonl", rows)
    _build_qlora_run(qlora_dir, n=1)
    _write_json(steer_dir / "run.json", {"run_id": "steer-1", "status": "completed", "manifest_hash": "abc"})

    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")
    html_text = (package_dir / "review.html").read_text(encoding="utf-8")
    assert "<script>alert" not in html_text
    assert "\\u003cscript" in html_text


def test_readme_mentions_all_four_files(run_dirs, tmp_path):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")
    readme = (package_dir / "README.md").read_text(encoding="utf-8")
    for name in ("pairs.jsonl", "pairs.csv", "review.html"):
        assert name in readme


def test_empty_records_still_writes_valid_empty_package(tmp_path):
    package_dir = write_human_eval_package([], tmp_path / "output")
    assert (package_dir / "pairs.jsonl").read_text(encoding="utf-8") == ""
    with (package_dir / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert len(rows) == 1  # header only, no data rows
    for col in ANNOTATION_COLUMNS:
        assert col in rows[0]
    assert "0 examples" in (package_dir / "README.md").read_text(encoding="utf-8")
    assert (package_dir / "review.html").exists()
