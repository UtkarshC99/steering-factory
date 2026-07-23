"""Tests for human_eval_export.py: the paired steering/QLoRA annotation
package built from two finalized run directories.

Fixture style matches test_comparison.py's own (write generations.jsonl/
vectors/index.jsonl/qlora.json directly to a tmp_path run dir) rather than
running real run_steering/run_qlora end to end -- run_qlora needs real
peft/trl (optional dependency, not installed in this environment), and the
join logic under test here only reads what's already on disk, exactly like
comparison.build_comparison does.
"""
import csv
import json
from pathlib import Path

import pytest

from steering_factory.human_eval_export import (
    ANNOTATION_COLUMNS,
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
                  # kept for parity with the fixture pattern in
                  # test_comparison.py so best_steering_config always finds
                  # a real winner rather than an artificially thin one.


def _steering_rows(model_id="m1", recipe_id="r1", behavior_id="domain_classification", n=N_PER_SPLIT):
    """Two configs on validation (mean_diff/L5/c=1 wins; pca/L8/c=2 loses),
    held-out test rows for BOTH the winner and the baseline (coefficient=0,
    same method/layer as the winner) -- the baseline rows are what
    build_human_eval_records's steering_baseline_output column reads."""
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
        # Baseline: SAME method/layer/token_scope as the winner, coefficient=0.
        rows.append({"model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                     "split": "test", "method": "mean_diff", "layer_idx": 5, "coefficient": 0.0,
                     "token_scope": "all", "example_id": f"test-{i}", "exact_match": 0.0,
                     "category": "cat-a", "prompt": f"test-prompt-{i}", "output": f"baseline-test-output-{i}",
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
    # Only "test" split rows exist in both fixtures' overlap (validation has
    # no qlora rows in this fixture) -- N_PER_SPLIT test examples.
    assert len(records) == N_PER_SPLIT
    assert {r["example_id"] for r in records} == {f"test-{i}" for i in range(N_PER_SPLIT)}


def test_winner_output_matches_best_steering_config(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        assert record["winner_method"] == "mean_diff"
        assert record["winner_layer_idx"] == 5
        assert record["winner_coefficient"] == 1.0
        assert record["steering_winner_output"].startswith("winner-test-output-")


def test_baseline_output_is_coefficient_zero_same_config(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        assert record["steering_baseline_output"].startswith("baseline-test-output-")


def test_qlora_output_present(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        assert record["qlora_output"].startswith("qlora-test-output-")


def test_context_fields_carried_from_winner_row(run_dirs):
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


def test_annotation_columns_present_and_empty(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        for col in ANNOTATION_COLUMNS:
            assert record[col] == ""


def test_auto_scores_carried_alongside_outputs(run_dirs):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    for record in records:
        assert record["steering_winner_exact_match"] == 1.0
        assert record["steering_baseline_exact_match"] == 0.0
        assert record["qlora_exact_match"] == 0.5


def test_no_validation_data_skips_the_pair_entirely(tmp_path):
    # best_steering_config returns None with no validation rows -- the pair
    # must be skipped, not crash.
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    test_only_rows = [r for r in _steering_rows() if r["split"] == "test"]
    _write_jsonl(steer_dir / "results" / "generations.jsonl", test_only_rows)
    _build_qlora_run(qlora_dir)
    records = build_human_eval_records(steer_dir, qlora_dir)
    assert records == []


def test_no_matching_qlora_recipe_produces_no_records(tmp_path):
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
    assert all(r["qlora_output"].endswith("-HIGHN") for r in records)


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


def test_csv_round_trips_with_stable_header_and_annotation_columns(run_dirs, tmp_path):
    steer_dir, qlora_dir = run_dirs
    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")

    with (package_dir / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert len(rows) == len(records)
    for col in ANNOTATION_COLUMNS:
        assert col in fieldnames
        assert fieldnames.index(col) >= fieldnames.index("qlora_output")  # annotation columns come last
        assert all(row[col] == "" for row in rows)
    assert "steering_winner_output" in fieldnames
    assert "steering_baseline_output" in fieldnames
    assert "qlora_output" in fieldnames


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
    # The three outputs and the download button must actually be present.
    assert "steering_winner_output" not in html_text or "winner-test-output" in html_text
    assert "Download annotations as CSV" in html_text
    assert "qlora-test-output" in html_text


def test_html_escapes_output_text(tmp_path):
    """A prompt/output containing HTML-special characters must not break
    the page structure or allow injection -- html.escape is applied."""
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    rows = _steering_rows(n=1)
    for row in rows:
        if row["example_id"] == "test-0" and row["split"] == "test":
            row["output"] = "<script>alert('x')</script> & \"quotes\""
    _write_jsonl(steer_dir / "results" / "generations.jsonl", rows)
    _build_qlora_run(qlora_dir, n=1)
    _write_json(steer_dir / "run.json", {"run_id": "steer-1", "status": "completed", "manifest_hash": "abc"})

    records = build_human_eval_records(steer_dir, qlora_dir)
    package_dir = write_human_eval_package(records, tmp_path / "output")
    html_text = (package_dir / "review.html").read_text(encoding="utf-8")
    assert "<script>alert" not in html_text
    assert "&lt;script&gt;" in html_text


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
    # No records, but the fixed context/config/output/annotation columns
    # are still known statically -- a header-only CSV (still valid,
    # openable, showing the right columns) is more useful to a reviewer
    # than a fully empty file with no clue what columns to expect.
    with (package_dir / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert len(rows) == 1  # header only, no data rows
    for col in ANNOTATION_COLUMNS:
        assert col in rows[0]
    assert "0 paired examples" in (package_dir / "README.md").read_text(encoding="utf-8")
    assert (package_dir / "review.html").exists()
