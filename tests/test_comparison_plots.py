"""Tests for full_config_grid (comparison.py) and the static PNG plots
(comparison_plots.py) built from the comparison report data.

matplotlib is already a project dependency (requirements.txt), so these
run normally; they degrade gracefully (skip) if it's ever absent, matching
comparison_plots.py's own "return None, don't raise" contract.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from steering_factory.comparison import build_comparison, full_config_grid
from steering_factory.comparison_plots import (
    matplotlib_available,
    plot_config_grid_heatmap,
    plot_data_efficiency_curve,
    render_comparison_plots,
)
from steering_factory.comparison_report import render_comparison_markdown

from test_comparison import _build_qlora_run, _build_steering_run, _write_json, _write_jsonl

pytestmark = pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")


def _grid_rows(vary_layer=True, vary_coefficient=True, winner_quality=1.0, other_quality=0.2):
    layers = (5, 10, 15) if vary_layer else (5,)
    coeffs = (-1.0, 0.0, 1.0, 2.0) if vary_coefficient else (1.0,)
    rows = []
    for layer in layers:
        for coeff in coeffs:
            for i in range(3):
                quality = winner_quality if (layer == (layers[-1] if vary_layer else layers[0]) and coeff == 1.0) else other_quality
                rows.append({"split": "test", "method": "mean_diff", "layer_idx": layer, "coefficient": coeff,
                             "token_scope": "all", "example_id": f"{layer}-{coeff}-{i}", "exact_match": quality})
    return rows


# --- full_config_grid --------------------------------------------------------

def test_full_config_grid_reports_every_config_not_just_the_winner():
    rows = _grid_rows()
    grid = full_config_grid(rows, "domain_classification")
    assert len(grid) == 3 * 4  # 3 layers x 4 coefficients
    assert all("test_quality" in r and "n_test" in r for r in grid)


def test_full_config_grid_ignores_validation_rows():
    rows = _grid_rows()
    for row in rows:
        row["split"] = "validation"
    assert full_config_grid(rows, "domain_classification") == []


def test_full_config_grid_returns_empty_without_recognizable_quality_key():
    assert full_config_grid([{"split": "test", "method": "mean_diff", "layer_idx": 1, "coefficient": 1.0}], None) == []


def test_full_config_grid_is_deterministically_ordered():
    rows = _grid_rows()
    grid_a = full_config_grid(rows, "domain_classification")
    grid_b = full_config_grid(list(reversed(rows)), "domain_classification")
    assert grid_a == grid_b


# --- plot_data_efficiency_curve ---------------------------------------------

def test_plot_data_efficiency_curve_writes_a_file(tmp_path):
    data_efficiency = {
        "model_id": "m1", "recipe_id": "r1", "quality_key": "exact_match",
        "steering": [{"labeled_examples": 8, "quality": 0.2}, {"labeled_examples": 64, "quality": 0.8}],
        "qlora": [{"labeled_examples": 200, "quality": 0.5}],
    }
    path = plot_data_efficiency_curve(data_efficiency, tmp_path / "curve.png")
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_data_efficiency_curve_returns_none_when_no_points(tmp_path):
    data_efficiency = {"model_id": "m1", "recipe_id": "r1", "quality_key": "exact_match", "steering": [], "qlora": []}
    assert plot_data_efficiency_curve(data_efficiency, tmp_path / "curve.png") is None


# --- plot_config_grid_heatmap ------------------------------------------------

def test_plot_config_grid_heatmap_writes_a_file_for_a_2d_grid(tmp_path):
    grid = full_config_grid(_grid_rows(), "domain_classification")
    path = plot_config_grid_heatmap(grid, tmp_path / "heatmap.png", qlora_quality=0.5)
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_config_grid_heatmap_returns_none_for_single_layer_or_coefficient(tmp_path):
    # Only one layer varies (no coefficient variation) -- not a real 2D grid.
    grid = full_config_grid(_grid_rows(vary_coefficient=False), "domain_classification")
    assert plot_config_grid_heatmap(grid, tmp_path / "heatmap.png") is None


def test_plot_config_grid_heatmap_returns_none_for_empty_grid(tmp_path):
    assert plot_config_grid_heatmap([], tmp_path / "heatmap.png") is None


def test_plot_config_grid_heatmap_selects_requested_method(tmp_path):
    rows = _grid_rows()
    rows += [{**r, "method": "pca"} for r in _grid_rows(winner_quality=0.9)]
    grid = full_config_grid(rows, "domain_classification")
    path = plot_config_grid_heatmap(grid, tmp_path / "heatmap.png", method="pca")
    assert path is not None


# --- render_comparison_plots + markdown embedding ---------------------------

def test_render_comparison_plots_writes_curve_for_each_entry(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)
    comparison = build_comparison(steer_dir, qlora_dir)

    out = tmp_path / "plots"
    written = render_comparison_plots(comparison, out)
    assert "m1__r1" in written
    assert any("data_efficiency" in name for name in written["m1__r1"])
    for name in written["m1__r1"]:
        assert (out / name).exists()


def test_render_comparison_plots_skips_excluded_entries(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    _write_jsonl(steer_dir / "results" / "generations.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification", "split": "validation",
         "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0, "token_scope": "all", "example_id": "v",
         "exact_match": 1.0},
        {"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification", "split": "test",
         "method": "mean_diff", "layer_idx": 5, "coefficient": 1.0, "token_scope": "all", "example_id": "t",
         "exact_match": 1.0},
    ])
    _write_jsonl(steer_dir / "vectors" / "index.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "method": "mean_diff", "layer_idx": 5, "num_pairs": 1,
         "vector_path": str(steer_dir / "vectors" / "v1.pt")},
    ])
    (steer_dir / "vectors" / "v1.pt").write_bytes(b"0")
    _write_json(steer_dir / "telemetry.json", {"wall_time_s": 1.0})
    _write_jsonl(qlora_dir / "results" / "generations.jsonl", [
        {"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification", "split": "validation",
         "example_id": "qv", "exact_match": 1.0},
        {"model_id": "m1", "recipe_id": "r1", "behavior_id": "domain_classification", "split": "test",
         "example_id": "qt", "exact_match": 1.0},
    ])
    _write_json(qlora_dir / "results" / "qlora.json", [
        {"model_id": "m1", "recipe_id": "r1", "num_train_records": 1, "wall_time_s": 1.0, "adapter_size_bytes": 1, "adapter_dir": "x"},
    ])

    comparison = build_comparison(steer_dir, qlora_dir)  # n=1 everywhere -> excluded, not comparisons
    assert comparison["comparisons"] == []
    written = render_comparison_plots(comparison, tmp_path / "plots")
    assert written == {}


def test_markdown_embeds_plot_image_links_when_paths_given(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)
    comparison = build_comparison(steer_dir, qlora_dir)

    markdown = render_comparison_markdown(comparison, plot_paths={"m1__r1": ["m1__r1__data_efficiency.png"]})
    assert "## Visualizations" in markdown
    assert "![m1__r1__data_efficiency.png](m1__r1__data_efficiency.png)" in markdown


def test_markdown_omits_visualizations_section_without_plot_paths(tmp_path):
    steer_dir, qlora_dir = tmp_path / "steer", tmp_path / "qlora"
    _build_steering_run(steer_dir)
    _build_qlora_run(qlora_dir, test_quality=0.5)
    comparison = build_comparison(steer_dir, qlora_dir)

    markdown = render_comparison_markdown(comparison)  # no plot_paths -- e.g. matplotlib absent
    assert "## Visualizations" not in markdown
