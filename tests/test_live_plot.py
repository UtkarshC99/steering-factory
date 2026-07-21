"""Headless smoke test for live_plot.py -- constructs each FigureWidget and
feeds it synthetic events, without an actual Jupyter/Colab kernel. Skipped
if plotly/ipywidgets aren't installed, since live_plot is a notebook-only
extra, not a dependency of the core package.
"""
import pytest

pytest.importorskip("plotly")
pytest.importorskip("ipywidgets")

from steering_factory.live_plot import BiPOConvergencePlot, QLoRALossPlot, SteeringGridPlot


def test_steering_grid_plot_ingests_generation_events():
    plot = SteeringGridPlot(update_every=1)
    for i in range(1, 4):
        plot.on_event({
            "event": "generation", "i": i, "n": 3,
            "perplexity_ratio_vs_baseline": 1.0 + i * 0.1,
            "js_divergence_vs_baseline": 0.05 * i,
        })
    assert list(plot.figure.data[0].x) == [1, 2, 3]
    assert list(plot.figure.data[0].y) == [1.1, 1.2, 1.3]
    assert "3/3" in plot.figure.layout.title.text


def test_steering_grid_plot_ignores_vector_events():
    plot = SteeringGridPlot(update_every=1)
    plot.on_event({"event": "vector", "i": 1, "n": 1})
    assert plot._vector_events == 1
    assert list(plot.figure.data[0].x) == []


def test_steering_grid_plot_buffers_until_update_every():
    plot = SteeringGridPlot(update_every=3)
    plot.on_event({"event": "generation", "i": 1, "n": 5, "perplexity_ratio_vs_baseline": 1.0, "js_divergence_vs_baseline": 0.1})
    plot.on_event({"event": "generation", "i": 2, "n": 5, "perplexity_ratio_vs_baseline": 1.1, "js_divergence_vs_baseline": 0.2})
    # Only 2 of 3 buffered -- widget should not have redrawn yet.
    assert list(plot.figure.data[0].x) == []
    plot.on_event({"event": "generation", "i": 3, "n": 5, "perplexity_ratio_vs_baseline": 1.2, "js_divergence_vs_baseline": 0.3})
    assert list(plot.figure.data[0].x) == [1, 2, 3]


def test_qlora_loss_plot_ingests_train_log_events():
    plot = QLoRALossPlot(update_every=1)
    plot.on_event({"event": "train_log", "step": 1, "max_steps": 10, "loss": 2.0, "grad_norm": 0.5, "model_id": "m", "recipe_id": "r"})
    plot.on_event({"event": "train_log", "step": 2, "max_steps": 10, "loss": 1.5, "grad_norm": 0.4, "model_id": "m", "recipe_id": "r"})
    assert list(plot.figure.data[0].y) == [2.0, 1.5]
    assert "m/r" in plot.figure.layout.title.text


def test_qlora_loss_plot_skips_events_without_loss():
    plot = QLoRALossPlot(update_every=1)
    plot.on_event({"event": "train_log", "step": 1, "loss": None})
    assert list(plot.figure.data[0].x) == []


def test_bipo_convergence_plot_ingests_opt_step_events():
    plot = BiPOConvergencePlot(update_every=1)
    for i in range(1, 4):
        plot.on_event({"event": "opt_step", "i": i, "n": 3, "loss": -float(i), "grad_norm": 0.1 * i})
    assert list(plot.figure.data[0].y) == [-1.0, -2.0, -3.0]
