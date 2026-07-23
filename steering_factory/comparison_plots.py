"""Static PNG visualizations for the steering-vs-QLoRA comparison report.

matplotlib is imported ONLY in this module -- `comparison.py` (the pure
data-joining/curve-building logic) and `comparison_report.py`'s Markdown/
JSON writers stay plot-dependency-free, exactly the isolation
`live_plot.py` already applies for plotly. Unlike `live_plot.py`, these
plots are NOT live widgets: `report.md`/`report.json` are static,
post-hoc artifacts meant to be viewed later (possibly outside any
notebook, e.g. downloaded from Drive), so a plotly `FigureWidget` -- which
only updates inside the kernel that displayed it -- is the wrong tool
here. Plain PNGs referenced from Markdown render everywhere with no JS
runtime and no size bloat.

If matplotlib isn't installed, every function here returns None instead of
raising -- `comparison_report.write_comparison_report` treats a None
return as "skip this plot", so a report still writes successfully (just
without images) in a minimal/CI environment that never installed the
optional plotting extra.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: never requires a display, safe in CI/Colab/servers
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when matplotlib is absent
    _MATPLOTLIB_AVAILABLE = False


def matplotlib_available() -> bool:
    return _MATPLOTLIB_AVAILABLE


def plot_data_efficiency_curve(
    data_efficiency: Dict[str, Any], output_path: str | Path,
) -> Optional[Path]:
    """Quality vs. labeled-example count, one line per arm, log-scaled x
    axis (N typically spans 8 to 512+, where a linear axis would crowd the
    low end). This is the direct answer to "at what point does one method
    overtake the other" on the data-volume axis: the crossing point of the
    two lines, if any, is exactly that point. Returns None (does nothing)
    if matplotlib is unavailable or there isn't at least one point in
    either arm's series to plot.
    """
    if not _MATPLOTLIB_AVAILABLE:
        return None
    steering_points = sorted(data_efficiency.get("steering", []), key=lambda p: p["labeled_examples"])
    qlora_points = sorted(data_efficiency.get("qlora", []), key=lambda p: p["labeled_examples"])
    if not steering_points and not qlora_points:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    if steering_points:
        ax.plot([p["labeled_examples"] for p in steering_points], [p["quality"] for p in steering_points],
                marker="o", label="steering", color="#4C72B0")
    if qlora_points:
        ax.plot([p["labeled_examples"] for p in qlora_points], [p["quality"] for p in qlora_points],
                marker="s", label="qlora", color="#DD8452")
    ax.set_xscale("log")
    ax.set_xlabel("labeled examples (N)")
    ax.set_ylabel(f"quality ({data_efficiency.get('quality_key', 'n/a')})")
    ax.set_title(f"Data efficiency -- {data_efficiency.get('model_id')} / {data_efficiency.get('recipe_id')}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_config_grid_heatmap(
    config_grid: List[Dict[str, Any]], output_path: str | Path,
    qlora_quality: Optional[float] = None, method: Optional[str] = None,
) -> Optional[Path]:
    """Layer x coefficient heatmap of held-out test quality for one
    extraction method (pass `method` to select one out of a multi-method
    grid; defaults to the first method present). This answers "with what
    parameters does steering do best/worst" and gives context for whether
    the validation-selected winner is a robust peak or an isolated spike
    surrounded by much worse neighbors. If `qlora_quality` is given, it is
    annotated as a reference line's worth of context in the title so the
    reader can see at a glance how much of the grid actually beats QLoRA,
    not just the single selected point.

    Returns None if matplotlib is unavailable, the grid is empty, or the
    selected method has fewer than 2 distinct layers or coefficients (a
    heatmap needs a 2D grid; a single row/column isn't one).
    """
    if not _MATPLOTLIB_AVAILABLE or not config_grid:
        return None
    import numpy as np

    target_method = method or config_grid[0]["method"]
    rows = [r for r in config_grid if r["method"] == target_method]
    layers = sorted({r["layer_idx"] for r in rows if r["layer_idx"] is not None})
    coeffs = sorted({r["coefficient"] for r in rows if r["coefficient"] is not None})
    if len(layers) < 2 or len(coeffs) < 2:
        return None

    grid = np.full((len(layers), len(coeffs)), np.nan)
    layer_pos = {layer: i for i, layer in enumerate(layers)}
    coeff_pos = {coeff: i for i, coeff in enumerate(coeffs)}
    for row in rows:
        grid[layer_pos[row["layer_idx"]], coeff_pos[row["coefficient"]]] = row["test_quality"]

    fig, ax = plt.subplots(figsize=(max(4, len(coeffs) * 0.8), max(3, len(layers) * 0.6)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(coeffs)))
    ax.set_xticklabels([str(c) for c in coeffs])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xlabel("coefficient")
    ax.set_ylabel("layer")
    title = f"Layer x coefficient quality -- {target_method}"
    if qlora_quality is not None:
        beats = int(np.nansum(grid > qlora_quality))
        total = int(np.sum(~np.isnan(grid)))
        title += f"\n({beats}/{total} configs beat qlora's {qlora_quality:.3f})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="test quality")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_comparison_plots(comparison: Dict[str, Any], output_root: str | Path) -> Dict[str, List[str]]:
    """Renders every plot for every entry in `comparison["comparisons"]`
    (excluded entries have no reliable data to plot and are skipped, same
    discipline as the winner table). Returns {relative_path: [entry keys]}
    style mapping actually just {model_id}__{recipe_id}: [png relative
    paths written], so `comparison_report.py` can embed them as Markdown
    image links without recomputing anything. Silently produces an empty
    mapping if matplotlib isn't installed -- callers should not treat that
    as an error."""
    output = Path(output_root)
    written: Dict[str, List[str]] = {}
    if not _MATPLOTLIB_AVAILABLE:
        return written

    for entry in comparison.get("comparisons", []):
        key = f"{entry['model_id']}__{entry['recipe_id']}"
        paths: List[str] = []

        curve_path = plot_data_efficiency_curve(
            entry.get("data_efficiency", {}), output / f"{key}__data_efficiency.png",
        )
        if curve_path is not None:
            paths.append(curve_path.name)

        heatmap_path = plot_config_grid_heatmap(
            entry.get("config_grid", []), output / f"{key}__config_grid.png",
            qlora_quality=entry["qlora"].get("test_quality"), method=entry["steering"].get("method"),
        )
        if heatmap_path is not None:
            paths.append(heatmap_path.name)

        if paths:
            written[key] = paths
    return written
