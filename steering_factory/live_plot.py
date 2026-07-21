"""In-notebook live plots driven by RunCallback events.

Plotly is imported ONLY in this module -- `callbacks.py` and every runner
entrypoint stay dependency-free so the CLI/subprocess path never needs
plotly or ipywidgets installed. These classes are meant to be constructed
in a notebook kernel, `display()`-ed once, then passed as a callback into
an in-process `run_steering(..., callback=...)` call: because the
`FigureWidget` lives in the same kernel as the display, mutating
`fig.data[i].y` re-renders the already-displayed cell in place -- no
`clear_output`/redraw, no subprocess boundary to cross.

If plotly/ipywidgets aren't installed, importing this module raises
ImportError with a clear message; nothing else in the package imports it,
so that's only ever surfaced when a notebook explicitly opts in.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - exercised only when plotly is absent
    raise ImportError(
        "live_plot requires plotly and ipywidgets: pip install plotly ipywidgets. "
        "This is only needed for in-notebook live plots -- the CLI/subprocess path "
        "and the rest of steering_factory do not depend on it."
    ) from exc


def _fmt_eta(elapsed_s: float, i: int, n: int) -> str:
    if i <= 0 or n <= 0:
        return "eta n/a"
    remaining = elapsed_s / i * (n - i)
    return f"eta {remaining:.0f}s"


class SteeringGridPlot:
    """Live plot for `run_steering`/`run_extract`/`run_evaluate` progress.

    Tracks two running series against generation index: perplexity-ratio-
    vs-baseline and JS-divergence-vs-baseline (both already computed per
    row by the runner, just not previously surfaced until the run ended).
    Buffers incoming events and only touches the widget every
    `update_every` events, so a fast grid doesn't thrash the frontend.
    """

    def __init__(self, update_every: int = 5, title: str = "Steering grid progress"):
        self.update_every = max(1, update_every)
        self._started = time.perf_counter()
        self._buffered = 0
        self._x: List[int] = []
        self._ppl: List[Optional[float]] = []
        self._js: List[Optional[float]] = []
        self._vector_events = 0

        self.figure = go.FigureWidget(
            data=[
                go.Scatter(x=[], y=[], mode="lines+markers", name="perplexity ratio vs baseline"),
                go.Scatter(x=[], y=[], mode="lines+markers", name="JS divergence vs baseline", yaxis="y2"),
            ],
            layout=go.Layout(
                title=title,
                xaxis=dict(title="generation index"),
                yaxis=dict(title="perplexity ratio"),
                yaxis2=dict(title="JS divergence", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                height=380,
                margin=dict(t=60, b=40),
            ),
        )

    def on_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "vector":
            self._vector_events += 1
            return
        if kind not in ("generation",):
            return
        self._x.append(event.get("i", len(self._x) + 1))
        self._ppl.append(event.get("perplexity_ratio_vs_baseline"))
        self._js.append(event.get("js_divergence_vs_baseline"))
        self._buffered += 1
        if self._buffered < self.update_every:
            return
        self._buffered = 0
        self._redraw(event)

    def _redraw(self, event: Dict[str, Any]) -> None:
        with self.figure.batch_update():
            self.figure.data[0].x = self._x
            self.figure.data[0].y = self._ppl
            self.figure.data[1].x = self._x
            self.figure.data[1].y = self._js
            i, n = event.get("i"), event.get("n")
            elapsed = time.perf_counter() - self._started
            suffix = f" -- {i}/{n} ({_fmt_eta(elapsed, i, n)})" if i and n else ""
            self.figure.layout.title = f"Steering grid progress{suffix}"


class QLoRALossPlot:
    """Live plot for `train_qlora`'s `"train_log"` events (forwarded from
    transformers' own Trainer logging, one point per `logging_steps`)."""

    def __init__(self, update_every: int = 1, title: str = "QLoRA training"):
        self.update_every = max(1, update_every)
        self._buffered = 0
        self._step: List[int] = []
        self._loss: List[Optional[float]] = []
        self._grad_norm: List[Optional[float]] = []

        self.figure = go.FigureWidget(
            data=[
                go.Scatter(x=[], y=[], mode="lines+markers", name="loss"),
                go.Scatter(x=[], y=[], mode="lines+markers", name="grad norm", yaxis="y2"),
            ],
            layout=go.Layout(
                title=title,
                xaxis=dict(title="step"),
                yaxis=dict(title="loss"),
                yaxis2=dict(title="grad norm", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                height=380,
                margin=dict(t=60, b=40),
            ),
        )

    def on_event(self, event: Dict[str, Any]) -> None:
        if event.get("event") != "train_log" or event.get("loss") is None:
            return
        self._step.append(event.get("step", len(self._step) + 1))
        self._loss.append(event.get("loss"))
        self._grad_norm.append(event.get("grad_norm"))
        self._buffered += 1
        if self._buffered < self.update_every:
            return
        self._buffered = 0
        with self.figure.batch_update():
            self.figure.data[0].x = self._step
            self.figure.data[0].y = self._loss
            self.figure.data[1].x = self._step
            self.figure.data[1].y = self._grad_norm
            model_id, recipe_id = event.get("model_id"), event.get("recipe_id")
            step, max_steps = event.get("step"), event.get("max_steps")
            label = f" -- {model_id}/{recipe_id}" if model_id else ""
            progress = f" ({step}/{max_steps})" if max_steps else f" (step {step})" if step else ""
            self.figure.layout.title = f"QLoRA training{label}{progress}"


class BiPOConvergencePlot:
    """Live plot for the `optimized` (BiPO-lite) extraction method's
    gradient-step loss, distinct from the grid-cell progress in
    SteeringGridPlot -- this is the one extraction method with a real
    per-step optimization loop rather than a single closed-form pass."""

    def __init__(self, update_every: int = 1, title: str = "BiPO-lite vector optimization"):
        self.update_every = max(1, update_every)
        self._buffered = 0
        self._step: List[int] = []
        self._loss: List[Optional[float]] = []
        self._grad_norm: List[Optional[float]] = []

        self.figure = go.FigureWidget(
            data=[
                go.Scatter(x=[], y=[], mode="lines+markers", name="loss"),
                go.Scatter(x=[], y=[], mode="lines+markers", name="grad norm", yaxis="y2"),
            ],
            layout=go.Layout(
                title=title,
                xaxis=dict(title="optimizer step"),
                yaxis=dict(title="loss"),
                yaxis2=dict(title="grad norm", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                height=380,
                margin=dict(t=60, b=40),
            ),
        )

    def on_event(self, event: Dict[str, Any]) -> None:
        if event.get("event") != "opt_step":
            return
        self._step.append(event.get("i", len(self._step) + 1))
        self._loss.append(event.get("loss"))
        self._grad_norm.append(event.get("grad_norm"))
        self._buffered += 1
        if self._buffered < self.update_every:
            return
        self._buffered = 0
        with self.figure.batch_update():
            self.figure.data[0].x = self._step
            self.figure.data[0].y = self._loss
            self.figure.data[1].x = self._step
            self.figure.data[1].y = self._grad_norm
            i, n = event.get("i"), event.get("n")
            suffix = f" -- step {i}/{n}" if i and n else ""
            self.figure.layout.title = f"BiPO-lite vector optimization{suffix}"
