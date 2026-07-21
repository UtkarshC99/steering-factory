"""Optional progress-event callbacks for the runner and finetune modules.

This module has no plotting dependencies -- it's safe to import from the
hot path of `run_steering`/`run_qlora`/etc. unconditionally. Live plotting
lives in `live_plot.py`, which imports plotly only when actually used (by
the notebook, in-process). The CLI never constructs a callback, so every
`callback=None` default keeps the subprocess/CLI path byte-for-byte
unchanged from before this module existed.

Every runner entrypoint that accepts a `callback` calls `.on_event(dict)`
at points where it already has the data in hand -- this is a cheap emit,
not a restructuring of the loops it's threaded through.
"""
from __future__ import annotations

import logging
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RunCallback(Protocol):
    def on_event(self, event: Dict[str, Any]) -> None: ...


def _safe(callback: Optional[RunCallback], event: Dict[str, Any]) -> None:
    """Call `callback.on_event(event)` if callback is not None, swallowing
    and logging any exception -- a bug in a progress/plotting callback must
    never fail the underlying experiment."""
    if callback is None:
        return
    try:
        callback.on_event(event)
    except Exception:
        logger.exception("Callback raised while handling event %r; ignoring.", event.get("event"))


class JsonlCallback:
    """Appends every event to `<run>/progress/<arm>.jsonl` via the run's own
    ArtifactStore. Cheap, dependency-free, and always safe to attach --
    leaves a partial trace on disk even if the run later crashes, which a
    single write-once-at-finalize summary cannot do."""

    def __init__(self, store, arm: str):
        self.store = store
        self.arm = arm
        self._relative = f"progress/{arm}.jsonl"

    def on_event(self, event: Dict[str, Any]) -> None:
        self.store.append_jsonl(self._relative, event)


class CompositeCallback:
    """Fans one event out to several callbacks. Each child is isolated: one
    child raising (e.g. a live plot mid-redraw) never stops the others or
    the run itself."""

    def __init__(self, callbacks: Iterable[RunCallback]):
        self.callbacks: List[RunCallback] = list(callbacks)

    def on_event(self, event: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            _safe(callback, event)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


class PrintSummaryCallback:
    """Prints a plain-text summary of the previous block every k events,
    to `print()` (plain Colab/Jupyter cell output).

    This exists as a fallback that doesn't depend on ipywidgets rendering
    correctly -- `plotly.FigureWidget` requires Colab's custom widget
    manager to be enabled (`google.colab.output.enable_custom_widget_manager()`)
    and can silently render blank if it isn't, or on a widget-manager
    version mismatch. Plain `print` output in cell scrollback has no such
    failure mode, so this is meant to run *alongside* the live plots
    (attach both to the same `CompositeCallback`), not replace them.

    Groups by the event kinds the runner/finetune modules already emit:
    "generation" (steering grid), "train_log" (QLoRA), "opt_step"
    (BiPO-lite extraction). Each block's summary covers only the events
    since the previous summary, not a running total, so it reflects what
    just happened rather than smoothing it away.
    """

    def __init__(self, every: int = 10):
        self.every = max(1, every)
        self._counts: Dict[str, int] = {}
        self._block_started: Dict[str, float] = {}
        self._buffers: Dict[str, Dict[str, List[Optional[float]]]] = {}

    def _buffer_for(self, kind: str) -> Dict[str, List[Optional[float]]]:
        return self._buffers.setdefault(kind, {})

    def on_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("event")
        handler = {
            "generation": self._on_generation,
            "train_log": self._on_train_log,
            "opt_step": self._on_opt_step,
        }.get(kind)
        if handler is None:
            return
        handler(event)

    def _tick(self, kind: str, event: Dict[str, Any], fields: Dict[str, Optional[float]], header: str) -> None:
        buf = self._buffer_for(kind)
        for key, value in fields.items():
            buf.setdefault(key, []).append(value)
        count = self._counts.get(kind, 0) + 1
        self._counts[kind] = count
        self._block_started.setdefault(kind, time.perf_counter())
        if count % self.every != 0:
            return
        elapsed = time.perf_counter() - self._block_started[kind]
        rate = self.every / elapsed if elapsed > 0 else None
        summary = ", ".join(f"mean {key}={_fmt(_mean(values))}" for key, values in buf.items())
        rate_str = f", {rate:.2f}/s" if rate else ""
        print(f"[{header}] steps {count - self.every + 1}-{count}: {summary}{rate_str}")
        buf.clear()
        self._block_started[kind] = time.perf_counter()

    def _on_generation(self, event: Dict[str, Any]) -> None:
        i, n = event.get("i"), event.get("n")
        progress = f"{i}/{n} " if i and n else ""
        header = f"steering {progress}model={event.get('model_id')} recipe={event.get('recipe_id')} method={event.get('method')}"
        self._tick("generation", event, {
            "perplexity_ratio_vs_baseline": event.get("perplexity_ratio_vs_baseline"),
            "js_divergence_vs_baseline": event.get("js_divergence_vs_baseline"),
        }, header)

    def _on_train_log(self, event: Dict[str, Any]) -> None:
        step, max_steps = event.get("step"), event.get("max_steps")
        progress = f"{step}/{max_steps} " if step and max_steps else f"step {step} " if step else ""
        header = f"qlora {progress}model={event.get('model_id')} recipe={event.get('recipe_id')}"
        self._tick("train_log", event, {
            "loss": event.get("loss"),
            "grad_norm": event.get("grad_norm"),
        }, header)

    def _on_opt_step(self, event: Dict[str, Any]) -> None:
        i, n = event.get("i"), event.get("n")
        progress = f"{i}/{n} " if i and n else ""
        header = f"bipo-lite {progress}"
        self._tick("opt_step", event, {
            "loss": event.get("loss"),
            "grad_norm": event.get("grad_norm"),
        }, header)
