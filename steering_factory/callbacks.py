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
