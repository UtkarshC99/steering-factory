from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:60] or "topic"


@dataclass
class VectorRecord:
    id: str
    topic: str
    model_name: str
    layer_idx: int
    method: str
    pooling: str
    num_pairs: int
    created_at: float
    vector_file: str
    best_positive_coefficient: Optional[float] = None
    best_negative_coefficient: Optional[float] = None
    convergence_summary: Optional[Dict[str, Any]] = None
    sweep_summary: Optional[List[Dict[str, Any]]] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Flat-file store: one .pt tensor per vector plus a single index.json
    registry. Nothing fancy, fully local, easy to inspect/back up/diff."""

    def __init__(self, vectors_dir: str):
        self.vectors_dir = vectors_dir
        os.makedirs(self.vectors_dir, exist_ok=True)
        self.index_path = os.path.join(self.vectors_dir, "index.json")
        if not os.path.exists(self.index_path):
            self._write_index([])

    def _read_index(self) -> List[dict]:
        with open(self.index_path, "r") as f:
            return json.load(f)

    def _write_index(self, records: List[dict]):
        with open(self.index_path, "w") as f:
            json.dump(records, f, indent=2)

    def save(
        self,
        topic: str,
        model_name: str,
        layer_idx: int,
        method: str,
        pooling: str,
        num_pairs: int,
        vector: torch.Tensor,
        best_positive_coefficient: Optional[float] = None,
        best_negative_coefficient: Optional[float] = None,
        convergence_summary: Optional[Dict[str, Any]] = None,
        sweep_summary: Optional[List[Dict[str, Any]]] = None,
        notes: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> VectorRecord:
        vec_id = uuid.uuid4().hex[:10]
        fname = f"{_slugify(topic)}__{method}__L{layer_idx}__{vec_id}.pt"
        fpath = os.path.join(self.vectors_dir, fname)
        torch.save(vector.detach().cpu(), fpath)

        # Trim sweep_summary text fields for the index (full text lives in
        # extra if you really want it -- index.json stays small/diffable).
        trimmed_sweep = None
        if sweep_summary:
            trimmed_sweep = [
                {k: v for k, v in row.items() if k != "text"} for row in sweep_summary
            ]

        record = VectorRecord(
            id=vec_id,
            topic=topic,
            model_name=model_name,
            layer_idx=layer_idx,
            method=method,
            pooling=pooling,
            num_pairs=num_pairs,
            created_at=time.time(),
            vector_file=fname,
            best_positive_coefficient=best_positive_coefficient,
            best_negative_coefficient=best_negative_coefficient,
            convergence_summary=convergence_summary,
            sweep_summary=trimmed_sweep,
            notes=notes,
            extra=extra or {},
        )

        records = self._read_index()
        records.append(asdict(record))
        self._write_index(records)
        return record

    def list(self, topic_filter: Optional[str] = None, model_filter: Optional[str] = None) -> List[dict]:
        records = self._read_index()
        if topic_filter:
            records = [r for r in records if topic_filter.lower() in r["topic"].lower()]
        if model_filter:
            records = [r for r in records if model_filter.lower() in r["model_name"].lower()]
        return sorted(records, key=lambda r: r["created_at"], reverse=True)

    def get(self, vec_id: str) -> Optional[dict]:
        for r in self._read_index():
            if r["id"] == vec_id:
                return r
        return None

    def load_vector(self, vec_id: str) -> Optional[torch.Tensor]:
        record = self.get(vec_id)
        if not record:
            return None
        path = os.path.join(self.vectors_dir, record["vector_file"])
        return torch.load(path, map_location="cpu")

    def delete(self, vec_id: str):
        records = self._read_index()
        keep = [r for r in records if r["id"] != vec_id]
        removed = [r for r in records if r["id"] == vec_id]
        for r in removed:
            path = os.path.join(self.vectors_dir, r["vector_file"])
            if os.path.exists(path):
                os.remove(path)
        self._write_index(keep)
