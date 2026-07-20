"""Immutable, analysis-friendly artifact storage for experiment runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml

from .experiment_types import RunArtifact
from .manifest import manifest_hash


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_snapshot() -> Dict[str, Any]:
    gpu = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        gpu.update({"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(), "cuda": torch.version.cuda})
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "gpu": gpu, "git_revision": _git_revision()}


def fingerprint_records(records: Iterable[Dict[str, Any]]) -> str:
    canonical = "\n".join(json.dumps(r, sort_keys=True, default=str) for r in records)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ArtifactStore:
    def __init__(self, root: str | Path, manifest: Dict[str, Any], command: Optional[str] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.path = self.root / self.run_id
        self.path.mkdir()  # never reuse an existing run directory
        self.manifest = manifest
        self.artifact = RunArtifact(self.run_id, str(self.path), manifest_hash(manifest), "running", _utc_now())
        self.write_yaml("resolved_manifest.yaml", manifest)
        self.write_json("run.json", self.artifact.to_dict())
        self.write_json("environment.json", environment_snapshot())
        self.write_json("command.json", {"command": command, "started_at": self.artifact.started_at})

    def write_json(self, relative: str, payload: Any) -> Path:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return path

    def write_yaml(self, relative: str, payload: Any) -> Path:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return path

    def write_jsonl(self, relative: str, rows: Iterable[Dict[str, Any]]) -> Path:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
        return path

    def write_table(self, stem: str, rows: Iterable[Dict[str, Any]]) -> None:
        """Write JSONL always and Parquet when the runtime supports it."""
        materialized = list(rows)
        self.write_jsonl(f"{stem}.jsonl", materialized)
        try:
            import pandas as pd
            path = self.path / f"{stem}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(materialized).to_parquet(path, index=False)
        except Exception as exc:
            self.write_json(f"{stem}.parquet.unavailable.json", {"reason": str(exc)})

    def save_vector(self, name: str, vector: torch.Tensor, metadata: Dict[str, Any]) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        path = self.path / "vectors" / f"{safe}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(vector.detach().cpu(), path)
        self.write_json(f"vectors/{safe}.metadata.json", metadata)
        return path

    def finalize(self, error: Optional[BaseException] = None) -> None:
        self.artifact.status = "failed" if error else "completed"
        self.artifact.ended_at = _utc_now()
        if error:
            self.artifact.error = "".join(traceback.format_exception_only(type(error), error)).strip()
            self.write_json("failure.json", {"error": self.artifact.error, "traceback": traceback.format_exc()})
        self.write_json("run.json", self.artifact.to_dict())
