"""Manifest loading, dotted overrides, and validation for experiment runs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

REQUIRED_TOP_LEVEL = {"experiment", "artifacts", "models", "recipes"}


def _parse_value(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(manifest: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(manifest)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be path=value, got '{item}'.")
        path, raw = item.split("=", 1)
        cursor = result
        parts = path.split(".")
        if not all(parts):
            raise ValueError(f"Invalid override path '{path}'.")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = _parse_value(raw)
    return result


def validate_manifest(manifest: Dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL - set(manifest)
    if missing:
        raise ValueError(f"Manifest missing top-level keys: {sorted(missing)}")
    if not isinstance(manifest["models"], list) or not manifest["models"]:
        raise ValueError("models must be a non-empty list")
    if not isinstance(manifest["recipes"], list) or not manifest["recipes"]:
        raise ValueError("recipes must be a non-empty list")
    if not manifest["artifacts"].get("root"):
        raise ValueError("artifacts.root is required")
    for recipe in manifest["recipes"]:
        for field in ("id", "behavior", "dataset"):
            if field not in recipe:
                raise ValueError(f"Recipe missing '{field}': {recipe}")


def manifest_hash(manifest: Dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_manifest(path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    result = apply_overrides(raw, overrides)
    validate_manifest(result)
    return result
