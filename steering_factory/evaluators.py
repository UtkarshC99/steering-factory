"""Deterministic evaluators used by recipes; judge integration is explicit."""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Dict, Iterable, List


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


def exact_match(prediction: str, target: str) -> Dict[str, Any]:
    return {"exact_match": float(_norm(prediction) == _norm(target))}


def classification_score(prediction: str, target: str, allowed_labels: Iterable[str] | None = None) -> Dict[str, Any]:
    pred = _norm(prediction)
    allowed = {_norm(v) for v in allowed_labels or []}
    return {"exact_match": float(pred == _norm(target)), "valid_label": float(not allowed or pred in allowed)}


def structured_score(prediction: str, schema: Dict[str, Any] | None = None, target: Any = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"parse_success": 0.0, "schema_valid": None, "required_field_coverage": None, "leaf_exact_match": None}
    try:
        value = json.loads(prediction)
        result["parse_success"] = 1.0
    except (TypeError, json.JSONDecodeError):
        return result
    if schema:
        try:
            import jsonschema
            jsonschema.validate(value, schema)
            result["schema_valid"] = 1.0
        except ImportError:
            result["schema_valid"] = None
        except Exception:
            result["schema_valid"] = 0.0
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if required and isinstance(value, dict):
            result["required_field_coverage"] = sum(k in value for k in required) / len(required)
    if target is not None:
        result["leaf_exact_match"] = float(value == target)
    return result


def abstention_score(prediction: str, answerable: bool, target: str | None = None) -> Dict[str, Any]:
    lowered = _norm(prediction)
    abstained = any(marker in lowered for marker in ("i don't know", "cannot determine", "not enough information", "unsure"))
    correct = bool(target is not None and _norm(prediction) == _norm(target))
    return {"abstained": float(abstained), "appropriate_abstention": float(abstained and not answerable),
            "false_abstention": float(abstained and answerable), "correct_answer": float(correct and answerable)}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value)):
                numeric.setdefault(key, []).append(float(value))
    return {"n": len(rows), **{f"mean_{key}": sum(values) / len(values) for key, values in numeric.items()}}


def expected_calibration_error(confidences: List[float], correctness: List[float], bins: int = 10) -> float:
    if len(confidences) != len(correctness) or not confidences:
        return float("nan")
    total, n = 0.0, len(confidences)
    for b in range(bins):
        lower, upper = b / bins, (b + 1) / bins
        indices = [i for i, c in enumerate(confidences) if lower <= c < upper or (b == bins - 1 and c == 1)]
        if indices:
            avg_c = sum(confidences[i] for i in indices) / len(indices)
            avg_a = sum(correctness[i] for i in indices) / len(indices)
            total += len(indices) / n * abs(avg_c - avg_a)
    return total
