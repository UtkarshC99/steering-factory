"""Matched steering-vs-QLoRA quality/cost comparison.

This is the Pillar-2 "money question" harness: given a steering run
directory and a QLoRA run directory produced from the *same* manifest (so
splits, prompts, and scorers line up), join them on (model_id, recipe_id),
pick the best config in each arm using validation only, report matched
test-split quality alongside normalized cost, and build a data-efficiency
curve (quality vs. labeled-example count) per arm.

Both arms already write `results/generations.jsonl` rows scored by the same
`runner._behavior_score` -> `evaluators.*` functions, tagged with
`"arm": "steering"` or `"arm": "qlora"` and a `split` field. That shared
schema is what makes the join possible without re-scoring anything here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Behavior family -> the row key used as its primary "did this work" signal.
# Falls back to a generic default when a family/key isn't recognized so new
# recipes don't silently produce an empty comparison.
_QUALITY_KEY_BY_BEHAVIOR = {
    "defensive_refusal": "safe_refusal",
    "domain_classification": "exact_match",
    "appropriate_abstention": "correct_answer",
    "structured_output": "leaf_exact_match",
}
_DEFAULT_QUALITY_KEYS = ("leaf_exact_match", "exact_match", "correct_answer", "safe_refusal")


def _quality_key(behavior_id: Optional[str], rows: List[Dict[str, Any]]) -> Optional[str]:
    if behavior_id and behavior_id in _QUALITY_KEY_BY_BEHAVIOR:
        key = _QUALITY_KEY_BY_BEHAVIOR[behavior_id]
        if any(key in r for r in rows):
            return key
    for key in _DEFAULT_QUALITY_KEYS:
        if any(key in r for r in rows):
            return key
    return None


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _steering_config_key(row: Dict[str, Any]) -> Tuple:
    return (row.get("method"), row.get("layer_idx"), row.get("coefficient"), row.get("token_scope"))


def best_steering_config(
    rows: List[Dict[str, Any]], behavior_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Selects the (method, layer, coefficient, token_scope) config that
    maximizes mean quality on the VALIDATION split only, per the holdout
    discipline the README requires ("Select layers, coefficients, and stack
    weights on validation only"). Returns the winning config's held-out TEST
    quality, plus its own validation quality for transparency, or None if
    there is no validation data to select on.
    """
    quality_key = _quality_key(behavior_id, rows)
    if quality_key is None:
        return None
    val_rows = [r for r in rows if r.get("split") == "validation"]
    if not val_rows:
        return None
    by_config: Dict[Tuple, List[Dict[str, Any]]] = {}
    for row in val_rows:
        by_config.setdefault(_steering_config_key(row), []).append(row)

    scored = [(cfg, _mean(group, quality_key)) for cfg, group in by_config.items()]
    scored = [(cfg, q) for cfg, q in scored if q is not None]
    if not scored:
        return None
    best_cfg, best_val_quality = max(scored, key=lambda item: item[1])

    test_rows = [r for r in rows if r.get("split") == "test" and _steering_config_key(r) == best_cfg]
    test_quality = _mean(test_rows, quality_key)
    method, layer_idx, coefficient, token_scope = best_cfg
    return {
        "quality_key": quality_key,
        "method": method, "layer_idx": layer_idx, "coefficient": coefficient, "token_scope": token_scope,
        "validation_quality": best_val_quality, "test_quality": test_quality,
        "n_validation": len(by_config[best_cfg]), "n_test": len(test_rows),
    }


def qlora_quality(rows: List[Dict[str, Any]], behavior_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """QLoRA has exactly one trained config per (model, recipe) -- no
    layer/coefficient grid to select over -- so this just reports its
    validation and held-out test quality directly."""
    quality_key = _quality_key(behavior_id, rows)
    if quality_key is None:
        return None
    val_rows = [r for r in rows if r.get("split") == "validation"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    return {
        "quality_key": quality_key,
        "validation_quality": _mean(val_rows, quality_key),
        "test_quality": _mean(test_rows, quality_key),
        "n_validation": len(val_rows), "n_test": len(test_rows),
    }


def _steering_cost(run_dir: Path, model_id: str, recipe_id: str, vector_rows: List[Dict[str, Any]], gen_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    telemetry = _read_json(run_dir / "telemetry.json")
    own_vectors = [v for v in vector_rows if v.get("model_id") == model_id and v.get("recipe_id") == recipe_id]
    own_rows = [r for r in gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
    latencies = [r["latency_s"] for r in own_rows if r.get("latency_s") is not None]
    tokens = [r["tokens_generated"] for r in own_rows if r.get("tokens_generated")]
    ms_per_token = None
    if latencies and tokens and sum(tokens) > 0:
        ms_per_token = 1000.0 * sum(latencies) / sum(tokens)
    artifact_bytes = sum(Path(v["vector_path"]).stat().st_size for v in own_vectors if Path(v["vector_path"]).exists())
    labeled_examples = max((v.get("num_pairs", 0) for v in own_vectors), default=0)
    return {
        "arm": "steering",
        "wall_time_s": telemetry.get("wall_time_s"),
        "gpu_peak_allocated_bytes": telemetry.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": telemetry.get("gpu_peak_reserved_bytes"),
        "artifact_bytes": artifact_bytes,
        "labeled_examples": labeled_examples,
        "num_configs_evaluated": len(own_vectors),
        "per_request_ms_per_token": ms_per_token,
    }


def _qlora_cost(qlora_results: List[Dict[str, Any]], model_id: str, recipe_id: str, gen_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    match = next((r for r in qlora_results if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id), {})
    own_rows = [r for r in gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
    latencies = [r["latency_s"] for r in own_rows if r.get("latency_s") is not None]
    tokens = [r["tokens_generated"] for r in own_rows if r.get("tokens_generated")]
    ms_per_token = None
    if latencies and tokens and sum(tokens) > 0:
        ms_per_token = 1000.0 * sum(latencies) / sum(tokens)
    train_time = match.get("wall_time_s")
    eval_time = match.get("eval_wall_time_s")
    total_wall_time = (train_time or 0) + (eval_time or 0) if (train_time is not None or eval_time is not None) else None
    return {
        "arm": "qlora",
        "wall_time_s": total_wall_time,
        "train_wall_time_s": train_time,
        "eval_wall_time_s": eval_time,
        "artifact_bytes": match.get("adapter_size_bytes"),
        "labeled_examples": match.get("num_train_records"),
        "train_loss": match.get("train_loss"),
        "per_request_ms_per_token": ms_per_token,
    }


def build_comparison(steering_run: str | Path, qlora_run: str | Path) -> Dict[str, Any]:
    """Joins a steering run and a QLoRA run on (model_id, recipe_id) and
    returns the matched quality/cost report as a plain dict (JSON-safe)."""
    steer_dir = Path(steering_run)
    qlora_dir = Path(qlora_run)

    steer_gen_rows = _read_jsonl(steer_dir / "results" / "generations.jsonl")
    qlora_gen_rows = _read_jsonl(qlora_dir / "results" / "generations.jsonl")
    vector_rows = _read_jsonl(steer_dir / "vectors" / "index.jsonl")
    qlora_results = _read_json(qlora_dir / "results" / "qlora.json")
    if isinstance(qlora_results, dict):
        qlora_results = []

    behavior_by_recipe: Dict[str, str] = {}
    for row in steer_gen_rows + qlora_gen_rows:
        if row.get("recipe_id") and row.get("behavior_id"):
            behavior_by_recipe.setdefault(row["recipe_id"], row["behavior_id"])

    pairs = sorted({(r.get("model_id"), r.get("recipe_id")) for r in steer_gen_rows} &
                   {(r.get("model_id"), r.get("recipe_id")) for r in qlora_gen_rows})

    entries = []
    for model_id, recipe_id in pairs:
        behavior_id = behavior_by_recipe.get(recipe_id)
        steer_rows = [r for r in steer_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
        qlora_rows = [r for r in qlora_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]

        steer_quality = best_steering_config(steer_rows, behavior_id)
        qlora_q = qlora_quality(qlora_rows, behavior_id)
        if steer_quality is None or qlora_q is None:
            continue

        entries.append({
            "model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
            "steering": {**steer_quality, **_steering_cost(steer_dir, model_id, recipe_id, vector_rows, steer_gen_rows)},
            "qlora": {**qlora_q, **_qlora_cost(qlora_results, model_id, recipe_id, qlora_gen_rows)},
        })

    return {
        "created_at": time.time(),
        "steering_run": str(steer_dir), "qlora_run": str(qlora_dir),
        "comparisons": entries,
    }


def data_efficiency_curve(steering_run: str | Path, qlora_run: str | Path, model_id: str, recipe_id: str) -> Dict[str, Any]:
    """Quality as a function of labeled-example count for each arm, for one
    (model, recipe). Steering varies `num_pairs` implicitly through however
    many steer-split configs were extracted with different pair counts (if
    the manifest's steer split size was swept); QLoRA varies by however many
    (model, recipe) runs with different `matched_split` sizes exist across
    the supplied run. Both are returned as sorted (n_examples, quality)
    series so a caller can plot them directly.
    """
    steer_dir = Path(steering_run)
    qlora_dir = Path(qlora_run)
    steer_gen_rows = _read_jsonl(steer_dir / "results" / "generations.jsonl")
    qlora_gen_rows = _read_jsonl(qlora_dir / "results" / "generations.jsonl")
    vector_rows = _read_jsonl(steer_dir / "vectors" / "index.jsonl")
    qlora_results = _read_json(qlora_dir / "results" / "qlora.json")
    if isinstance(qlora_results, dict):
        qlora_results = []

    behavior_id = next((r.get("behavior_id") for r in steer_gen_rows
                         if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id), None)
    quality_key = _quality_key(behavior_id, steer_gen_rows + qlora_gen_rows)

    steer_points: List[Tuple[int, float]] = []
    if quality_key:
        own_vectors = {v["vector_path"]: v for v in vector_rows if v.get("model_id") == model_id and v.get("recipe_id") == recipe_id}
        by_pairs: Dict[int, List[Dict[str, Any]]] = {}
        test_rows = [r for r in steer_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id and r.get("split") == "test"]
        for v in own_vectors.values():
            n_pairs = v.get("num_pairs")
            matching = [r for r in test_rows if r.get("method") == v.get("method") and r.get("layer_idx") == v.get("layer_idx")]
            if n_pairs is not None and matching:
                by_pairs.setdefault(n_pairs, []).extend(matching)
        steer_points = sorted((n, _mean(rows, quality_key)) for n, rows in by_pairs.items() if _mean(rows, quality_key) is not None)

    qlora_points: List[Tuple[int, float]] = []
    if quality_key:
        for result in qlora_results:
            if result.get("model_id") != model_id or result.get("recipe_id") != recipe_id:
                continue
            n = result.get("num_train_records")
            matching = [r for r in qlora_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id and r.get("split") == "test"]
            q = _mean(matching, quality_key)
            if n is not None and q is not None:
                qlora_points.append((n, q))
        qlora_points = sorted(qlora_points)

    return {
        "model_id": model_id, "recipe_id": recipe_id, "quality_key": quality_key,
        "steering": [{"labeled_examples": n, "quality": q} for n, q in steer_points],
        "qlora": [{"labeled_examples": n, "quality": q} for n, q in qlora_points],
    }
