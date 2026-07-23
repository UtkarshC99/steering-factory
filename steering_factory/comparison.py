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

# Below this many held-out rows, a quality number is noise, not signal -- a
# 0-vs-1 result on a single test example proved exactly this on a real run
# (Qwen3/Gemma3 safety_refusal at n_test=1). Entries below the floor on
# EITHER split are excluded from the winner table entirely rather than
# reported as if they meant something. Overridable per call for smaller
# smoke-test fixtures, but the report-facing entry point (build_comparison)
# defaults to this.
DEFAULT_MIN_SPLIT_SIZE = 20


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

    The unsteered baseline (coefficient == 0.0) is excluded from selection:
    on a real run (Qwen3/Gemma3 safety_refusal, n_test=1) the baseline won
    validation simply because every real coefficient scored 0 on a single
    noisy example, and the base model got reported as "the steering
    result" -- which it is not. `baseline_quality` (the c=0.0 config's own
    held-out test quality, computed separately, independent of which
    config wins) and `beat_baseline` (whether the winning steered config's
    test quality exceeds it) are returned alongside the winner so a caller
    can tell "steering achieved X, doing nothing achieves Y" apart.
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

    steered_configs = {cfg: group for cfg, group in by_config.items() if cfg[2] != 0.0}
    scored = [(cfg, _mean(group, quality_key)) for cfg, group in steered_configs.items()]
    scored = [(cfg, q) for cfg, q in scored if q is not None]

    baseline_cfg = next((cfg for cfg in by_config if cfg[2] == 0.0), None)
    baseline_test_rows = [r for r in rows if r.get("split") == "test" and baseline_cfg is not None and _steering_config_key(r) == baseline_cfg]
    baseline_quality = _mean(baseline_test_rows, quality_key) if baseline_test_rows else None

    if not scored:
        return None
    best_cfg, best_val_quality = max(scored, key=lambda item: item[1])

    test_rows = [r for r in rows if r.get("split") == "test" and _steering_config_key(r) == best_cfg]
    test_quality = _mean(test_rows, quality_key)
    beat_baseline = None
    if test_quality is not None and baseline_quality is not None:
        beat_baseline = test_quality > baseline_quality
    method, layer_idx, coefficient, token_scope = best_cfg
    return {
        "quality_key": quality_key,
        "method": method, "layer_idx": layer_idx, "coefficient": coefficient, "token_scope": token_scope,
        "validation_quality": best_val_quality, "test_quality": test_quality,
        "baseline_quality": baseline_quality, "beat_baseline": beat_baseline,
        "n_validation": len(by_config[best_cfg]), "n_test": len(test_rows),
    }


def full_config_grid(rows: List[Dict[str, Any]], behavior_id: Optional[str]) -> List[Dict[str, Any]]:
    """Every (method, layer, coefficient, token_scope) config's held-out
    TEST quality, not just the validation-selected winner -- this is what a
    layer/coefficient response-surface plot needs (best_steering_config only
    returns the single winning point; a plot needs the whole grid to show
    how close the runner-up configs are, i.e. whether the winner is robust
    or a fluke on this data). Selection discipline is unchanged: this
    reports each config's own test rows, it does not re-select anything, so
    it never substitutes for validation-based selection -- it is a
    diagnostic surface, not a second selection path.

    Returns one row per (method, layer_idx, coefficient, token_scope) with
    its mean test quality and row count, sorted by (method, layer, coefficient)
    for stable, deterministic plot ordering.
    """
    quality_key = _quality_key(behavior_id, rows)
    if quality_key is None:
        return []
    test_rows = [r for r in rows if r.get("split") == "test"]
    by_config: Dict[Tuple, List[Dict[str, Any]]] = {}
    for row in test_rows:
        by_config.setdefault(_steering_config_key(row), []).append(row)

    grid = []
    for cfg, group in sorted(by_config.items(), key=lambda item: (str(item[0][0]), item[0][1] or 0, item[0][2] or 0)):
        method, layer_idx, coefficient, token_scope = cfg
        quality = _mean(group, quality_key)
        if quality is None:
            continue
        grid.append({
            "method": method, "layer_idx": layer_idx, "coefficient": coefficient,
            "token_scope": token_scope, "test_quality": quality, "n_test": len(group),
        })
    return grid


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


def _steering_cost(
    run_dir: Path, model_id: str, recipe_id: str, vector_rows: List[Dict[str, Any]],
    gen_rows: List[Dict[str, Any]], selected_config: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """`full_sweep_wall_time_s` is the entire run's telemetry wall time --
    every method/layer/coefficient/token_scope this manifest evaluated, not
    just the winning config. That is exploration cost, not deployment cost,
    and reporting it as "steering's cost" against QLoRA's one-trained-
    adapter time is a category error (a real run showed 7247s of sweep vs
    177s of one QLoRA training run, which reads as "steering is 40x
    slower" when it measured 12 configs vs 1). `selected_config_extraction_cost_s`
    amortizes the full sweep's wall time evenly across every extracted
    vector -- a rough but honest per-vector share when telemetry only
    tracks one whole-run timer -- so it can be compared to QLoRA's one
    training run on the same footing.
    """
    telemetry = _read_json(run_dir / "telemetry.json")
    own_vectors = [v for v in vector_rows if v.get("model_id") == model_id and v.get("recipe_id") == recipe_id]
    own_rows = [r for r in gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
    latencies = [r["latency_s"] for r in own_rows if r.get("latency_s") is not None]
    tokens = [r["tokens_generated"] for r in own_rows if r.get("tokens_generated")]
    ms_per_token = None
    if latencies and tokens and sum(tokens) > 0:
        ms_per_token = 1000.0 * sum(latencies) / sum(tokens)

    selected_vector = None
    if selected_config is not None:
        method, layer_idx, _coefficient, _token_scope = selected_config
        selected_vector = next(
            (v for v in own_vectors if v.get("method") == method and v.get("layer_idx") == layer_idx), None,
        )
    artifact_bytes = (
        Path(selected_vector["vector_path"]).stat().st_size
        if selected_vector is not None and Path(selected_vector["vector_path"]).exists()
        else sum(Path(v["vector_path"]).stat().st_size for v in own_vectors if Path(v["vector_path"]).exists())
    )
    labeled_examples = (
        selected_vector.get("num_pairs", 0) if selected_vector is not None
        else max((v.get("num_pairs", 0) for v in own_vectors), default=0)
    )

    full_sweep_wall_time_s = telemetry.get("wall_time_s")
    num_configs = len(own_vectors)
    selected_config_extraction_cost_s = (
        full_sweep_wall_time_s / num_configs if full_sweep_wall_time_s is not None and num_configs else None
    )

    return {
        "arm": "steering",
        "selected_config_extraction_cost_s": selected_config_extraction_cost_s,
        "full_sweep_wall_time_s": full_sweep_wall_time_s,
        "gpu_peak_allocated_bytes": telemetry.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": telemetry.get("gpu_peak_reserved_bytes"),
        "artifact_bytes": artifact_bytes,
        "labeled_examples": labeled_examples,
        "num_configs_evaluated": num_configs,
        "per_request_ms_per_token": ms_per_token,
    }


def _qlora_cost(qlora_results: List[Dict[str, Any]], model_id: str, recipe_id: str, gen_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """`gen_rows` and `qlora_results` are expected to already be filtered
    to a single (model, recipe, num_train_records) point by the caller
    (build_comparison pins every entry to the max-N adapter when an
    N-sweep produced more than one) -- this function itself has no
    N-awareness, it just reports whatever single config it's handed."""
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


def build_comparison(
    steering_run: str | Path, qlora_run: str | Path, min_split_size: int = DEFAULT_MIN_SPLIT_SIZE,
) -> Dict[str, Any]:
    """Joins a steering run and a QLoRA run on (model_id, recipe_id) and
    returns the matched quality/cost report as a plain dict (JSON-safe).

    Entries where either arm's `n_test` or `n_validation` falls below
    `min_split_size` are routed to `excluded`, not `comparisons` -- a
    quality number computed on a handful of held-out examples is noise, and
    the previous behavior of reporting it as a normal "winner" row is
    exactly how a real run turned "n_test=1" into a false 0.0-vs-1.0
    result. Pass `min_split_size=0` to disable the floor (e.g. for tests
    that intentionally exercise tiny fixtures)."""
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
    excluded = []
    for model_id, recipe_id in pairs:
        behavior_id = behavior_by_recipe.get(recipe_id)
        all_steer_rows = [r for r in steer_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
        all_qlora_rows = [r for r in qlora_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
        own_vectors = [v for v in vector_rows if v.get("model_id") == model_id and v.get("recipe_id") == recipe_id]

        # An N-sweep (experiment.n_sweep) can produce multiple vectors/
        # adapters per (model, recipe), one per labeled-example count. The
        # headline quality/cost table pins to the MAX N each arm reached --
        # the "best shot" each arm gets with the most labeled data it was
        # given -- while `data_efficiency_curve` below still reports every
        # point. Without this, best_steering_config/qlora_quality would
        # silently pool rows from every N together as if they were one
        # config, which is exactly the kind of scope-mixing Part 1 fixed
        # for cost; the same discipline applies here.
        steer_ns = {v.get("num_pairs") for v in own_vectors if v.get("num_pairs") is not None}
        max_steer_n = max(steer_ns) if steer_ns else None
        if max_steer_n is not None:
            max_n_configs = {(v.get("method"), v.get("layer_idx")) for v in own_vectors if v.get("num_pairs") == max_steer_n}
            steer_rows = [r for r in all_steer_rows if (r.get("method"), r.get("layer_idx")) in max_n_configs]
            vector_rows_at_max_n = [v for v in own_vectors if v.get("num_pairs") == max_steer_n]
        else:
            steer_rows, vector_rows_at_max_n = all_steer_rows, own_vectors

        qlora_ns = {r.get("num_train_records") for r in all_qlora_rows if r.get("num_train_records") is not None}
        max_qlora_n = max(qlora_ns) if qlora_ns else None
        qlora_rows = (
            [r for r in all_qlora_rows if r.get("num_train_records") == max_qlora_n]
            if max_qlora_n is not None else all_qlora_rows
        )
        qlora_results_at_max_n = (
            [r for r in qlora_results if r.get("num_train_records") == max_qlora_n]
            if max_qlora_n is not None else qlora_results
        )

        steer_quality = best_steering_config(steer_rows, behavior_id)
        qlora_q = qlora_quality(qlora_rows, behavior_id)
        if steer_quality is None or qlora_q is None:
            continue

        min_n = min(steer_quality["n_test"], steer_quality["n_validation"], qlora_q["n_test"], qlora_q["n_validation"])
        if min_n < min_split_size:
            excluded.append({
                "model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                "reason": "insufficient_data", "min_split_size": min_split_size,
                "n_test_steering": steer_quality["n_test"], "n_validation_steering": steer_quality["n_validation"],
                "n_test_qlora": qlora_q["n_test"], "n_validation_qlora": qlora_q["n_validation"],
            })
            continue

        selected_config = (steer_quality["method"], steer_quality["layer_idx"], steer_quality["coefficient"], steer_quality["token_scope"])
        entries.append({
            "model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
            "steering": {**steer_quality, **_steering_cost(steer_dir, model_id, recipe_id, vector_rows_at_max_n, steer_rows, selected_config)},
            "qlora": {**qlora_q, **_qlora_cost(qlora_results_at_max_n, model_id, recipe_id, qlora_rows)},
            "config_grid": full_config_grid(steer_rows, behavior_id),
            "data_efficiency": data_efficiency_curve(steer_dir, qlora_dir, model_id, recipe_id),
        })

    return {
        "created_at": time.time(),
        "steering_run": str(steer_dir), "qlora_run": str(qlora_dir),
        "min_split_size": min_split_size,
        "comparisons": entries,
        "excluded": excluded,
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
            # Filtering by num_train_records too (not just model_id/recipe_id)
            # matters once an N-sweep trains multiple adapters for the same
            # (model, recipe): every adapter's eval rows carry the same
            # model_id/recipe_id, so without this an N=512 adapter's quality
            # would get averaged in as if it were the N=8 adapter's point.
            matching = [r for r in qlora_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id
                        and r.get("split") == "test" and r.get("num_train_records") == n]
            q = _mean(matching, quality_key)
            if n is not None and q is not None:
                qlora_points.append((n, q))
        qlora_points = sorted(qlora_points)

    return {
        "model_id": model_id, "recipe_id": recipe_id, "quality_key": quality_key,
        "steering": [{"labeled_examples": n, "quality": q} for n, q in steer_points],
        "qlora": [{"labeled_examples": n, "quality": q} for n, q in qlora_points],
    }
