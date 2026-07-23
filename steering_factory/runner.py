"""High-level orchestration that is usable from a CLI, notebook, or Python."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .artifacts import ArtifactStore, fingerprint_records
from .behaviors import get_behavior
from .callbacks import RunCallback, _safe
from .datasets import build_adapter, leakage_report, stable_split
from .experiment_types import SplitPlan
from .evaluators import aggregate, safety_metric_bundle
from .evaluators import abstention_score, classification_score, refusal_score_judge, refusal_score_substring, structured_score

# behavior_ids scored as safe-refusal-vs-unsafe-compliance -- the original
# defensive_refusal plus the safety sub-taxonomy added in behaviors.py.
# Listed explicitly rather than by family lookup so a custom/user-defined
# safety behavior (behaviors.get_behavior's escape hatch) isn't silently
# scored this way just because its family happens to be "safety".
_REFUSAL_SCORED_BEHAVIORS = frozenset({
    "defensive_refusal", "harmful_instruction_compliance", "jailbreak_susceptibility",
    "deception_sycophancy", "pii_leakage",
})


def _load_normalized_records(manifest: Dict[str, Any], store: ArtifactStore) -> List[Dict[str, Any]]:
    """Load, split, normalize, and persist data inside the current run."""
    all_records: List[Dict[str, Any]] = []
    for recipe in manifest["recipes"]:
        behavior_data = recipe["behavior"]
        behavior_id = behavior_data if isinstance(behavior_data, str) else behavior_data["id"]
        description = behavior_data.get("description") if isinstance(behavior_data, dict) else None
        behavior = get_behavior(behavior_id, description)
        dataset_cfg = recipe["dataset"]
        adapter = build_adapter(dataset_cfg.get("adapter", "huggingface"))
        records = adapter.load(dataset_cfg)
        split = SplitPlan(**recipe.get("splits", manifest.get("splits", {})))
        assignments = stable_split(records, split)
        report = leakage_report(records, assignments)
        normalized = []
        for record in records:
            item = dict(record)
            item["split"] = assignments[str(item.get("id", item.get("prompt")))]
            example = adapter.normalize(item, behavior)
            normalized.append({**example.__dict__, "recipe_id": recipe["id"]})
        all_records.extend(normalized)
        store.write_json(f"data/{recipe['id']}.provenance.json", {
            "adapter": adapter.name, "dataset": dataset_cfg, "behavior": behavior.__dict__,
            "fingerprint": fingerprint_records(records), "leakage": report, "split_plan": split.__dict__,
        })
    store.write_jsonl("data/examples.jsonl", all_records)
    return all_records


def prepare_data(manifest: Dict[str, Any], command: str | None = None) -> ArtifactStore:
    store = ArtifactStore(manifest["artifacts"]["root"], manifest, command)
    try:
        all_records = _load_normalized_records(manifest, store)
        store.write_jsonl("data/examples.jsonl", all_records)
        store.write_json("data/summary.json", {"records": len(all_records), "recipes": len(manifest["recipes"])})
        store.finalize()
        return store
    except Exception as error:
        store.finalize(error)
        raise


def _behavior_score(prediction: str, example: Dict[str, Any], judge: Optional[Any] = None) -> Dict[str, Any]:
    """`judge`, if given, must implement `GeneratorBackend.score_behavior`
    (generators/base.py) -- every safety-family behavior_id is then scored
    via `refusal_score_judge` instead of the cheap substring baseline
    (`refusal_score_substring`, still used when judge is None, e.g. the
    CLI/subprocess path never constructs one). Always None from the CLI;
    a caller opts in explicitly by constructing a generator backend and
    passing it through run_steering/run_evaluate/run_qlora."""
    family = example["behavior_id"]
    metadata = example.get("metadata", {})
    if family == "structured_output":
        return structured_score(prediction, metadata.get("schema"), metadata.get("target"))
    if family == "appropriate_abstention":
        return abstention_score(prediction, bool(metadata.get("answerable")), metadata.get("target"))
    if family == "domain_classification":
        return classification_score(prediction, example.get("positive", ""))
    if family in _REFUSAL_SCORED_BEHAVIORS:
        if judge is not None:
            return refusal_score_judge(prediction, judge)
        return refusal_score_substring(prediction)
    return {}


def _resolve_layers(layers: List[Any], num_layers: int) -> List[int]:
    """Named layer bands are resolved deterministically for each architecture."""
    from .model_utils import layer_index_from_fraction

    resolved = []
    for layer in layers:
        if isinstance(layer, int):
            resolved.append(layer)
        elif layer == "early_mid":
            resolved.append(layer_index_from_fraction(num_layers, 0.35))
        elif layer == "late_mid":
            resolved.append(layer_index_from_fraction(num_layers, 0.75))
        else:
            resolved.append(layer_index_from_fraction(num_layers, 0.6))
    return sorted(set(resolved))


def _count_extract_units(manifest: Dict[str, Any], examples: List[Dict[str, Any]], num_layers_by_model: Dict[str, int]) -> int:
    """Total (model, recipe, method, layer, n_sweep point) extraction units, for progress denominators."""
    recipe_map = {r["id"]: r for r in manifest["recipes"]}
    n_sweep = _n_sweep_values(manifest, default_n=None)
    total = 0
    for model_cfg in manifest["models"]:
        num_layers = num_layers_by_model.get(model_cfg.get("id"))
        for recipe_id, recipe in recipe_map.items():
            steer = [e for e in examples if e["recipe_id"] == recipe_id and e["split"] == "steer"]
            if not steer or num_layers is None:
                continue
            methods = recipe.get("extraction", ["mean_diff"])
            layers = recipe.get("application", {}).get("layers", [0.6])
            points = n_sweep or [len(steer)]
            total += len(methods) * len(_resolve_layers(layers, num_layers)) * len(points)
    return total


def _n_sweep_values(manifest: Dict[str, Any], default_n: Optional[int]) -> Optional[List[int]]:
    """`experiment.n_sweep` (e.g. [8, 16, 32, 64, 128, 256, 512]) is the
    labeled-example-count axis the comparison report's data-efficiency
    curve plots. Absent (the default) means run once at the full steer/
    train split -- identical to every manifest before this existed. When
    present, values larger than the available pool are silently dropped
    (there is nothing more to subsample) rather than raising, so one
    manifest can be reused across recipes/models whose steer splits differ
    in size."""
    raw = manifest.get("experiment", {}).get("n_sweep")
    if not raw:
        return None
    return sorted({int(n) for n in raw if n > 0})


def _subsample_n(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Deterministic (seeded) subsample of `n` items, sorted by a stable
    key first so the same seed always yields the same subsample regardless
    of the input list's incoming order (e.g. dict-iteration order across a
    rebuilt `examples` list should not change which N=8 subsample is used
    between two otherwise-identical runs)."""
    import random

    if n >= len(items):
        return list(items)
    ordered = sorted(items, key=lambda e: str(e.get("id", e.get("prompt", ""))))
    rng = random.Random(f"{seed}:n={n}")
    return rng.sample(ordered, n)


def run_extract(manifest: Dict[str, Any], command: str | None = None, callback: Optional[RunCallback] = None) -> ArtifactStore:
    """Extract and persist steering vectors only -- no generation/evaluation.

    Cheap to iterate on: swap extraction methods or layer grids without
    re-running the (much more expensive) generation sweep. `run_evaluate`
    consumes the vectors this writes under `vectors/`.

    `callback`, if given, receives one `on_event` call per extracted vector
    plus a final `"extract_done"` event. Always `None` from the CLI -- see
    `callbacks.py` for why this can never change subprocess/CLI behavior.
    """
    import torch
    from . import extraction
    from .config import TargetModelConfig
    from .model_utils import load_model

    store = ArtifactStore(manifest["artifacts"]["root"], manifest, command)
    started = time.perf_counter()
    try:
        examples = _load_normalized_records(manifest, store)
        recipe_map = {r["id"]: r for r in manifest["recipes"]}
        vector_rows: List[Dict[str, Any]] = []
        done = 0
        extraction_batch_size = manifest.get("extraction", {}).get("batch_size", 16)
        for model_cfg in manifest["models"]:
            fields = {key: value for key, value in model_cfg.items() if key in TargetModelConfig.__dataclass_fields__}
            loaded = load_model(TargetModelConfig(**fields))
            store.write_json(f"models/{model_cfg.get('id', 'model')}.json", {
                "requested": model_cfg, "resolved_layer_path": loaded.layer_path, "num_layers": loaded.num_layers,
                "hidden_size": loaded.hidden_size, "device": loaded.device,
                "model_revision": getattr(loaded.model.config, "_commit_hash", None),
                "tokenizer_revision": getattr(loaded.tokenizer, "init_kwargs", {}).get("_commit_hash"),
            })
            total = _count_extract_units(manifest, examples, {model_cfg.get("id"): loaded.num_layers})
            for recipe_id, recipe in recipe_map.items():
                recipe_examples = [e for e in examples if e["recipe_id"] == recipe_id]
                steer = [e for e in recipe_examples if e["split"] == "steer"]
                if not steer:
                    continue
                pairs = [{"prompt": e["prompt"], "compliant": e["positive"], "non_compliant": e["negative"]} for e in steer]
                methods = recipe.get("extraction", ["mean_diff"])
                grid = recipe.get("application", {})
                layers = grid.get("layers", [0.6])
                resolved_layers = _resolve_layers(layers, loaded.num_layers)
                for method in methods:
                    for layer_idx in resolved_layers:
                        result = extraction.extract(method, loaded, layer_idx, pairs, callback=callback, batch_size=extraction_batch_size)
                        metadata = {"model": model_cfg.get("name_or_path"), "model_id": model_cfg.get("id"),
                                    "hidden_size": loaded.hidden_size, "layer_idx": layer_idx, "method": method,
                                    "recipe_id": recipe_id, "num_pairs": len(pairs), "convergence": result.convergence}
                        vector_path = store.save_vector(f"{model_cfg.get('id', 'model')}__{recipe_id}__{method}__L{layer_idx}", result.vector, metadata)
                        vector_rows.append({**metadata, "vector_path": str(vector_path), "vector_norm": float(result.vector.norm())})
                        done += 1
                        _safe(callback, {"arm": "extract", "event": "vector", "i": done, "n": total,
                                          "model_id": model_cfg.get("id"), "recipe_id": recipe_id,
                                          "method": method, "layer_idx": layer_idx,
                                          "vector_norm": float(result.vector.norm())})
            del loaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        store.write_table("vectors/index", vector_rows)
        _safe(callback, {"arm": "extract", "event": "extract_done", "vectors": len(vector_rows)})
        telemetry = {"wall_time_s": time.perf_counter() - started,
                     "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                     "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                     "vectors": len(vector_rows)}
        store.write_json("telemetry.json", telemetry)
        store.finalize()
        return store
    except Exception as error:
        store.finalize(error)
        raise


def _load_vector_index(run_dir: Path) -> List[Dict[str, Any]]:
    index_path = run_dir / "vectors" / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"No vectors/index.jsonl in {run_dir}; run `extract` first.")
    rows = []
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                import json
                rows.append(json.loads(line))
    return rows


def _count_evaluate_units(vector_rows: List[Dict[str, Any]], recipe_map: Dict[str, Any], examples: List[Dict[str, Any]]) -> int:
    """Total (vector, token_scope, example, coefficient) generation units."""
    total = 0
    for vrow in vector_rows:
        recipe = recipe_map.get(vrow["recipe_id"])
        if recipe is None:
            continue
        evaluate_examples = [e for e in examples if e["recipe_id"] == vrow["recipe_id"] and e["split"] in ("validation", "test")]
        if not evaluate_examples:
            continue
        grid = recipe.get("application", {})
        coefficients = grid.get("coefficients", [0.0, 1.0])
        token_scopes = grid.get("token_scopes", ["all"])
        n_coeffs = len(sorted(set([float(c) for c in coefficients] + [0.0])))
        total += len(token_scopes) * len(evaluate_examples) * n_coeffs
    return total


def _evaluate_vector_grid(
    loaded: Any,
    vector: Any,
    layer_idx: int,
    method: str,
    recipe_id: str,
    model_id: Optional[str],
    model_name: Optional[str],
    evaluate_examples: List[Dict[str, Any]],
    coefficients: List[float],
    token_scopes: List[str],
    max_new_tokens: int,
    batch_size: int,
    callback: Optional[RunCallback],
    progress: Dict[str, int],
    judge: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Generates and scores every (token_scope, coefficient, example) row
    for one already-extracted vector, batching examples together at each
    fixed (token_scope, coefficient) -- the axis every row in a batch can
    share, since one `SteeringHook` carries one coefficient/vector for its
    whole active scope (see sweep.generate_with_steering_batch). This is
    the single implementation shared by `run_evaluate` and `run_steering`,
    which used to duplicate this loop; extracted so a future change to the
    grid/batching logic can't drift between the two.

    Emits one `"generation"` callback event PER ROW (not per batch), so
    live plots and PrintSummaryCallback's cadence are unaffected by
    batch_size -- `progress` is a shared mutable {"done": int, "total": int}
    dict so callers can track one running count across multiple vectors.

    `latency_s` on each row is amortized from the batch's measured wall
    time (`batch_wall_time_s / batch_size`); both are kept on the row so
    downstream cost analysis (e.g. the steering-vs-QLoRA comparison) can
    tell measured-per-row apart from amortized-per-row rather than
    silently trusting the amortized number as a per-row measurement."""
    from . import metrics, sweep

    rows: List[Dict[str, Any]] = []
    all_coefficients = sorted(set([float(c) for c in coefficients] + [0.0]))
    prompts = [e["prompt"] for e in evaluate_examples]

    for token_scope in token_scopes:
        # Baseline (coefficient 0.0) is computed once per example per
        # token_scope and reused for every other coefficient's
        # perplexity-ratio/JS-divergence-vs-baseline -- unchanged from the
        # pre-batching loop's behavior, just computed for the whole batch
        # of examples in one call instead of one example at a time.
        baseline_results = sweep.generate_with_steering_batch(
            loaded, layer_idx, vector, 0.0, prompts, max_new_tokens, token_scope,
        )
        results_by_coefficient: Dict[float, List] = {0.0: baseline_results}
        for coefficient in all_coefficients:
            if coefficient == 0.0:
                continue
            results_by_coefficient[coefficient] = sweep.generate_with_steering_batch(
                loaded, layer_idx, vector, coefficient, prompts, max_new_tokens, token_scope,
            )

        for example_idx, example in enumerate(evaluate_examples):
            baseline = baseline_results[example_idx]
            baseline_ppl = metrics.perplexity_from_logprobs(baseline.token_logprobs)
            for coefficient in all_coefficients:
                result = results_by_coefficient[coefficient][example_idx]
                text, logp, probs = result.text, result.token_logprobs, result.first_step_probs
                score = _behavior_score(text, example, judge)
                ppl = metrics.perplexity_from_logprobs(logp)
                example_metadata = example.get("metadata", {}) or {}
                row = {"model_id": model_id, "model_name": model_name, "arm": "steering",
                       "recipe_id": recipe_id, "behavior_id": example["behavior_id"], "example_id": example["id"],
                       "split": example["split"], "category": example.get("category"), "method": method,
                       "layer_idx": layer_idx, "coefficient": coefficient, "token_scope": token_scope, "prompt": example["prompt"],
                       "output": text, "latency_s": result.latency_s, "batch_size": result.batch_size,
                       "batch_wall_time_s": result.batch_wall_time_s,
                       "tokens_generated": len(logp), "perplexity": ppl,
                       "perplexity_ratio_vs_baseline": (ppl / baseline_ppl) if baseline_ppl and baseline_ppl > 0 else None,
                       "js_divergence_vs_baseline": metrics.js_divergence(probs, baseline.first_step_probs) if probs is not None and baseline.first_step_probs is not None else None,
                       "repetition_score": metrics.repetition_score(text), "distinct_2": metrics.distinct_n(text, 2),
                       "benchmark": example_metadata.get("benchmark"), "is_safe_control": example_metadata.get("is_safe_control"),
                       **score}
                rows.append(row)
                progress["done"] += 1
                _safe(callback, {"arm": "steering", "event": "generation", "i": progress["done"], "n": progress["total"],
                                  "model_id": model_id, "recipe_id": recipe_id, "method": method,
                                  "layer_idx": layer_idx, "coefficient": coefficient, "token_scope": token_scope,
                                  "perplexity_ratio_vs_baseline": row["perplexity_ratio_vs_baseline"],
                                  "js_divergence_vs_baseline": row["js_divergence_vs_baseline"]})
    return rows


def _run_capability_probe_for_vector(
    loaded: Any, vector: Any, layer_idx: int, method: str, recipe_id: str,
    model_id: Optional[str], model_name: Optional[str], coefficients: List[float], token_scope: str,
    max_new_tokens: int, base_cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Runs the benign capability probe (capability_probe.py) at every
    coefficient already being evaluated for this vector (reusing the same
    grid `_evaluate_vector_grid` used, not a separately-configured one),
    plus the base model once per model_id (cached in `base_cache` across
    every recipe/vector for that model, since the base model's own
    capability never changes across recipes -- no reason to re-measure it
    once per vector).

    Returns rows keyed the same way `full_config_grid` keys the quality
    grid -- (model_id, recipe_id, method, layer_idx, coefficient) -- so
    comparison.py can join capability retention onto the same config a
    quality point belongs to, without a separate lookup scheme.
    """
    from .capability_probe import capability_quality, capability_retention, run_capability_probe_steering

    rows: List[Dict[str, Any]] = []
    cache_key = model_id or "model"
    if cache_key not in base_cache:
        base_cache[cache_key] = run_capability_probe_steering(
            loaded, vector, layer_idx, 0.0, token_scope, max_new_tokens,
        )
    base_rows = base_cache[cache_key]
    base_quality = capability_quality(base_rows)

    for coefficient in coefficients:
        steered_rows = (
            base_rows if coefficient == 0.0
            else run_capability_probe_steering(loaded, vector, layer_idx, coefficient, token_scope, max_new_tokens)
        )
        rows.append({
            "model_id": model_id, "model_name": model_name, "recipe_id": recipe_id,
            "method": method, "layer_idx": layer_idx, "coefficient": coefficient, "token_scope": token_scope,
            "base_capability_quality": base_quality,
            "steered_capability_quality": capability_quality(steered_rows),
            "capability_retention": capability_retention(base_rows, steered_rows),
            "n_probes": len(steered_rows),
        })
    return rows


def run_evaluate(
    manifest: Dict[str, Any], vectors_run: str | Path, command: str | None = None,
    callback: Optional[RunCallback] = None, judge: Optional[Any] = None,
) -> ArtifactStore:
    """Load vectors saved by `run_extract` and execute the generation/eval grid.

    `vectors_run` is the artifact directory produced by a prior `extract`
    (or `run`) invocation. This lets you re-evaluate the same vectors under
    a different coefficient/token-scope grid without re-extracting them.

    `callback`, if given, receives one `on_event` call per generated row
    (with the perplexity-ratio/JS-divergence already computed for that
    row) plus a final `"evaluate_done"` event. Always `None` from the CLI.

    `judge`, if given (any `generators.base.GeneratorBackend`), scores
    every safety-family behavior via `refusal_score_judge` instead of the
    cheap substring baseline -- see `_behavior_score`. Always `None` from
    the CLI; a caller opts in explicitly.
    """
    import torch

    source = Path(vectors_run)
    vector_rows = _load_vector_index(source)
    store = ArtifactStore(manifest["artifacts"]["root"], manifest, command)
    started = time.perf_counter()
    try:
        examples = _load_normalized_records(manifest, store)
        recipe_map = {r["id"]: r for r in manifest["recipes"]}
        progress = {"done": 0, "total": _count_evaluate_units(vector_rows, recipe_map, examples)}
        rows: List[Dict[str, Any]] = []
        loaded_models: Dict[str, Any] = {}
        decoding_cfg = manifest.get("decoding", {})
        max_new_tokens = decoding_cfg.get("max_new_tokens", 96)
        batch_size = decoding_cfg.get("batch_size", 16)

        def _get_loaded(model_id: str, model_name: str):
            if model_id not in loaded_models:
                from .config import TargetModelConfig
                from .model_utils import load_model
                model_cfg = next((m for m in manifest["models"] if m.get("id") == model_id), {"name_or_path": model_name})
                fields = {key: value for key, value in model_cfg.items() if key in TargetModelConfig.__dataclass_fields__}
                loaded_models[model_id] = load_model(TargetModelConfig(**fields))
            return loaded_models[model_id]

        for vrow in vector_rows:
            recipe_id = vrow["recipe_id"]
            recipe = recipe_map.get(recipe_id)
            if recipe is None:
                continue
            loaded = _get_loaded(vrow["model_id"], vrow["model"])
            vector_path = Path(vrow["vector_path"])
            if not vector_path.is_absolute():
                vector_path = source / vector_path.relative_to(source) if str(vector_path).startswith(str(source)) else vector_path
            vector = torch.load(vector_path, map_location="cpu")

            recipe_examples = [e for e in examples if e["recipe_id"] == recipe_id]
            evaluate_examples = [e for e in recipe_examples if e["split"] in ("validation", "test")]
            if not evaluate_examples:
                continue
            grid = recipe.get("application", {})
            coefficients = grid.get("coefficients", [0.0, 1.0])
            token_scopes = grid.get("token_scopes", ["all"])
            layer_idx = vrow["layer_idx"]
            method = vrow["method"]

            rows.extend(_evaluate_vector_grid(
                loaded, vector, layer_idx, method, recipe_id, vrow["model_id"], vrow["model"],
                evaluate_examples, coefficients, token_scopes, max_new_tokens, batch_size, callback, progress, judge,
            ))
        del loaded_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        store.write_table("results/generations", rows)
        store.write_json("results/summary.json", aggregate(rows))
        if any("safe_refusal" in row for row in rows):
            store.write_json("results/safety_summary.json", safety_metric_bundle(rows))
        store.write_json("results/vectors_source.json", {"source_run": str(source)})
        _safe(callback, {"arm": "steering", "event": "evaluate_done", "rows": len(rows)})
        telemetry = {"wall_time_s": time.perf_counter() - started,
                     "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                     "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                     "rows": len(rows)}
        store.write_json("telemetry.json", telemetry)
        store.finalize()
        return store
    except Exception as error:
        store.finalize(error)
        raise


def run_steering(
    manifest: Dict[str, Any], command: str | None = None, callback: Optional[RunCallback] = None,
    judge: Optional[Any] = None,
) -> ArtifactStore:
    """Execute the full extract + evaluate grid in one artifact, and persist
    raw, scored outputs. Convenience wrapper around `run_extract` followed
    by `run_evaluate` against the vectors it just wrote, kept as a single
    artifact directory for backward compatibility with existing manifests
    and the `run` CLI command.

    This deliberately evaluates small manifest grids serially: results are
    auditable and Colab users can reduce models/recipes with dotted overrides.

    `callback`, if given, receives `"vector"` events (one per extracted
    vector) and `"generation"` events (one per generated+scored row), each
    carrying an `i`/`n` progress pair. The CLI never passes one, so this
    param existing has no effect on the `python -m steering_factory run`
    subprocess path -- see `callbacks.py`.

    `judge`, if given (any `generators.base.GeneratorBackend`), scores
    every safety-family behavior via `refusal_score_judge` instead of the
    cheap substring baseline -- see `_behavior_score`. Always `None` from
    the CLI; a caller opts in explicitly.
    """
    import torch
    from . import extraction
    from .config import TargetModelConfig
    from .model_utils import load_model

    store = ArtifactStore(manifest["artifacts"]["root"], manifest, command)
    started = time.perf_counter()
    try:
        examples = _load_normalized_records(manifest, store)
        recipe_map = {r["id"]: r for r in manifest["recipes"]}
        rows: List[Dict[str, Any]] = []
        vector_rows: List[Dict[str, Any]] = []
        capability_rows: List[Dict[str, Any]] = []
        vectors_done = 0
        generations_done = 0
        decoding_cfg = manifest.get("decoding", {})
        max_new_tokens = decoding_cfg.get("max_new_tokens", 96)
        batch_size = decoding_cfg.get("batch_size", 16)
        extraction_batch_size = manifest.get("extraction", {}).get("batch_size", 16)
        # Opt-in (default off): probing every extracted vector's full
        # coefficient grid multiplies generation cost roughly by
        # len(CAPABILITY_PROBES) / len(evaluate_examples) on top of the
        # existing sweep -- proportional, not a new order of magnitude, but
        # real cost a manifest should choose deliberately rather than pay
        # by default on every run written before this existed.
        capability_cfg = manifest.get("capability_probe", {})
        capability_probe_enabled = bool(capability_cfg.get("enabled", False))
        probe_max_new_tokens = capability_cfg.get("max_new_tokens", 24)
        base_capability_cache: Dict[str, List[Dict[str, Any]]] = {}
        for model_cfg in manifest["models"]:
            fields = {key: value for key, value in model_cfg.items() if key in TargetModelConfig.__dataclass_fields__}
            loaded = load_model(TargetModelConfig(**fields))
            store.write_json(f"models/{model_cfg.get('id', 'model')}.json", {
                "requested": model_cfg, "resolved_layer_path": loaded.layer_path, "num_layers": loaded.num_layers,
                "hidden_size": loaded.hidden_size, "device": loaded.device,
                "model_revision": getattr(loaded.model.config, "_commit_hash", None),
                "tokenizer_revision": getattr(loaded.tokenizer, "init_kwargs", {}).get("_commit_hash"),
            })
            vectors_total = _count_extract_units(manifest, examples, {model_cfg.get("id"): loaded.num_layers})
            for recipe_id, recipe in recipe_map.items():
                recipe_examples = [e for e in examples if e["recipe_id"] == recipe_id]
                steer = [e for e in recipe_examples if e["split"] == "steer"]
                evaluate = [e for e in recipe_examples if e["split"] in ("validation", "test")]
                if not steer or not evaluate:
                    continue
                methods = recipe.get("extraction", ["mean_diff"])
                grid = recipe.get("application", {})
                coefficients = grid.get("coefficients", [0.0, 1.0])
                token_scopes = grid.get("token_scopes", ["all"])
                layers = grid.get("layers", [0.6])
                resolved_layers = _resolve_layers(layers, loaded.num_layers)
                all_coefficients_for_probe = sorted(set([float(c) for c in coefficients] + [0.0]))
                n_coeffs = len(all_coefficients_for_probe)
                # generations_total is the per-recipe denominator (unchanged
                # from before batching existed) -- generations_done keeps
                # counting across the whole run, so "i" is monotonic but
                # "n" changes when a new recipe starts. Preserved as-is;
                # not something this batching change is meant to alter.
                generations_total = len(resolved_layers) * len(methods) * len(token_scopes) * len(evaluate) * n_coeffs

                # n_points is the labeled-example-count axis (experiment.n_sweep);
                # absent, this is exactly [len(steer)] -- one iteration using
                # every steer example, identical to every manifest before
                # n_sweep existed. Only the STEER/extraction pool is swept;
                # validation/test (`evaluate`) are never subsampled, so
                # quality at each N is always measured on the same held-out
                # data -- N is purely "how much labeled data built this vector."
                n_points = _n_sweep_values(manifest, default_n=None) or [len(steer)]
                for n_value in n_points:
                    steer_subset = _subsample_n(steer, n_value, manifest.get("splits", {}).get("seed", 17))
                    pairs = [{"prompt": e["prompt"], "compliant": e["positive"], "non_compliant": e["negative"]} for e in steer_subset]
                    for method in methods:
                        for layer_idx in resolved_layers:
                            result = extraction.extract(method, loaded, layer_idx, pairs, callback=callback, batch_size=extraction_batch_size)
                            metadata = {"model": model_cfg.get("name_or_path"), "model_id": model_cfg.get("id"),
                                        "hidden_size": loaded.hidden_size, "layer_idx": layer_idx, "method": method,
                                        "recipe_id": recipe_id, "num_pairs": len(pairs), "convergence": result.convergence}
                            vector_name = f"{model_cfg.get('id', 'model')}__{recipe_id}__{method}__L{layer_idx}__N{len(pairs)}"
                            vector_path = store.save_vector(vector_name, result.vector, metadata)
                            vector_rows.append({**metadata, "vector_path": str(vector_path), "vector_norm": float(result.vector.norm())})
                            vectors_done += 1
                            _safe(callback, {"arm": "steering", "event": "vector", "i": vectors_done, "n": vectors_total,
                                              "model_id": model_cfg.get("id"), "recipe_id": recipe_id,
                                              "method": method, "layer_idx": layer_idx, "num_pairs": len(pairs),
                                              "vector_norm": float(result.vector.norm())})
                            progress = {"done": generations_done, "total": generations_total * len(n_points)}
                            new_rows = _evaluate_vector_grid(
                                loaded, result.vector, layer_idx, method, recipe_id, model_cfg.get("id"), model_cfg.get("name_or_path"),
                                evaluate, coefficients, token_scopes, max_new_tokens, batch_size, callback, progress, judge,
                            )
                            rows.extend(new_rows)
                            generations_done = progress["done"]

                            if capability_probe_enabled:
                                capability_rows.extend(_run_capability_probe_for_vector(
                                    loaded, result.vector, layer_idx, method, recipe_id,
                                    model_cfg.get("id"), model_cfg.get("name_or_path"),
                                    all_coefficients_for_probe, token_scopes[0], probe_max_new_tokens,
                                    base_capability_cache,
                                ))
            del loaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        store.write_table("results/generations", rows)
        store.write_json("results/summary.json", aggregate(rows))
        if any("safe_refusal" in row for row in rows):
            store.write_json("results/safety_summary.json",
                              safety_metric_bundle(rows, capability_rows if capability_probe_enabled else None))
        store.write_table("vectors/index", vector_rows)
        if capability_probe_enabled:
            store.write_table("results/capability_probe", capability_rows)
        telemetry = {"wall_time_s": time.perf_counter() - started, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                     "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                     "rows": len(rows), "vectors": len(vector_rows)}
        store.write_json("telemetry.json", telemetry)
        _safe(callback, {"arm": "steering", "event": "run_done", "rows": len(rows), "vectors": len(vector_rows)})
        store.finalize()
        return store
    except Exception as error:
        store.finalize(error)
        raise


def run_qlora(
    manifest: Dict[str, Any], command: str | None = None, callback: Optional[RunCallback] = None,
    judge: Optional[Any] = None,
) -> ArtifactStore:
    """Train a separate matched adapter per model/recipe steer set, then
    evaluate each adapter on the same validation/test examples the steering
    arm uses, scored with the identical evaluator (`_behavior_score`). This
    is what makes `compare` able to do a matched quality/cost comparison
    instead of just diffing run metadata: both arms end up with
    `results/generations.jsonl` rows in the same schema.

    `callback`, if given, is forwarded into `train_qlora` where it receives
    one `on_log`-derived `"train_log"` event per `logging_steps` (loss,
    grad_norm, learning_rate, epoch) via a small TrainerCallback adapter,
    plus `"adapter_done"`/`"eval_generation"` events around evaluation.
    Always `None` from the CLI -- see `callbacks.py`.

    `judge`, if given (any `generators.base.GeneratorBackend`), scores
    every safety-family behavior via `refusal_score_judge` instead of the
    cheap substring baseline -- see `_behavior_score`. Always `None` from
    the CLI; a caller opts in explicitly.
    """
    import time as _time
    from .finetune import evaluate_qlora_adapter, train_qlora
    store = ArtifactStore(manifest["artifacts"]["root"], manifest, command)
    try:
        examples = _load_normalized_records(manifest, store)
        config = dict(manifest.get("finetune", {}))
        if config.get("backend", "qlora") != "qlora":
            raise ValueError("Only the qlora fine-tuning backend is currently implemented.")
        results = []
        eval_rows: List[Dict[str, Any]] = []
        decoding_cfg = manifest.get("decoding", {})
        max_new_tokens = decoding_cfg.get("max_new_tokens", 96)
        eval_batch_size = decoding_cfg.get("batch_size", 16)
        for model in manifest["models"]:
            for recipe in manifest["recipes"]:
                recipe_examples = [e for e in examples if e["recipe_id"] == recipe["id"]]
                train_records = [e for e in recipe_examples if e["split"] == config.get("matched_split", "steer")]
                eval_examples = [e for e in recipe_examples if e["split"] in ("validation", "test")]
                if not train_records:
                    continue

                # Same n_points axis as run_steering's extraction loop: only
                # the TRAIN pool is swept, eval_examples (validation/test)
                # stay fixed, so quality at each N is measured on identical
                # held-out data across the whole sweep -- and across arms,
                # matching how run_steering only subsamples `steer`.
                n_points = _n_sweep_values(manifest, default_n=None) or [len(train_records)]
                for n_value in n_points:
                    train_subset = _subsample_n(train_records, n_value, manifest.get("splits", {}).get("seed", 17))
                    output_dir = store.path / "qlora" / model["id"] / recipe["id"] / f"N{len(train_subset)}"
                    qlora_config = {**config, "model_name": model["name_or_path"], "output_dir": str(output_dir)}
                    train_result = train_qlora(train_subset, qlora_config, callback=callback,
                                                callback_context={"model_id": model["id"], "recipe_id": recipe["id"]})
                    results.append({"model_id": model["id"], "recipe_id": recipe["id"], "num_train_records": len(train_subset), **train_result})
                    _safe(callback, {"arm": "qlora", "event": "adapter_done", "model_id": model["id"], "recipe_id": recipe["id"],
                                      "num_train_records": len(train_subset),
                                      "train_loss": train_result.get("train_loss"), "global_step": train_result.get("global_step")})

                    if not eval_examples:
                        continue
                    eval_started = _time.perf_counter()
                    eval_result = evaluate_qlora_adapter(
                        eval_examples, model["name_or_path"], train_result["adapter_dir"],
                        max_new_tokens=max_new_tokens, trust_remote_code=bool(model.get("trust_remote_code", False)),
                        batch_size=eval_batch_size,
                    )
                    for row in eval_result["rows"]:
                        example = next(e for e in eval_examples if e["id"] == row["example_id"])
                        example_metadata = example.get("metadata", {}) or {}
                        score = _behavior_score(row["output"], example, judge)
                        eval_rows.append({"model_id": model["id"], "model_name": model["name_or_path"],
                                           "recipe_id": recipe["id"], "arm": "qlora",
                                           "num_train_records": len(train_subset),
                                           "benchmark": example_metadata.get("benchmark"),
                                           "is_safe_control": example_metadata.get("is_safe_control"),
                                           **row, **score})
                        _safe(callback, {"arm": "qlora", "event": "eval_generation", "model_id": model["id"],
                                          "recipe_id": recipe["id"], "example_id": row["example_id"]})
                    results[-1]["eval_wall_time_s"] = _time.perf_counter() - eval_started
                    results[-1]["eval_records"] = len(eval_examples)
        store.write_json("results/qlora.json", results)
        if eval_rows:
            store.write_table("results/generations", eval_rows)
            store.write_json("results/summary.json", aggregate(eval_rows))
            if any("safe_refusal" in row for row in eval_rows):
                store.write_json("results/safety_summary.json", safety_metric_bundle(eval_rows))
        _safe(callback, {"arm": "qlora", "event": "qlora_done", "adapters": len(results)})
        store.finalize()
        return store
    except Exception as error:
        store.finalize(error)
        raise


def write_evaluation(store: ArtifactStore, rows: List[Dict[str, Any]], name: str = "evaluation") -> Dict[str, Any]:
    """Persist per-example rows and an aggregation; called by evaluator plugins."""
    store.write_jsonl(f"results/{name}.jsonl", rows)
    summary = aggregate(rows)
    store.write_json(f"results/{name}.summary.json", summary)
    return summary


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    import json
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Render a Markdown summary from an existing, finalized run directory.

    Loads no model and touches no GPU -- purely reads what `extract`,
    `evaluate`, or `run` already wrote (run.json, results/generations.jsonl,
    results/summary.json, vectors/index.jsonl) and formats it for humans.
    """
    import json

    source = Path(run_dir)
    run_meta = json.loads((source / "run.json").read_text(encoding="utf-8")) if (source / "run.json").exists() else {}
    summary = json.loads((source / "results" / "summary.json").read_text(encoding="utf-8")) if (source / "results" / "summary.json").exists() else {}
    generations = _read_jsonl(source / "results" / "generations.jsonl")
    vector_rows = _read_jsonl(source / "vectors" / "index.jsonl")

    lines = [f"# Run report: {run_meta.get('run_id', source.name)}", "",
              f"- Status: {run_meta.get('status', 'unknown')}",
              f"- Started: {run_meta.get('started_at', 'n/a')}",
              f"- Ended: {run_meta.get('ended_at', 'n/a')}",
              f"- Manifest hash: {run_meta.get('manifest_hash', 'n/a')}", ""]

    if vector_rows:
        lines.append("## Vectors")
        lines.append("")
        lines.append("| model | recipe | method | layer | pairs | norm |")
        lines.append("|---|---|---|---|---|---|")
        for v in vector_rows:
            lines.append(f"| {v.get('model_id')} | {v.get('recipe_id')} | {v.get('method')} | {v.get('layer_idx')} | {v.get('num_pairs')} | {v.get('vector_norm', 0):.3f} |")
        lines.append("")

    if summary:
        lines.append("## Aggregate metrics")
        lines.append("")
        for key, value in sorted(summary.items()):
            if key == "n":
                continue
            lines.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
        lines.append("")

    if generations:
        by_method_layer: Dict[Any, List[Dict[str, Any]]] = {}
        for row in generations:
            key = (row.get("recipe_id"), row.get("method"), row.get("layer_idx"), row.get("coefficient"))
            by_method_layer.setdefault(key, []).append(row)
        lines.append("## Results by recipe / method / layer / coefficient")
        lines.append("")
        lines.append(f"- Total scored rows: {len(generations)}")
        lines.append(f"- Distinct configurations: {len(by_method_layer)}")
        lines.append("")

    text = "\n".join(lines)
    destination = Path(output_path) if output_path else source / "report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _run_metadata_index(run_paths: Iterable[str | Path], output_root: str | Path) -> Path:
    """Build a machine-readable index without assuming a particular metric.
    Used when `compare` is given runs that aren't a clean steering+QLoRA
    pair (more than two runs, or two runs of the same arm) -- still useful
    for eyeballing run status/provenance side by side."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_path in map(Path, run_paths):
        run_file = run_path / "run.json"
        if not run_file.exists():
            raise FileNotFoundError(run_file)
        import json
        with run_file.open() as handle:
            run = json.load(handle)
        rows.append({"run_id": run["run_id"], "status": run["status"], "root": str(run_path), "manifest_hash": run["manifest_hash"]})
    path = output / "comparison.json"
    import json
    with path.open("w") as handle:
        json.dump({"runs": rows, "created_at": time.time()}, handle, indent=2)
    return path


def _is_qlora_run(run_path: Path) -> bool:
    return (run_path / "results" / "qlora.json").exists()


def _is_steering_run(run_path: Path) -> bool:
    return (run_path / "vectors" / "index.jsonl").exists()


def compare(run_paths: Iterable[str | Path], output_root: str | Path) -> Path:
    """Compares two run directories. When one is a steering run (has
    `vectors/index.jsonl`) and the other is a QLoRA run (has
    `results/qlora.json`), builds the matched quality/cost comparison from
    `comparison.build_comparison` and writes it via
    `comparison_report.write_comparison_report` (report.json + report.md).

    Falls back to a plain run-metadata index (previous behavior) for any
    other combination -- e.g. two steering runs, more than two runs, or a
    QLoRA run whose adapters were never evaluated (no
    `results/generations.jsonl`) -- so `compare` never hard-fails on
    legitimate non-Pillar-2 uses.
    """
    from .comparison import build_comparison
    from .comparison_report import write_comparison_report

    paths = [Path(p) for p in run_paths]
    if len(paths) == 2:
        steering_path = next((p for p in paths if _is_steering_run(p) and not _is_qlora_run(p)), None)
        qlora_path = next((p for p in paths if _is_qlora_run(p)), None)
        if steering_path is not None and qlora_path is not None and steering_path != qlora_path:
            comparison = build_comparison(steering_path, qlora_path)
            return write_comparison_report(comparison, output_root)

    return _run_metadata_index(run_paths, output_root)
