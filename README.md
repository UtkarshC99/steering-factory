# Steering Factory

Steering Factory is a Colab-first, reproducible framework for activation-steering experiments on open-weight language models. It replaces the earlier single-prompt Streamlit workflow with versioned manifests, benchmark adapters, immutable artifacts, and matched steering-versus-QLoRA comparisons.

The framework is designed for research, not deployment. Its safety recipes take a defensive-research stance: an unsafe/undesirable direction is extracted only to *measure* how present it is (held out) and to demonstrate hardening against it — via directional ablation (projecting the direction out) or conditional/gated steering (confining an intervention to inputs that actually trigger it) — never to ship the direction as an operational jailbreak. Every safety behavior's `positive` example is the safe completion, so a positive coefficient always steers toward safety, consistent across the whole safety sub-taxonomy.

## What it supports

- Two default open-weight SLM configurations: `Qwen/Qwen3-4B` and `google/gemma-3-4b-it`.
- Eight recipe families: defensive safety/refusal, multi-domain classification, calibrated abstention, JSON/structured output, plus a safety sub-taxonomy (harmful-instruction compliance, jailbreak susceptibility, deception/sycophancy, PII leakage).
- Steering extraction ablations: CAA mean difference, PCA, whitened contrastive (diagonal Fisher-style), and preference-optimized extraction.
- Intervention ablations: additive steering (layer/coefficient selection, single or scheduled layers, naïve or orthogonal vector stacking), directional ablation ("abliteration" — projects a direction *out* of the residual stream rather than adding one in, bounded by construction), and CAST-style conditional/gated steering (fires only when an input's activation projects above a threshold onto a separate condition vector — for cutting benign over-refusal without weakening the intervention on genuinely unsafe inputs).
- Baselines: unsteered, prompt/few-shot, and a matched QLoRA comparison arm, both swept across a labeled-example-count axis (`experiment.n_sweep`) for a real data-efficiency comparison rather than a single point.
- Real benchmark adapters — HarmBench, AbstentionBench, JSONSchemaBench, XSTest (benign over-refusal control) — each gated behind an explicit `dataset.reviewed_dataset_card: true` manifest flag; see [Real benchmark data](#real-benchmark-data-gated-datasets) below.
- A benign general-capability regression control (opt-in via `capability_probe.enabled`) measuring collateral damage from an intervention, independent of the target-behavior quality metric.
- Local, analysis-friendly artifacts with raw per-example outputs, metrics, resolved manifests, provenance, splits, environment data, timing, and failure state.

## Colab quick start

1. Upload [colab_run.ipynb](notebooks/colab_run.ipynb) to Colab and select a GPU runtime. It clones or fast-forwards the configured GitHub repository into `/content/steering-factory`.
2. Run the install cell, then mount Drive when prompted.
3. Keep `ARTIFACT_ROOT=/content/steering_factory_artifacts` for fast local writes. The notebook mirrors every finalized run to `My Drive/Projects/steering-factory/<run-id>` through `PUBLISH_ROOT`.
4. Run:

```bash
python -m steering_factory prepare-data manifests/default_colab.yaml \
  --set artifacts.root=/content/steering_factory_artifacts \
  --set artifacts.publish_root=/content/drive/MyDrive/Projects/steering-factory
```

The tracked notebook intentionally contains only those setup and CLI calls. It does not duplicate experiment logic.

The default manifest uses small, safe local fixtures so the pipeline can be validated offline. Replace each `dataset.path` with benchmark-normalized JSONL or use the `huggingface` adapter after reviewing the dataset card, license, and revision. Do not commit downloaded datasets or run artifacts.

## Manifest and overrides

`manifests/default_colab.yaml` is the source of truth for the model matrix, recipes, split seed, extraction/application grid, QLoRA settings, and artifact root. Overrides retain YAML types:

```bash
python -m steering_factory prepare-data manifests/default_colab.yaml \
  --set artifacts.root=/content/drive/MyDrive/steering-artifacts \
  --set splits.seed=29
```

Available commands are `prepare-data`, `run`, `extract`, `evaluate`, `finetune`, `compare`, and `report`.

- `run` executes the full extract + evaluate grid in one artifact (backward-compatible convenience wrapper).
- `extract` runs extraction only and saves vectors under `vectors/index.jsonl`, without generating.
- `evaluate <manifest> --vectors-run <run-dir>` loads vectors saved by a prior `extract`/`run` and executes the generation/eval grid against them, so you can re-score under a different coefficient/token-scope grid without re-extracting.
- `report <run-dir>` renders a Markdown summary (`report.md`) from an existing, finalized run directory. Loads no model.
- `finetune` trains one matched QLoRA adapter per model/recipe steer split, then evaluates each adapter on the same validation/test examples the steering arm uses, scored with the identical evaluator. Requires a Colab GPU and the explicit target modules in the manifest.

`compare` produces different things depending on what it is given:

```bash
python -m steering_factory compare /path/to/steering-run /path/to/qlora-run \
  --output-root /content/steering_factory_artifacts/comparisons
```

- If one path is a steering run (has `vectors/index.jsonl`) and the other is a QLoRA run (has `results/qlora.json`), `compare` builds the matched steering-vs-QLoRA quality/cost report: the best steering config is selected on validation only, held-out test quality is reported alongside QLoRA's held-out test quality for the same (model, recipe), and cost (a selected-config extraction-cost estimate, labeled-example count, artifact size, ms/token) is normalized across both arms. Writes `report.json` and `report.md`, plus a data-efficiency curve PNG (quality vs. labeled-example count per arm — see `experiment.n_sweep` below) and a layer×coefficient quality heatmap for the winning steering method, when matplotlib is installed.
- For any other combination (two steering runs, more than two runs, an untrained/unevaluated QLoRA run), `compare` falls back to a plain run-metadata index (`comparison.json`) so it never hard-fails on legitimate non-comparison uses.

**Report honesty guarantees.** A `(model, recipe)` pair whose `n_test` or `n_validation` falls below `min_split_size` (default 20) is routed to a separate `excluded` section, never reported as a normal quality/winner row — a quality number computed on a handful of held-out examples is noise, not signal. The unsteered baseline (`coefficient == 0.0`) is never selectable as the "winning" steering config; every entry reports `baseline_quality`/`beat_baseline` explicitly so a steered config that merely ties the base model can't be presented as a steering result. `assert_disjoint_selection_and_test_rows` (`comparison.py`) is a structural safety net asserting no `example_id` used for validation-based selection also appears in the reported held-out test rows — an invariant that already holds by construction, made an explicit, tested assertion so a future refactor can't silently violate it.

## Labeled-example-count sweep (`experiment.n_sweep`)

```yaml
experiment:
  n_sweep: [8, 16, 32, 64, 128, 256, 512]
```

Runs the steering arm's extraction and the QLoRA arm's training at each labeled-example count in the list (subsampling the steer/train pool deterministically per point), while validation/test are never subsampled — quality at every N is measured on identical held-out data across the whole sweep and both arms. This is what makes the comparison report's data-efficiency curve real rather than a single point. Absent (every manifest before this existed), the run is exactly one iteration at the full steer/train pool size.

## Benign capability-regression control (`capability_probe`)

```yaml
capability_probe:
  enabled: true       # opt-in, default off
  max_new_tokens: 24
```

A recipe's own quality metric only measures the target behavior; it says nothing about collateral damage elsewhere, and steering/fine-tuning arms are known to diverge most on collateral utility loss, not target-behavior efficacy. When enabled, `run_steering` runs a small, fixed set of benign general-knowledge/arithmetic/instruction-following prompts (deliberately not behavior-specific) through the base model once per model, and through each extracted vector at every coefficient already in its evaluation grid, reporting `capability_retention = steered_quality / base_quality` (`None`, not 0 or 1, when unmeasured or when the base model already scored zero on every probe — a real ratio would be meaningless there). Off by default because it multiplies generation cost roughly by `num_probes / num_eval_examples` on top of the existing sweep — a manifest should opt in deliberately.

## GPU batching

Generation (the steering coefficient sweep and the QLoRA evaluation arm) and extraction
(the closed-form methods' forward passes) batch multiple examples/pairs together per GPU
call instead of running one at a time, controlled by two manifest fields:

```yaml
decoding:
  max_new_tokens: 96
  batch_size: 16   # examples per generate() call; also used by the QLoRA eval arm
extraction:
  batch_size: 16   # (compliant, non_compliant) pairs per forward pass
finetune:
  batch_size: 4                  # QLoRA micro-batch
  gradient_accumulation_steps: 2  # effective batch size = batch_size * this, kept
                                  # constant at 8 relative to the old 1x8 default so
                                  # training dynamics and the resulting adapter don't change
```

Batching is done across examples/pairs at a fixed (method, layer, coefficient,
token_scope) — the one axis every row in a batch can share, since one steering hook
carries a single coefficient/vector for its whole active scope. It never batches across
coefficients. Left-padding (the decoder-only convention) is used throughout and is
required for correct results on this project's target architectures (Qwen3, Gemma-3; both
RoPE + attention-mask-aware). Batching is verified bit-for-bit equivalent to batch size 1
— same output text, logprobs, and JS divergence — across ragged prompt lengths and all
three `token_scope` values; see `tests/test_sweep_batching.py`,
`tests/test_extraction_batching.py`, `tests/test_finetune_batching.py`, and the end-to-end
`tests/test_runner_batching_e2e.py`.

`results/generations.jsonl` rows carry both `latency_s` (amortized: `batch_wall_time_s /
batch_size`) and the measured `batch_wall_time_s`/`batch_size` alongside it, so cost
analysis (including the steering-vs-QLoRA comparison) can distinguish a true per-row
measurement from a batch-amortized estimate rather than silently treating one as the other.

Default `batch_size: 16` is sized for an L4 (22.5 GB VRAM): at ~30-token prompts and
`max_new_tokens: 96`, KV cache is negligible and Gemma-3's 262k-token vocab in retained
generation scores is the dominant memory cost, well within budget at this batch size with
large headroom to spare. Lower it for smaller GPUs; there is no assumption tying it to a
specific card.

## Artifact layout

Every invocation creates a new `<artifact-root>/<timestamp>-<id>/` directory. When `artifacts.publish_root` is set, each finalized run is copied to `<publish-root>/<timestamp>-<id>/`; `publish.json` records the destination or any Drive-copy error. It includes:

```text
resolved_manifest.yaml       exact effective configuration
run.json                     run id, status, time, error and manifest hash
environment.json             Python, Torch, GPU and Git revision when available
command.json                 command provenance
data/examples.jsonl          normalized records with split/provenance fields
data/*.provenance.json       source fingerprint, split plan and leakage checks
vectors/*.pt                 saved vectors
vectors/*.metadata.json      layer/model/hidden-size/intervention metadata
results/*.jsonl              per-example generations and scores
results/*.summary.json       aggregated metrics
failure.json                 traceback for failed runs
```

Track the source, license/revision, exact split IDs, template hash, model/tokenizer revision, quantization, seed, decoding settings, wall time, tokens, latency, throughput, peak VRAM, vector/adapter size, and evaluator identity for every model-execution plugin. Existing metrics—perplexity ratio, JS divergence, repetition, distinct-n, vector norm, coefficient, layer, and convergence—remain part of the steering evaluator output.

## Quantitative evaluation

Report benchmark test results separately from synthetic augmentation. Select layers, coefficients, and stack weights on validation only.

- **Safety:** safe/refusal rate, unsafe-compliance rate, per-category breakdown, false-refusal rate on benign controls, and harmless-task regression — all rolled up by `evaluators.safety_metric_bundle` into `results/safety_summary.json` whenever a recipe's rows carry `safe_refusal`. Refusal itself is scored either by a cheap deterministic substring matcher (`evaluators.refusal_score_substring`, always available) or by an LLM judge (`evaluators.refusal_score_judge`, opt-in — pass any `generators.base.GeneratorBackend` as `judge=` to `run_steering`/`run_evaluate`/`run_qlora`; every judged row records `judge_model`, `judge_rubric_hash`, and `judge_raw_score`). False-refusal is measured specifically on XSTest's benign (non-contrast) rows via the `is_safe_control` field XSTestAdapter attaches. Use controlled sources such as [HarmBench](https://www.harmbench.org/) only after reviewing their intended research use — see [Real benchmark data](#real-benchmark-data-gated-datasets).
- **Classification:** exact match, macro-F1, allowed-label validity, confidence calibration, and per-domain/cross-domain results.
- **Abstention:** appropriate and false abstention, answer accuracy when answering, selective accuracy, Brier score, and ECE. The benchmark adapter is [AbstentionBench](https://huggingface.co/datasets/facebook/AbstentionBench) (`adapter: abstentionbench`).
- **Structured output:** parse success, JSON Schema validity, required-field coverage, and exact leaf-value accuracy. The schema source is [JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench) (`adapter: jsonschemabench`).
- **Cost:** extraction/training/evaluation time, GPU-hours, token counts, peak allocated/reserved VRAM, artifact size, and steering request overhead.

## Real benchmark data (gated datasets)

`manifests/benchmarks.yaml` points at real HuggingFace datasets instead of the local JSONL
fixtures `manifests/default_colab.yaml` uses. Every adapter that loads one of these
(`harmbench`, `abstentionbench`, `jsonschemabench`, `xstest`, all in `steering_factory/datasets.py`)
refuses to run without an explicit `dataset.reviewed_dataset_card: true` in the manifest —
this is a deliberate speed bump and an auditable record that a human reviewed the license
and terms of use, not just a runtime check.

**Do this before setting any `reviewed_dataset_card` flag to `true`:**

1. Open the dataset's page on HuggingFace and read the dataset card in full:
   [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench) (gated — requires
   agreeing to share contact info; MIT-licensed code, review the intended research-use terms),
   [facebook/AbstentionBench](https://huggingface.co/datasets/facebook/AbstentionBench)
   (cc-by-nc-4.0), [epfl-dlab/JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench)
   (MIT), [walledai/XSTest](https://huggingface.co/datasets/walledai/XSTest) (CC-BY-4.0 prompts —
   real harmful-request text in the "contrast" half; review before use).
2. **Verify the adapter's column mapping against the live dataset viewer**, not just this
   README or the adapter's docstring. AbstentionBench's schema
   (`question`/`reference_answers`/`should_abstain`/`metadata_json`) and JSONSchemaBench's
   (`json_schema`/`unique_id`) were confirmed against their live dataset cards during
   development. **HarmBench's (`Behavior`/`SemanticCategory`/`BehaviorID`) and XSTest's
   (`prompt`/`type`) were sourced from published paper/repo documentation only** — both
   datasets are access-gated and could not be independently verified against the live schema
   from the development environment that built these adapters. Check the actual column names
   in the HuggingFace dataset viewer before trusting a real run's results; if they've
   changed, update the adapter's `mapping`/column-read logic in `datasets.py` to match.
3. Request access where required (HarmBench) and confirm you're authorized to use the data
   for this research purpose.
4. Only then set `reviewed_dataset_card: true` for that recipe in your manifest.

This step needs a GPU session with network access (e.g. Colab) — it cannot be validated
offline or in a CI environment.

## Extending the framework

- Add a `DatasetAdapter` in `steering_factory/datasets.py`, normalize records to `ContrastiveExample`, and preserve source/revision/license metadata. Follow the `HarmBenchAdapter`/`AbstentionBenchAdapter`/`JSONSchemaBenchAdapter`/`XSTestAdapter` pattern for a benchmark that ships prompts/labels but not ready-made contrastive pairs: `normalize` is where the (positive, negative) contrast is synthesized, and that construction is the substantive per-benchmark logic, not just column renaming.
- Add a behavior to `steering_factory/behaviors.py`; custom behaviors require an explicit description in the manifest. `_REFUSAL_SCORED_BEHAVIORS` in `runner.py` lists which behavior_ids get refusal-style scoring explicitly, by id — not by family — so a custom safety-family behavior isn't silently scored this way just because its family happens to be `"safety"`.
- Add extraction methods behind `steering_factory.extraction.extract`, and record enough vector metadata to reject incompatible hidden sizes/models.
- Add intervention modes behind `steering_factory/hooks.py`/`interventions.py`, following `AblationHook`/`ConditionalSteeringHook`'s pattern: verify batch-safety under left-padding against this project's actual target architecture family (Llama/Qwen3/Gemma-3 — a GPT-2-family stand-in was found NOT to be batch-safe here without extra handling), and anchor the new mode's correctness against the existing additive `SteeringHook` at its degenerate extremes (e.g. a gate that always/never fires should be bit-identical to a known baseline).
- Add evaluators in `steering_factory/evaluators.py`. Use deterministic metrics where labels or schemas exist; judge outputs must include judge model/version, rubric/template hash, and errors — see `refusal_score_judge` for the pattern (falls back to a deterministic scorer if the judge declines, rather than producing no score).
- Attach a model execution or QLoRA trainer plugin only when its chat template, target modules, labels, decoding, and telemetry are explicit. It must write per-example rows through `runner.write_evaluation`.

## Explainability and research next steps

Use convergence curves, layer/coefficient response curves, vector norms and cosine similarity, per-category failure tables, and vector orthogonality/interference matrices to understand an intervention before making claims about it. A high training sweep score is not evidence of held-out steering. The comparison report's layer×coefficient heatmap (`comparison_plots.py`) is a first step here — it shows whether a validation-selected winning config sits in a broad, robust region or is an isolated spike surrounded by much worse neighbors.

Implemented: execute recipes with an N-sweep across labeled-example counts (`experiment.n_sweep`); compare steering and QLoRA at matched quality/cost points (`compare`); benign capability regression (`capability_probe`); a held-out vulnerability protocol guard (`comparison.assert_disjoint_selection_and_test_rows`).

Still recommended as next studies: execute all recipes across several seeds and report mean ± CI, not single-run numbers; calibrate evaluator disagreement (run two judges, report agreement); the cross-behavior vector-stacking/interference study (`interventions.compose`'s orthogonal mode exists but hasn't been exercised as a full experiment); a full explainability report layer (convergence/layer-response/orthogonality plots as one artifact, not ad hoc). Do not infer transferability or compositionality from in-family single-vector results.

## Development

```bash
pip install -r requirements.txt
pytest tests/
python -m steering_factory prepare-data manifests/default_colab.yaml --set artifacts.root=/tmp/steering-artifacts
```

The existing Streamlit prototype remains in `steering_factory/app.py` for interactive exploration of legacy vectors. New research runs should use manifests and artifacts.
