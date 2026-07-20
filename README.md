# Steering Factory

Steering Factory is a Colab-first, reproducible framework for activation-steering experiments on open-weight language models. It replaces the earlier single-prompt Streamlit workflow with versioned manifests, benchmark adapters, immutable artifacts, and matched steering-versus-QLoRA comparisons.

The framework is designed for research, not deployment. Its safety recipes are defensive: they measure refusal robustness and alignment regressions using controlled benchmark data and benign controls.

## What it supports

- Two default open-weight SLM configurations: `Qwen/Qwen3-4B` and `google/gemma-3-4b-it`.
- Four recipe families: defensive safety/refusal, multi-domain classification, calibrated abstention, and JSON/structured output.
- Steering extraction ablations: CAA mean difference, PCA, whitened contrastive (diagonal Fisher-style), and preference-optimized extraction.
- Intervention ablations: layer/coefficient selection, single or scheduled layers, scale normalization, and naïve or orthogonal vector stacking.
- Baselines: unsteered, prompt/few-shot, and a matched QLoRA comparison arm.
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

Available commands are `prepare-data`, `run`, `extract`, `evaluate`, `finetune`, `compare`, and `report`. `run`, `extract`, `evaluate`, and `report` execute the configured steering grid; `finetune` trains one matched QLoRA adapter per model/recipe steer split. The latter requires a Colab GPU and the explicit target modules in the manifest.

To create a comparison index after runs:

```bash
python -m steering_factory compare /path/to/run-a /path/to/run-b \
  --output-root /content/steering_factory_artifacts/comparisons
```

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

- **Safety:** safe/refusal rate, unsafe-compliance rate, category breakdown, false refusal on benign controls, and harmless-task regression. Use controlled sources such as [HarmBench](https://www.harmbench.org/) only after reviewing their intended research use.
- **Classification:** exact match, macro-F1, allowed-label validity, confidence calibration, and per-domain/cross-domain results.
- **Abstention:** appropriate and false abstention, answer accuracy when answering, selective accuracy, Brier score, and ECE. The intended benchmark adapter is [AbstentionBench](https://huggingface.co/datasets/facebook/AbstentionBench).
- **Structured output:** parse success, JSON Schema validity, required-field coverage, and exact leaf-value accuracy. The intended schema source is [JSONSchemaBench](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench).
- **Cost:** extraction/training/evaluation time, GPU-hours, token counts, peak allocated/reserved VRAM, artifact size, and steering request overhead.

## Extending the framework

- Add a `DatasetAdapter` in `steering_factory/datasets.py`, normalize records to `ContrastiveExample`, and preserve source/revision/license metadata.
- Add a behavior to `steering_factory/behaviors.py`; custom behaviors require an explicit description in the manifest.
- Add extraction methods behind `steering_factory.extraction.extract`, and record enough vector metadata to reject incompatible hidden sizes/models.
- Add evaluators in `steering_factory/evaluators.py`. Use deterministic metrics where labels or schemas exist; judge outputs must include judge model/version, rubric/template hash, and errors.
- Attach a model execution or QLoRA trainer plugin only when its chat template, target modules, labels, decoding, and telemetry are explicit. It must write per-example rows through `runner.write_evaluation`.

## Explainability and research next steps

Use convergence curves, layer/coefficient response curves, vector norms and cosine similarity, per-category failure tables, and vector orthogonality/interference matrices to understand an intervention before making claims about it. A high training sweep score is not evidence of held-out steering.

Recommended next studies are: execute all recipes across several seeds; compare steering and QLoRA at matched quality/cost points; test benign capability regression; calibrate evaluator disagreement; then add carefully controlled cross-family vector-transfer experiments. Do not infer transferability or compositionality from in-family single-vector results.

## Development

```bash
pip install -r requirements.txt
pytest tests/
python -m steering_factory prepare-data manifests/default_colab.yaml --set artifacts.root=/tmp/steering-artifacts
```

The existing Streamlit prototype remains in `steering_factory/app.py` for interactive exploration of legacy vectors. New research runs should use manifests and artifacts.
