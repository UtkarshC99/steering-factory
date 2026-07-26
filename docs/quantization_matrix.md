# Quantization × parameter matrix

This document explains how model-weight quantization interacts with
activation steering and QLoRA, and lays out the preset experiment manifests
in [`manifests/quantization/`](../manifests/quantization/) that test specific,
falsifiable hypotheses about that interaction — rather than one large,
unfocused grid sweep.

## The precision path: what quantization actually changes

Tracing the real code path from extraction through steering to generation:

1. **Extraction pools activations, then immediately upcasts to fp32.**
   `_pooled_activations_batch` in
   [`extraction.py`](../steering_factory/extraction.py) runs a forward pass
   at the model's compute dtype (bf16 under 4bit/8bit bitsandbytes
   quantization, or whatever `dtype` is set to when unquantized), pools the
   captured activations, and returns `pooled.float().cpu()` — fp32, on CPU.
   Every downstream vector operation (`_collect_diffs`, PCA, whitening,
   the optimized/BiPO-lite gradient loop) is fp32 arithmetic. The
   **stored** vector on disk is fp32.

2. **At steer time, the fp32 vector is downcast to the residual stream's
   dtype.** `SteeringHook._hook` in
   [`hooks.py`](../steering_factory/hooks.py) does
   `vec = self.vector.to(hidden_states.dtype)` before adding
   `coefficient * vec` to the residual stream. So the addition itself
   happens in the model's compute dtype (bf16), not fp32 — but that
   downcast is the same regardless of whether the *weights* are quantized,
   because bf16 is the compute dtype in every precision level this matrix
   tests (see "Why bf16, not fp16/fp32" below).

**The conclusion this leads to:** quantization does **not** change the
arithmetic precision of extraction or the steering addition — every vector
computation and the hook's own add are done at a dtype independent of
whether the underlying weights are 4-bit, 8-bit, or unquantized. What
quantization changes is **which weights produce the activations and logits**
those operations see. A 4-bit nf4-quantized weight matrix produces a
measurably different forward pass than the same weights at full precision —
the direction extracted from those activations, and the model's own
behavior when steered, both inherit that difference.

This has three separable, measurable consequences, which the four presets
below are each designed to isolate one of:

| # | Consequence | Isolated by |
|---|---|---|
| 1 | The extracted **direction itself shifts** because the diffs come from quantized forward passes | H1 (quality gap vs the bf16 control), H3 (which layer is robust) |
| 2 | The **model's own behavior** (what "compliant"/"refusing" looks like, post-quantization) shifts independent of steering | the bf16 control run, used as every other run's baseline reference |
| 3 | **QLoRA's trainable capacity** changes with the base's bit-depth — an adapter sits on top of (and is more sensitive to) the base weights' own representational error than a coarse steering direction is | H1, directly (steering-vs-QLoRA gap comparison) |

## Why bf16, not fp16 or fp32, as the unquantized control

bf16 is the compute dtype used throughout this pipeline already —
`bnb_4bit_compute_dtype` for the 4-bit path, and `torch_dtype` for every
`from_pretrained` call regardless of quantization
(`model_utils.py:resolve_dtype`/`bnb_config_for`). Two precision levels were
deliberately excluded from this matrix:

- **fp32**: would double memory again for a *compute-dtype* question
  (arithmetic range/precision) that is orthogonal to the *weight-fidelity*
  question this matrix is actually about. fp32 doesn't tell you anything
  about quantization error that bf16 doesn't already show as the compute
  dtype every other precision level shares.
- **fp16**: same reasoning — fp16 vs bf16 is a numerical-range question
  (fp16 has a narrower exponent range, more prone to overflow in some
  activation regimes), not a weight-quantization question. Using it would
  introduce a second confounding variable alongside the one this matrix is
  designed to isolate.

So "full precision" here means **unquantized weights at bf16 compute** —
the honest apples-to-apples anchor against the 4-bit and 8-bit runs, which
also compute in bf16.

## Why a matched-precision design, not a fixed-4bit QLoRA baseline

"QLoRA" is, by definition, 4-bit ("Q" = quantized). Comparing steering at
varying precision against a QLoRA arm permanently pinned at 4-bit would
conflate two different questions ("does steering degrade less than QLoRA at
matched precision" vs "does steering at precision X beat QLoRA's *fixed*
4-bit baseline"). This matrix chooses the former: both arms now read the
same manifest model entry's `quantization`/`dtype` fields
(`finetune.py`'s `train_qlora`/`evaluate_qlora_adapter`/`evaluate_base_model`
were parameterized via `model_utils.bnb_config_for`/`resolve_dtype` — the
same helper the steering arm's `load_model` already used). When a model
entry is unquantized, the QLoRA arm runs plain LoRA on a full-precision base
(skipping `prepare_model_for_kbit_training`, which is only valid for an
actually-quantized base) — worth naming honestly as "LoRA-bf16" rather than
"QLoRA" in that case, since the "Q" no longer applies. Each preset's
`results/qlora.json`/generation rows record `quantization`/`dtype`
explicitly for exactly this reason.

## Parameter-justification table

What each tunable controls, and why "optimal" shifts by precision (see
"Per-preset design" below for how each preset actually spends its budget):

| parameter | controls | why it interacts with quantization |
|---|---|---|
| `quantization` (4bit / 8bit / none) | fidelity of the weights that produce the activations the vector is extracted from and steered against | the headline axis — see "The precision path" above |
| `n_sweep` | labeled-example budget per arm | lower-bit weights → noisier per-example activations → the data-efficiency curve may need more points/larger N to keep `min_split_size=20` cleared with margin, avoiding the exact n_test=1 false-signal failure mode `comparison.py` already guards against |
| `steer_fraction` | size of the steer pool feeding extraction | governs how many `n_sweep` points are non-redundant (`_subsample_n` returns the whole pool unchanged once `n >= len(steer)`) — second-order relative to quantization, held fixed across all four presets for comparability |
| `layers` | depth in the residual stream where the direction is applied/extracted | quantization error is not uniform across depth — a layer that's robust under bf16 is not necessarily robust under 4-bit (H3) |
| `coefficients` | steering dose, AND (2026-07-26) whether the extracted direction is bidirectional | quantization perturbs activation scale slightly, so the coefficient that was optimal at one precision may not be optimal at another. Widened to 13 points (from 7) and made a MANDATORY negative-analog check: a real run showed `safe_refusal` (bounded at 0, base models already refuse only 1-12% of the time) has almost no headroom to show a negative effect, while the unbounded `js_divergence_vs_baseline` showed the negative direction is real and often as strong as the positive one — see `comparison.DEFAULT_FLUENCY_CAP_RATIO`'s module docstring and memory: steering-metric-floor-effect. The same run also showed the fluent/degenerate boundary sits at a DIFFERENT magnitude per model (c=-2 was fluent for one model, degenerate — high repetition — for the other), which is why the grid is denser near ±0.25-0.75 rather than just wider at the extremes |
| extraction `method` (mean_diff / pca / whitened_mean_diff) | how the direction is estimated from the pooled diffs | `whitened_mean_diff` is the most sensitive to activation covariance, which quantization perturbs more than a raw mean; `mean_diff` is the least sensitive, making it the right choice when isolating a *different* axis (e.g. H3's layer scan) |

## Per-preset design

Each precision combination gets its **own** manifest rather than one shared
grid, because a grid answers "what's the single best config overall" — not
the goal here. A hypothesis-per-preset design answers "does *this specific
claimed effect* show up," which is what makes each result independently
falsifiable.

| file | precision (steering = LoRA, matched) | hypothesis | what's tuned for it |
|---|---|---|---|
| [`preset_4bit_baseline.yaml`](../manifests/quantization/preset_4bit_baseline.yaml) | 4bit / bf16 compute | none — this is the reference every other preset is measured against | identical grid to `manifests/benchmarks.yaml` (already validated on real GPU runs) |
| [`preset_8bit_midpoint.yaml`](../manifests/quantization/preset_8bit_midpoint.yaml) | 8bit / bf16 | **H1**: steering's quality gap to the bf16 control is smaller than QLoRA's gap, at matched precision — a direction is coarser than a learned weight delta, so it should degrade less under quantization | identical grid to the 4bit baseline (only precision varies, isolating the precision effect cleanly) |
| [`preset_bf16_control.yaml`](../manifests/quantization/preset_bf16_control.yaml) | none (unquantized) / bf16 | **H2** (the anchor): this run *is* ground truth — every other preset's quality is read as a delta from this run, not an absolute number | same grid, plus a widened `layers: [early, mid, late]` scan (cheapest preset to widen safely — most VRAM headroom) |
| [`preset_low_bit_layer_scan.yaml`](../manifests/quantization/preset_low_bit_layer_scan.yaml) | 4bit / bf16 (same weights as the baseline) | **H3**: quantization error is depth-dependent — the layer that's optimal under bf16 is not necessarily optimal under 4-bit | widened `layers` axis, narrowed everything else (`mean_diff` only, single `n_sweep` point, 2 coefficients) to afford the wider layer grid within a similar step budget — spending the budget on the axis the hypothesis needs, not spreading thin |

`preset_4bit_baseline.yaml` and `preset_bf16_control.yaml` are the two that
should run **first** — H1 and H3 both need the bf16 control's numbers to
compare against. `preset_low_bit_layer_scan.yaml` is a deliberate follow-up,
not part of the first confirming pass.

## Memory sizing per preset (A100 40GB, ~39.4 GiB usable)

This matrix originally targeted an L4 (22.03 GiB). Migrated to A100 40GB
partway through, and the sizing story has gone through two real revisions
since — both are worth keeping, since each corrected a real
over-/under-estimate rather than being a cosmetic change:

1. **A real fragmentation OOM** (`Tried to allocate 11.59 GiB`, only 10.84
   GiB free, 2.44 GiB "reserved but unallocated") happened on the L4
   because `torch.cuda.empty_cache()` was only called once per **model**,
   not per vector. Fixed with per-vector `empty_cache()` calls.
2. **A real OOM on `structured_output_real`** (since removed from every
   preset — see that recipe's own removal note in each manifest) traced to
   extraction's forward pass having no batch-size cap on a per-recipe
   varying-length input, allocating 2.38 GiB in a single tensor. Generation
   now has `sweep.run_with_oom_backoff` (halves and retries on OOM) plus
   `sweep.length_bucketed_chunks` (groups similar-length prompts so batch
   cost tracks real content); extraction still has neither, which is why
   its `batch_size` stays below decoding's in every preset.
3. **The batch-size ESTIMATE itself was wrong by ~2.6x.** The original
   derivation (below, kept for the historical record) extrapolated
   ≈0.4125 GiB per unit of batch from a single L4 measurement. A real
   A100 run (2026-07-26) measured the ACTUAL peak at decoding batch=64: **12.08
   GiB**, not the ~14 GiB the old model would have predicted at that
   batch size on this card's larger weight/activation mix — implying a
   corrected slope of ≈0.1575 GiB per unit of batch, roughly a third of
   the original estimate. batch_size was raised again on the strength of
   that real measurement, but only ~1.5x (decoding) / ~1.3x (extraction,
   the unprotected path), not to the much higher ceiling the corrected
   slope implies — the project has already been burned once by trusting
   an estimate over a measurement (this whole numbered list exists because
   of that), so another large one-off jump is deliberately avoided.

Original L4-era derivation (kept for reference, not current): a live L4
run measured 8.6 GiB at `batch_size=16` under 4bit (~2.0 GiB nf4 weights +
~6.6 GiB activations/KV-cache at that batch size) → ≈0.4125 GiB/unit-batch,
6.5 GiB margin, giving max-safe-batch ≈33/28/18 for 4bit/8bit/bf16 on a
22.03 GiB L4.

Current (A100, corrected slope ≈0.1575 GiB/unit-batch, same 6.5 GiB
margin, 39.4 GiB usable):

| preset | weights (~) | corrected ceiling | configured `decoding.batch_size` | configured `extraction.batch_size` |
|---|---|---|---|---|
| 4bit baseline | 2.0 GiB | ~196 | 96 | 62 |
| 8bit midpoint | 4.0 GiB | ~183 | 72 | 42 |
| bf16 control | 8.0 GiB | ~158 | 72 | 42 |
| low-bit layer scan (4bit) | 2.0 GiB | ~196 | 96 | 62 |

`extraction.batch_size` is deliberately BELOW `decoding.batch_size` in
every preset now, reversing the original claim that extraction is
"strictly cheaper per row" (true per row, irrelevant in practice: it has
no OOM backoff and `_collect_diffs` batches pos+neg together, doubling
the effective batch).

## Results-reporting template

For each preset, once run: state the expected effect direction (from the
hypothesis above) *before* looking at the numbers, then fill in the
observed result and whether it confirmed or refuted the hypothesis. This
keeps the matrix a set of pre-registered, checkable claims rather than an
open-ended sweep read after the fact.

Before quoting a `test_quality`/winner from `report.md`, also check (added
2026-07-26, see Tier 1 of that date's work):
- The `distinct%` column and the "Safety / degeneracy controls" section --
  a real run showed a QLoRA adapter score `safe_refusal=1.0` while
  collapsed to 1-6 distinct outputs and falsely refusing 68-96% of benign
  prompts. `winner` now reads `inconclusive (...)` when this happens, but
  a raw `test_quality` number pulled directly from `report.json` does not
  carry that caveat with it.
- The "Steering configs rejected for fluency" section -- a config that
  scored well on validation but was excluded for degenerate perplexity/
  repetition will show there instead of as the selected winner.
- `recipes[].steering.benign_control` for `harmful_instruction_compliance`
  and `sycophancy_agreement` specifically, both of which now have a
  matched control (XSTest, and the density check above) precisely because
  a bare quality number on those two behaviors is the easiest to game by
  refusing/agreeing unconditionally.

### H1 — `preset_8bit_midpoint.yaml` vs `preset_bf16_control.yaml`

- **Expected**: `|steering_quality(8bit) − steering_quality(bf16)|` <
  `|qlora_quality(8bit) − qlora_quality(bf16)|`, per (model, recipe).
- **Observed**: _fill in after running both presets and diffing
  `report.json`'s `comparisons[].steering.test_quality` /
  `comparisons[].qlora.test_quality` across the two runs._
- **Verdict**: _confirmed / refuted / inconclusive (below `min_split_size`)_

### H2 — `preset_bf16_control.yaml` (the anchor)

- Not itself confirmed/refuted — record the raw `steering.test_quality`,
  `qlora.test_quality`, and `baseline_quality` per (model, recipe) here as
  the reference values H1 and H3 are measured against.

### H3 — `preset_low_bit_layer_scan.yaml` vs `preset_bf16_control.yaml`'s layer scan

- **Expected**: the best-quality layer in `preset_low_bit_layer_scan.yaml`'s
  `config_grid` differs from the best-quality layer in
  `preset_bf16_control.yaml`'s `config_grid`, for at least one (model,
  recipe).
- **Observed**: _fill in after running both — compare each entry's
  `config_grid` (sorted by `test_quality`) top layer._
- **Verdict**: _confirmed / refuted per (model, recipe)_

## Running a preset

```
python -m steering_factory run manifests/quantization/preset_4bit_baseline.yaml
python -m steering_factory finetune manifests/quantization/preset_4bit_baseline.yaml
python -m steering_factory compare <steering_run_dir> <qlora_run_dir> --output-root <output_dir>
```

Repeat for `preset_bf16_control.yaml`, then diff the two `report.json`s for
H1/H2. `preset_8bit_midpoint.yaml` and `preset_low_bit_layer_scan.yaml`
follow the same pattern once the first pass is confirmed trustworthy
end-to-end.
