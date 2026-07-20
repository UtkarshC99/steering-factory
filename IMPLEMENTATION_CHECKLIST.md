# Steering Factory — Implementation Checklist

Status legend: `[x]` done in-repo · `[~]` partial / stubbed · `[ ]` not started

This checklist maps the three-pillar vision (safety attack-vectors, steering-vs-QLoRA
cost question, reusable components) onto concrete work. Items are ordered so that each
phase produces a *runnable, publishable result*, not just scaffolding.

---

## Phase 0 — Close the gaps in what already exists (fastest path to a real result)

- [~] **Wire the CLI commands to distinct behaviors.** Today `extract`, `evaluate`,
  `report` all alias `run_steering` (`cli.py:15`). Split them:
  - [ ] `extract` → extract + save vectors only (no generation), for cheap iteration.
  - [ ] `evaluate` → load saved vectors + run the eval grid.
  - [ ] `report` → render tables/plots from an existing run dir (no model load).
- [~] **Make `compare` do the steering-vs-QLoRA comparison** (currently only diffs
  `run.json` metadata). It must join a steering run and a QLoRA run on
  `(model, recipe)` and emit matched quality + cost rows. This is the Pillar-2
  deliverable — see Phase 2.
- [ ] **README ↔ code reconciliation.** README advertises commands/adapters/benchmarks
  that are aliases or fixtures. Either implement them or mark them "planned" so the
  repo doesn't over-claim.
- [ ] **Reconcile the safety framing.** README says safety recipes are *defensive only*;
  the research vision wants to *identify attack vectors & alignment vulnerabilities*.
  Decide the stance (recommended: defensive-research framing — "we extract the
  direction that *induces* unsafe compliance in order to measure and harden against
  it", holdout-measured, no operational uplift shipped) and make README + behavior
  descriptions consistent.
- [ ] **CI:** GitHub Action running `pytest` + a CPU-only smoke run of
  `prepare-data` on the fixtures. Tests already pass (32/32); lock that in.

---

## Phase 1 — Pillar 1: Safety / attack-vector steering

### Data & taxonomy
- [~] **Attack-vector / behavior taxonomy** in `behaviors.py`. Extend the registry beyond
  the 4 defaults with a *safety sub-taxonomy* (e.g. harmful-instruction compliance,
  jailbreak-susceptibility, deception/sycophancy, PII leakage), each with an explicit
  `description` used by the synthetic generator.
- [ ] **HarmBench adapter** (`datasets.py`): normalize prompts + category labels; record
  license/revision. Gate behind an explicit `--i-have-reviewed-dataset-card` style flag.
- [ ] **Benign control set** (false-refusal / over-refusal): wire a benign-but-adjacent
  prompt set (e.g. XSTest-style) so you can measure over-refusal, not just refusal.
- [ ] **Synthetic contrastive generation for each attack vector** using the existing
  generator backends (`generators/`). Validate diversity with the convergence check
  in `metrics.convergence_summary` (already flags near-duplicate pairs).

### Method (the safety-specific arm the literature now expects)
- [ ] **Directional ablation ("abliteration") arm** — project the refusal/unsafe direction
  *out* of the residual stream, not just add it. Distinct from all 4 current additive
  methods. Add as `interventions.py` mode `"ablate"`.
- [ ] **Conditional / gated steering (CAST-style)** — fire the vector only when the input
  activation's projection onto a *condition* vector exceeds a threshold. New arm; big
  for reducing over-refusal on benign controls.

### Evaluation
- [~] **Replace substring refusal scorer** (`runner.py:69`) with a proper classifier:
  keep the deterministic substring scorer as a cheap baseline, add an LLM-judge
  (generators already expose `score_behavior`) and record judge model/version/rubric hash.
- [ ] **Safety metric bundle** (README already lists these — implement the missing ones):
  safe-refusal rate, unsafe-compliance rate, per-category breakdown, **false-refusal on
  benign controls**, **harmless-task regression**. Report benchmark test separately from
  synthetic aug.
- [ ] **Holdout protocol enforcement:** select layer/coef on validation only; the
  vulnerability claim must be a *held-out* uplift number. Add a guard that refuses to
  report test metrics if any selection touched the test split.

---

## Phase 2 — Pillar 2: Steering vs QLoRA, the "money question"

- [x] QLoRA trainer exists (`finetune.py`), matched to the steer split.
- [ ] **Matched-quality comparison harness.** For each (model, recipe): find the steering
  config and the QLoRA checkpoint that hit the *same* task-quality band, then compare cost.
  This is the missing core of `compare`.
- [~] **Cost telemetry parity.** `run_steering` logs wall-time + peak VRAM; QLoRA logs
  train time + adapter size. Normalize both into one schema:
  - extraction/training GPU-hours, eval GPU-hours, peak VRAM, artifact bytes,
    per-request steering overhead (ms/token), tokens seen.
- [ ] **Data-efficiency curve:** steering (8/16/32/… pairs) vs QLoRA (100s–1000s examples)
  — quality as a function of labeled data. This is the headline plot for the CV bullet.
- [ ] **Comparison report artifact:** a single `comparison/report.{json,md,html}` with the
  matched quality/cost table + data-efficiency curve, per model × recipe.
- [ ] **Capability-regression control:** run a small benign general-capability probe before/
  after each intervention (steering *and* QLoRA) so "specialized gain" is net of collateral
  damage.

---

## Phase 3 — Reusable components (harden what the pillars share)

### Synthetic data generator
- [x] Backend abstraction + Anthropic/OpenAI-compat/local-HF (`generators/`).
- [x] Custom-behavior-via-prompt path (`behaviors.get_behavior` requires a description).
- [ ] **Persist generated pairs as first-class artifacts** with provenance (generator model,
  prompt hash, seed) so a run is reproducible from the manifest alone.
- [ ] **Dedup / quality filter** on generated pairs (embedding-similarity threshold) to
  prevent the false-convergence problem the templates already warn about.

### Extraction (ablation coverage)
- [x] `mean_diff` (CAA baseline), `pca`, `whitened_mean_diff` (diag-Fisher), `optimized`
  (BiPO-lite). **Meets the "3+ techniques + baseline" bar.**
- [ ] **Optional SAE-feature extraction arm** (interpretable/sparse direction) — 2025
  literature (SAS/YaPO) treats this as a distinct, more-stable family. Nice-to-have,
  scopes to models with available SAEs.
- [ ] **Pooling ablation** already supported (`last`/`mean`) but not swept in the manifest —
  add to the grid.

### Application (intervention ablation)
- [x] single / sum / **orthogonal** stacking (`interventions.compose`).
- [x] multi-layer + scheduled-layer injection, scale normalization option.
- [ ] `token_scopes: [all, generated_only]` is in the manifest but the hook applies to
  **all** positions — implement `generated_only` in `hooks.SteeringHook` (currently
  advertised, not enforced).
- [ ] **Cross-behavior stacking study:** the orthogonal-stacking machinery exists; add the
  experiment that stacks *orthogonal behaviors at different layers* and measures
  interference (vector orthogonality/cosine matrix is already in the README's plan).

### Evaluation & explainability
- [x] Deterministic scorers (classification / abstention / structured) + ECE + aggregation.
- [ ] **Judge-disagreement calibration** (README next-step): run 2 judges, report agreement.
- [ ] **Explainability plots:** convergence curves, layer/coef response surfaces, vector
  norm/cosine, per-category failure tables, orthogonality/interference matrix. Data is
  already emitted; the plotting/report layer is missing.

---

## Phase 4 — Rigor & write-up (what makes it publishable / CV-credible)

- [ ] **Multi-seed runs** (README recommends): ≥3 seeds per recipe; report mean ± CI, not
  single-run numbers.
- [ ] **Two-model matrix as default** (Qwen3-4B + Gemma-3-4B already in manifest) — confirm
  both actually run end-to-end on a Colab GPU.
- [ ] **Negative-result honesty:** explicitly state where steering *loses* to QLoRA. A
  balanced result is stronger for the "cost-effective alternative" claim than a
  cherry-picked win.
- [ ] **Reproducibility bundle:** one manifest + one notebook reproduces every headline
  number from published artifacts.

---

## Suggested execution order (minimal path to the CV bullet)

1. Phase 0: wire `extract`/`evaluate`/`report` + make `compare` real. *(unblocks everything)*
2. Phase 2 cost-comparison harness on the **existing fixtures** — get one real
   steering-vs-QLoRA table out, even on toy data.
3. Phase 1 HarmBench + benign-control adapters + proper refusal judge → first safety result.
4. Add the **ablation/abliteration + conditional-steering** arms (Phase 1 method + Phase 3).
5. Phase 4 multi-seed + write-up.
