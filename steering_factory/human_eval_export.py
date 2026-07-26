"""Human-evaluation onboarding package: for each held-out example, shows
EVERY swept steering config (not just the validation-selected winner)
alongside the QLoRA output, plus a shared manifest header, and writes them
as CSV/JSONL/a self-contained static HTML review page.

Pure post-processing -- reads a finalized steering run dir + QLoRA run dir
(the same two directories `comparison.build_comparison` reads), loads no
model, touches no GPU. Reuses `comparison`'s own read helpers so the
`quality_key`/behavior handling matches the headline report.

REBUILT 2026-07-26 from an earlier version that pinned exactly 4 outputs
per example (winner / baseline=0.0 / one mirrored negative / qlora),
following direct feedback: "I am only seeing coeff +1 and -1 and not the
other coefficients... ideally would want to see more examples alongside
not just the winners. And the associated numbers/values alongside each
card". All N swept coefficients are generated during a real run -- the
old exporter's 4-slot pinning was hiding data that already existed, not a
generation gap.

Row-pairing basis (unchanged): both arms' generation rows share
`example_id`, `split`, `model_id`, `recipe_id`.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .comparison import _quality_key, _read_json, _read_jsonl, _read_yaml, _steering_config_key

ANNOTATION_COLUMNS = ["selected_best_positive", "selected_best_negative", "selected_best_lora", "notes"]

# Column order for CSV/JSONL: context, then the selection summary (what a
# human picked, if anything -- this is what makes the CSV a useful export
# even before any review happens: it degrades to "everything blank" and
# still lists every config's own scores), then annotation columns last.
_CONTEXT_COLUMNS = [
    "model_id", "recipe_id", "behavior_id", "example_id", "split", "category",
    "benchmark", "is_safe_control", "prompt",
]

# Fields on a generation row that describe context/identity rather than a
# per-row score -- excluded from the "score" bundle attached to each card.
_NON_SCORE_ROW_KEYS = {
    "model_id", "model_name", "arm", "recipe_id", "behavior_id", "example_id", "split", "category",
    "method", "layer_idx", "coefficient", "token_scope", "prompt", "output", "batch_size",
    "num_train_records", "quantization", "dtype",
}


def _score_keys_present(rows: List[Dict[str, Any]]) -> List[str]:
    """Every extra key beyond the row schema's own fixed fields -- collected
    from whatever rows are actually present rather than hardcoded, so this
    module doesn't need updating every time a new behavior/scorer is added."""
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in _NON_SCORE_ROW_KEYS and key not in keys:
                keys.append(key)
    return sorted(keys)


def _card_scores(row: Dict[str, Any], score_keys: List[str]) -> Dict[str, Any]:
    return {key: row.get(key) for key in score_keys if row.get(key) is not None}


def _manifest_summary(steer_dir: str | Path) -> Dict[str, Any]:
    """Common manifest details for the package header: experiment name,
    seed, models (id/name_or_path/dtype/quantization), split fractions,
    n_sweep, and decoding.max_new_tokens -- the settings that apply to
    every card in the package rather than repeating per-example. Degrades
    to an empty dict (header simply omits the section) if
    resolved_manifest.yaml is missing, e.g. an older run predating it."""
    manifest = _read_yaml(Path(steer_dir) / "resolved_manifest.yaml")
    if not manifest:
        return {}
    experiment = manifest.get("experiment", {}) or {}
    splits = manifest.get("splits", {}) or {}
    decoding = manifest.get("decoding", {}) or {}
    return {
        "experiment_name": experiment.get("name"),
        "seed": experiment.get("seed"),
        "n_sweep": experiment.get("n_sweep"),
        "models": [
            {"id": m.get("id"), "name_or_path": m.get("name_or_path"),
             "dtype": m.get("dtype"), "quantization": m.get("quantization")}
            for m in manifest.get("models", []) or []
        ],
        "steer_fraction": splits.get("steer_fraction"),
        "validation_fraction": splits.get("validation_fraction"),
        "group_key": splits.get("group_key"),
        "max_new_tokens": decoding.get("max_new_tokens"),
    }


def build_human_eval_records(
    steering_run: str | Path, qlora_run: str | Path, splits: Tuple[str, ...] = ("validation", "test"),
) -> List[Dict[str, Any]]:
    """One record per (model_id, recipe_id, example_id, split) present in
    BOTH arms. Each record carries `steering_configs` -- one entry per
    (method, layer_idx, coefficient, token_scope) actually swept for that
    example, sorted by (method, layer, coefficient) for stable ordering --
    plus a single `qlora` entry (pinned to the max num_train_records point,
    matching `build_comparison`'s own pinning). `winner_config` (the
    validation-selected config `comparison.best_steering_config` would
    pick, if determinable) is included as a hint for which card to open
    first / pre-highlight, but every config is present regardless.

    A (model_id, recipe_id) pair with zero steering configs for a given
    example contributes no record for it (nothing to show); a pair with no
    overlap between the two arms at all contributes nothing, matching the
    old exporter's skip behavior.
    """
    steer_dir = Path(steering_run)
    qlora_dir = Path(qlora_run)

    steer_gen_rows = _read_jsonl(steer_dir / "results" / "generations.jsonl")
    qlora_gen_rows = _read_jsonl(qlora_dir / "results" / "generations.jsonl")
    qlora_results = _read_json(qlora_dir / "results" / "qlora.json")
    if isinstance(qlora_results, dict):
        qlora_results = []

    behavior_by_recipe: Dict[str, str] = {}
    for row in steer_gen_rows + qlora_gen_rows:
        if row.get("recipe_id") and row.get("behavior_id"):
            behavior_by_recipe.setdefault(row["recipe_id"], row["behavior_id"])

    pairs = sorted({(r.get("model_id"), r.get("recipe_id")) for r in steer_gen_rows} &
                   {(r.get("model_id"), r.get("recipe_id")) for r in qlora_gen_rows})

    steer_score_keys = _score_keys_present(steer_gen_rows)
    qlora_score_keys = _score_keys_present(qlora_gen_rows)

    records: List[Dict[str, Any]] = []
    for model_id, recipe_id in pairs:
        behavior_id = behavior_by_recipe.get(recipe_id)
        all_steer_rows = [r for r in steer_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
        all_qlora_rows = [r for r in qlora_gen_rows if r.get("model_id") == model_id and r.get("recipe_id") == recipe_id]
        if not all_steer_rows or not all_qlora_rows:
            continue

        qlora_ns = {r.get("num_train_records") for r in all_qlora_rows if r.get("num_train_records") is not None}
        max_qlora_n = max(qlora_ns) if qlora_ns else None
        qlora_rows_at_max_n = (
            [r for r in all_qlora_rows if r.get("num_train_records") == max_qlora_n]
            if max_qlora_n is not None else all_qlora_rows
        )
        qlora_by_key = {(r.get("example_id"), r.get("split")): r for r in qlora_rows_at_max_n}

        winner_cfg = None
        quality_key = _quality_key(behavior_id, all_steer_rows)
        if quality_key is not None:
            from .comparison import best_steering_config
            winner = best_steering_config(all_steer_rows, behavior_id)
            if winner is not None:
                winner_cfg = (winner["method"], winner["layer_idx"], winner["coefficient"], winner["token_scope"])

        by_example: Dict[Tuple[Any, str], List[Dict[str, Any]]] = {}
        for row in all_steer_rows:
            if row.get("split") not in splits:
                continue
            key = (row.get("example_id"), row.get("split"))
            by_example.setdefault(key, []).append(row)

        for (example_id, split), rows_for_example in sorted(by_example.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][0]))):
            qlora_row = qlora_by_key.get((example_id, split))
            if qlora_row is None:
                continue

            configs = sorted(
                rows_for_example,
                key=lambda r: (str(r.get("method")), r.get("layer_idx") or 0, r.get("coefficient") or 0),
            )
            first = configs[0]
            steering_configs = [
                {
                    "method": row.get("method"), "layer_idx": row.get("layer_idx"),
                    "coefficient": row.get("coefficient"), "token_scope": row.get("token_scope"),
                    "output": row.get("output"),
                    "is_baseline": row.get("coefficient") == 0.0,
                    "is_winner_config": winner_cfg is not None and _steering_config_key(row) == winner_cfg,
                    "scores": _card_scores(row, steer_score_keys),
                }
                for row in configs
            ]

            records.append({
                "model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                "example_id": example_id, "split": split,
                "category": first.get("category"), "benchmark": first.get("benchmark"),
                "is_safe_control": first.get("is_safe_control"), "prompt": first.get("prompt"),
                "steering_configs": steering_configs,
                "qlora": {"output": qlora_row.get("output"), "scores": _card_scores(qlora_row, qlora_score_keys)},
                "selected_best_positive": "", "selected_best_negative": "", "selected_best_lora": "",
                "notes": "",
            })

    return records


def _selection_summary_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens one record into the compact CSV/JSONL shape: every
    steering config gets its own column set (config_N_*), the qlora output/
    scores, and the (initially empty) human selections. This is the
    "expand to all coefficients" CSV shape -- unlike the old 4-pinned-slot
    version, an external tool importing pairs.csv sees the same full grid
    the HTML page does, not just the winner/baseline/one-negative subset."""
    flat: Dict[str, Any] = {
        "model_id": record["model_id"], "recipe_id": record["recipe_id"], "behavior_id": record["behavior_id"],
        "example_id": record["example_id"], "split": record["split"], "category": record.get("category"),
        "benchmark": record.get("benchmark"), "is_safe_control": record.get("is_safe_control"),
        "prompt": record.get("prompt"),
    }
    for i, cfg in enumerate(record["steering_configs"]):
        prefix = f"config_{i}"
        flat[f"{prefix}_method"] = cfg["method"]
        flat[f"{prefix}_layer_idx"] = cfg["layer_idx"]
        flat[f"{prefix}_coefficient"] = cfg["coefficient"]
        flat[f"{prefix}_token_scope"] = cfg["token_scope"]
        flat[f"{prefix}_output"] = cfg["output"]
        for score_key, value in cfg["scores"].items():
            flat[f"{prefix}_{score_key}"] = value
    flat["qlora_output"] = record["qlora"]["output"]
    for score_key, value in record["qlora"]["scores"].items():
        flat[f"qlora_{score_key}"] = value
    for col in ANNOTATION_COLUMNS:
        flat[col] = record.get(col, "")
    return flat


def _fieldnames(records: List[Dict[str, Any]]) -> List[str]:
    """Stable column order: fixed context columns, then every config_N_*/
    qlora_* column encountered (sorted for determinism within each config
    index), then annotation columns last. Built from the flattened rows,
    since the config columns are dynamic (N varies per example's own
    sweep)."""
    flat_rows = [_selection_summary_row(r) for r in records]
    extra: List[str] = []
    for row in flat_rows:
        for key in row:
            if key not in _CONTEXT_COLUMNS and key not in ANNOTATION_COLUMNS and key not in extra:
                extra.append(key)

    def _sort_key(key: str):
        if key.startswith("config_"):
            parts = key.split("_", 2)
            try:
                idx = int(parts[1])
            except (IndexError, ValueError):
                idx = 0
            return (0, idx, parts[2] if len(parts) > 2 else "")
        if key.startswith("qlora_"):
            return (1, 0, key)
        return (2, 0, key)

    return _CONTEXT_COLUMNS + sorted(extra, key=_sort_key) + ANNOTATION_COLUMNS


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = _fieldnames(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            flat = _selection_summary_row(record)
            writer.writerow({k: ("" if v is None else v) for k, v in flat.items()})


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")


def _render_review_html(records: List[Dict[str, Any]], manifest_summary: Dict[str, Any]) -> str:
    """Self-contained static page (no external requests) -- one example per
    screen, every swept coefficient as its own card, manifest details
    pinned at the top, click-to-select (not a dropdown) for best positive/
    negative/LoRA (baseline selectable in either steering slot), and a
    compact list view built from those selections."""

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    # `<` -> `<`: record content is real LLM output and can contain a
    # literal "</script>" sequence, which would close the script tag early
    # and inject the remainder as raw HTML -- see the original module's
    # identical note. json.dumps already handles quotes/backslashes.
    records_json = json.dumps(records, default=str).replace("<", "\\u003c")
    manifest_json = json.dumps(manifest_summary, default=str).replace("<", "\\u003c")

    manifest_bits = []
    if manifest_summary.get("experiment_name"):
        manifest_bits.append(f"<strong>{esc(manifest_summary['experiment_name'])}</strong>")
    if manifest_summary.get("seed") is not None:
        manifest_bits.append(f"seed={esc(manifest_summary['seed'])}")
    if manifest_summary.get("n_sweep"):
        manifest_bits.append(f"n_sweep={esc(manifest_summary['n_sweep'])}")
    if manifest_summary.get("steer_fraction") is not None:
        manifest_bits.append(
            f"splits: steer={esc(manifest_summary['steer_fraction'])} "
            f"val={esc(manifest_summary.get('validation_fraction'))}"
        )
    if manifest_summary.get("max_new_tokens") is not None:
        manifest_bits.append(f"max_new_tokens={esc(manifest_summary['max_new_tokens'])}")
    for m in manifest_summary.get("models", []) or []:
        manifest_bits.append(
            f"{esc(m.get('id'))}: {esc(m.get('name_or_path'))} "
            f"({esc(m.get('dtype'))}/{esc(m.get('quantization'))})"
        )
    manifest_header = (
        f'<div class="manifest-header">{" &middot; ".join(manifest_bits)}</div>' if manifest_bits else ""
    )

    return f"""<title>Human evaluation review</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 1.5rem; }}
h1 {{ font-size: 1.3rem; margin-bottom: 0.25rem; }}
.manifest-header {{ font-size: 0.8rem; opacity: 0.75; background: color-mix(in srgb, CanvasText 5%, Canvas); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 1rem; line-height: 1.6; }}
.toolbar {{ position: sticky; top: 0; background: Canvas; padding: 0.75rem 0; border-bottom: 1px solid color-mix(in srgb, CanvasText 20%, transparent); margin-bottom: 1rem; z-index: 2; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }}
button {{ font: inherit; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid color-mix(in srgb, CanvasText 30%, transparent); background: Canvas; color: CanvasText; cursor: pointer; }}
button:hover {{ background: color-mix(in srgb, CanvasText 8%, Canvas); }}
button.active-view {{ background: color-mix(in srgb, CanvasText 12%, Canvas); font-weight: 600; }}
.nav-group {{ display: flex; align-items: center; gap: 0.5rem; }}
.nav-pos {{ font-size: 0.85rem; opacity: 0.75; min-width: 5.5rem; text-align: center; }}
.spacer {{ flex: 1; }}

/* Detail view: one example per screen */
.example-meta {{ font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.5rem; }}
.prompt-box {{ margin-bottom: 1rem; padding: 0.75rem; border-radius: 8px; background: color-mix(in srgb, CanvasText 6%, Canvas); white-space: pre-wrap; }}
.prompt-box strong {{ display: block; margin-bottom: 0.25rem; font-size: 0.8rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.03em; }}
.section-label {{ font-size: 0.8rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.03em; margin: 1.25rem 0 0.5rem 0; }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.75rem; }}
.card {{ border: 1px solid color-mix(in srgb, CanvasText 15%, transparent); border-radius: 10px; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem; }}
.card.is-winner {{ border-color: color-mix(in srgb, Highlight 60%, CanvasText 15%); box-shadow: 0 0 0 1px color-mix(in srgb, Highlight 40%, transparent); }}
.card-title {{ font-size: 0.85rem; font-weight: 600; }}
.card-scores {{ font-size: 0.7rem; opacity: 0.75; display: flex; flex-wrap: wrap; gap: 0.4rem 0.6rem; }}
.card-output {{ white-space: pre-wrap; overflow-wrap: break-word; font-size: 0.85rem; background: color-mix(in srgb, CanvasText 4%, Canvas); border-radius: 6px; padding: 0.5rem; max-height: 14rem; overflow-y: auto; flex: 1; }}
.card-output.expanded {{ max-height: none; }}
.expand-btn {{ align-self: flex-start; font-size: 0.72rem; padding: 0.15rem 0.5rem; }}
.select-row {{ display: flex; gap: 0.35rem; flex-wrap: wrap; }}
.select-btn {{ font-size: 0.72rem; padding: 0.25rem 0.5rem; border-radius: 5px; }}
.select-btn.picked {{ background: Highlight; color: HighlightText; border-color: Highlight; }}
.qlora-card {{ border-color: color-mix(in srgb, CanvasText 25%, transparent); }}

/* List view */
#list-view table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
#list-view th, #list-view td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid color-mix(in srgb, CanvasText 12%, transparent); vertical-align: top; }}
#list-view th {{ position: sticky; top: 3.2rem; background: Canvas; }}
#list-view td.output-preview {{ max-width: 22rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
#list-view tr:hover {{ background: color-mix(in srgb, CanvasText 4%, Canvas); cursor: pointer; }}
.hidden {{ display: none !important; }}
textarea.notes {{ width: 100%; box-sizing: border-box; font: inherit; margin-top: 0.5rem; }}
</style>
<h1>Human evaluation review</h1>
{manifest_header}
<div class="toolbar">
  <button id="view-detail-btn" class="active-view">Detail view</button>
  <button id="view-list-btn">List view</button>
  <div class="nav-group" id="detail-nav">
    <button id="prev-btn">&larr; Prev</button>
    <span class="nav-pos" id="nav-pos"></span>
    <button id="next-btn">Next &rarr;</button>
  </div>
  <div class="spacer"></div>
  <button id="download-btn">Download annotations as CSV</button>
</div>
<div id="detail-view"></div>
<div id="list-view" class="hidden"></div>
<script>
const RECORDS = {records_json};
const MANIFEST = {manifest_json};
let current = 0;

function esc(v) {{
  const d = document.createElement('div');
  d.textContent = (v === null || v === undefined) ? '' : String(v);
  return d.innerHTML;
}}

function configTitle(cfg) {{
  let t = (cfg.method || '') + ' L' + cfg.layer_idx + ' c=' + cfg.coefficient + ' scope=' + cfg.token_scope;
  if (cfg.is_baseline) t += ' (baseline)';
  return t;
}}

function scoreBits(scores) {{
  return Object.entries(scores || {{}}).map(([k, v]) => {{
    const num = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
    return '<span>' + esc(k) + '=' + esc(num) + '</span>';
  }}).join('');
}}

function cardKeyForConfig(cfg, index) {{
  // Position-based, not value-based: a cross-recipe control (e.g. XSTest's
  // benign_over_refusal_control) can legitimately have two DIFFERENT
  // generations at the identical (method, layer, coefficient, token_scope)
  // -- one per source vector applied (e.g. its N=10 and N=40 extractions),
  // with no field on the row distinguishing which -- so a value-based key
  // would silently merge two distinct cards into one selectable target.
  return 'cfg' + index;
}}

function renderCard(record, cfg, cardKey, isLora) {{
  const selField = isLora ? 'selected_best_lora'
    : (cfg.coefficient < 0 ? 'selected_best_negative' : 'selected_best_positive');
  const picked = record[selField] === cardKey;
  const winnerClass = (!isLora && cfg.is_winner_config) ? ' is-winner' : '';
  const title = isLora ? 'QLoRA' : configTitle(cfg);
  const output = isLora ? record.qlora.output : cfg.output;
  const scores = isLora ? record.qlora.scores : cfg.scores;
  const selectButtons = isLora
    ? `<button class="select-btn${{picked ? ' picked' : ''}}" data-select="selected_best_lora" data-key="${{esc(cardKey)}}">Best LoRA</button>`
    : (cfg.coefficient <= 0
        ? `<button class="select-btn${{record.selected_best_negative === cardKey ? ' picked' : ''}}" data-select="selected_best_negative" data-key="${{esc(cardKey)}}">Best negative</button>
           <button class="select-btn${{record.selected_best_positive === cardKey ? ' picked' : ''}}" data-select="selected_best_positive" data-key="${{esc(cardKey)}}">Best positive</button>`
        : `<button class="select-btn${{record.selected_best_positive === cardKey ? ' picked' : ''}}" data-select="selected_best_positive" data-key="${{esc(cardKey)}}">Best positive</button>`);
  return `
    <div class="card${{isLora ? ' qlora-card' : ''}}${{winnerClass}}" data-card-key="${{esc(cardKey)}}">
      <div class="card-title">${{esc(title)}}</div>
      <div class="card-scores">${{scoreBits(scores)}}</div>
      <div class="card-output">${{esc(output)}}</div>
      <button class="expand-btn">Expand</button>
      <div class="select-row">${{selectButtons}}</div>
    </div>`;
}}

function renderDetail() {{
  const record = RECORDS[current];
  const container = document.getElementById('detail-view');
  if (!record) {{ container.innerHTML = '<p>No examples.</p>'; return; }}
  const configs = record.steering_configs || [];
  const cards = configs.map((cfg, i) => renderCard(record, cfg, cardKeyForConfig(cfg, i), false)).join('');
  const qloraCard = renderCard(record, null, '__qlora__', true);
  container.innerHTML = `
    <div class="example-meta">
      <strong>${{esc(record.model_id)}} / ${{esc(record.recipe_id)}}</strong>
      &middot; example <code>${{esc(record.example_id)}}</code>
      &middot; split <code>${{esc(record.split)}}</code>
      &middot; category <code>${{esc(record.category)}}</code>
    </div>
    <div class="prompt-box"><strong>Prompt</strong>${{esc(record.prompt)}}</div>
    <div class="section-label">Steering configs (${{configs.length}})</div>
    <div class="card-grid">${{cards}}</div>
    <div class="section-label">Fine-tuning</div>
    <div class="card-grid">${{qloraCard}}</div>
    <label>Notes<textarea class="notes" rows="2" data-notes-index="${{current}}">${{esc(record.notes)}}</textarea></label>
  `;
  document.getElementById('nav-pos').textContent = (current + 1) + ' / ' + RECORDS.length;
  container.querySelectorAll('.expand-btn').forEach((btn) => {{
    btn.addEventListener('click', () => {{
      const out = btn.previousElementSibling;
      out.classList.toggle('expanded');
      btn.textContent = out.classList.contains('expanded') ? 'Collapse' : 'Expand';
    }});
  }});
  container.querySelectorAll('[data-select]').forEach((btn) => {{
    btn.addEventListener('click', () => {{
      const field = btn.getAttribute('data-select');
      const key = btn.getAttribute('data-key');
      record[field] = (record[field] === key) ? '' : key;  // click again to deselect
      renderDetail();
    }});
  }});
  const notesEl = container.querySelector('[data-notes-index]');
  if (notesEl) notesEl.addEventListener('input', (e) => {{ record.notes = e.target.value; }});
}}

function pickedSummary(record, field) {{
  const key = record[field];
  if (!key) return '';
  if (key === '__qlora__') return 'QLoRA';
  const configs = record.steering_configs || [];
  const idx = configs.findIndex((c, i) => cardKeyForConfig(c, i) === key);
  return idx >= 0 ? configTitle(configs[idx]) : key;
}}

function renderList() {{
  const container = document.getElementById('list-view');
  const rows = RECORDS.map((r, i) => `
    <tr data-goto="${{i}}">
      <td>${{i + 1}}</td>
      <td>${{esc(r.model_id)}}</td>
      <td>${{esc(r.recipe_id)}}</td>
      <td>${{esc(r.example_id)}}</td>
      <td>${{esc(r.split)}}</td>
      <td>${{(r.steering_configs || []).length}}</td>
      <td>${{esc(pickedSummary(r, 'selected_best_positive'))}}</td>
      <td>${{esc(pickedSummary(r, 'selected_best_negative'))}}</td>
      <td>${{esc(pickedSummary(r, 'selected_best_lora'))}}</td>
      <td class="output-preview">${{esc((r.prompt || '').slice(0, 80))}}</td>
    </tr>`).join('');
  container.innerHTML = `
    <table>
      <thead><tr>
        <th>#</th><th>model</th><th>recipe</th><th>example</th><th>split</th><th>configs</th>
        <th>best +</th><th>best -</th><th>best LoRA</th><th>prompt</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>`;
  container.querySelectorAll('[data-goto]').forEach((tr) => {{
    tr.addEventListener('click', () => {{
      current = parseInt(tr.getAttribute('data-goto'), 10);
      showDetailView();
    }});
  }});
}}

function showDetailView() {{
  document.getElementById('detail-view').classList.remove('hidden');
  document.getElementById('detail-nav').classList.remove('hidden');
  document.getElementById('list-view').classList.add('hidden');
  document.getElementById('view-detail-btn').classList.add('active-view');
  document.getElementById('view-list-btn').classList.remove('active-view');
  renderDetail();
}}

function showListView() {{
  renderList();
  document.getElementById('detail-view').classList.add('hidden');
  document.getElementById('detail-nav').classList.add('hidden');
  document.getElementById('list-view').classList.remove('hidden');
  document.getElementById('view-list-btn').classList.add('active-view');
  document.getElementById('view-detail-btn').classList.remove('active-view');
}}

document.getElementById('view-detail-btn').addEventListener('click', showDetailView);
document.getElementById('view-list-btn').addEventListener('click', showListView);
document.getElementById('prev-btn').addEventListener('click', () => {{
  current = (current - 1 + RECORDS.length) % Math.max(RECORDS.length, 1);
  renderDetail();
}});
document.getElementById('next-btn').addEventListener('click', () => {{
  current = (current + 1) % Math.max(RECORDS.length, 1);
  renderDetail();
}});

function toCsvValue(v) {{
  if (v === null || v === undefined) v = '';
  v = String(v);
  if (/[",\\n]/.test(v)) v = '"' + v.replace(/"/g, '""') + '"';
  return v;
}}

function downloadCsv() {{
  const cols = ['model_id', 'recipe_id', 'example_id', 'split', 'selected_best_positive', 'selected_best_negative', 'selected_best_lora', 'notes'];
  const lines = [cols.join(',')];
  for (const r of RECORDS) {{
    lines.push(cols.map((c) => toCsvValue(
      c.startsWith('selected_') ? pickedSummary(r, c) : r[c]
    )).join(','));
  }}
  const blob = new Blob([lines.join('\\n')], {{ type: 'text/csv' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'human_eval_selections.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

document.getElementById('download-btn').addEventListener('click', downloadCsv);

showDetailView();
</script>
"""


def _render_readme(records: List[Dict[str, Any]], manifest_summary: Dict[str, Any]) -> str:
    n_configs = sum(len(r.get("steering_configs", [])) for r in records)
    lines = [
        "# Human evaluation package",
        "",
        f"{len(records)} examples across matched (model, recipe) pairs from a steering run and a QLoRA run "
        f"({n_configs} total steering-config generations across all examples).",
        "",
        "## Files",
        "",
        "- `pairs.jsonl` -- one record per example, each carrying the FULL set of swept steering configs (not",
        "  just a winner) plus the QLoRA output. This is the complete data; `pairs.csv` is a flattened view of it.",
        "- `pairs.csv` -- one row per example, one column-set per steering config (`config_0_*`, `config_1_*`, ...)",
        "  plus `qlora_*` and the (initially empty) selection columns. Import into a spreadsheet directly.",
        "- `review.html` -- self-contained static page (no server, no external requests). Detail view shows one",
        "  example per screen with every swept coefficient as its own card, plus manifest details at the top;",
        "  click a card's \"Best positive\" / \"Best negative\" / \"Best LoRA\" button to select it (baseline is",
        "  selectable the same way). List view is a compact table driven by those selections. A button downloads",
        "  your selections as CSV.",
        "",
        "## Record shape (pairs.jsonl)",
        "",
        "- `model_id`, `recipe_id`, `behavior_id`, `example_id`, `split`, `category`, `benchmark`,",
        "  `is_safe_control`, `prompt` -- context identifying the example.",
        "- `steering_configs` -- list of `{method, layer_idx, coefficient, token_scope, output, is_baseline,",
        "  is_winner_config, scores}`, one per swept config. `is_winner_config` flags the config",
        "  `comparison.best_steering_config` would select (a hint, not a restriction -- every config is present",
        "  regardless). `scores` is that config's own automated scores (e.g. `safe_refusal`, `perplexity_ratio_",
        "  vs_baseline`, `js_divergence_vs_baseline`), read directly off the stored generation row.",
        "- `qlora` -- `{output, scores}` from the matched-recipe QLoRA adapter at its largest trained example count.",
        "- `selected_best_positive`, `selected_best_negative`, `selected_best_lora` -- empty until a human selects",
        "  a card in `review.html`; then set to that config's key (`method|layer_idx|coefficient|token_scope`, or",
        "  `__qlora__`). Any card, including the baseline (`coefficient == 0`), is selectable in either steering slot.",
        "- `notes` -- free-text per-example notes.",
    ]
    if manifest_summary:
        lines += ["", "## Manifest summary", "", "```json", json.dumps(manifest_summary, indent=2, default=str), "```"]
    return "\n".join(lines) + "\n"


def write_human_eval_package(
    records: List[Dict[str, Any]], output_root: str | Path, manifest_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes `human_eval/{pairs.jsonl,pairs.csv,review.html,README.md}`
    under `output_root` (the same directory `write_comparison_report`
    writes `report.md`/`report.json` to). Returns the `human_eval/`
    directory path."""
    output = Path(output_root) / "human_eval"
    output.mkdir(parents=True, exist_ok=True)
    manifest_summary = manifest_summary or {}
    _write_jsonl(records, output / "pairs.jsonl")
    _write_csv(records, output / "pairs.csv")
    (output / "review.html").write_text(_render_review_html(records, manifest_summary), encoding="utf-8")
    (output / "README.md").write_text(_render_readme(records, manifest_summary), encoding="utf-8")
    return output
