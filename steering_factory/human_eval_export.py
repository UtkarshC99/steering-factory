"""Human-evaluation onboarding package: pairs each held-out example's
steering-winner, steering-baseline, and QLoRA outputs side by side, plus
empty annotation columns, and writes them as CSV/JSONL/a self-contained
static HTML review sheet.

Pure post-processing -- reads a finalized steering run dir + QLoRA run dir
(the same two directories `comparison.build_comparison` reads), loads no
model, touches no GPU. Reuses `comparison`'s own read helpers and
`best_steering_config` so the "winner" pinned here is the exact same
validation-selected config the headline comparison report already reports,
not a second, possibly-divergent selection.

Row-pairing basis (see the exploration behind this module): both arms'
generation rows share `example_id`, `split`, `model_id`, `recipe_id` --
enough to join exactly. A steering run has one row per example PER
(method, layer_idx, coefficient, token_scope) grid cell, so a single
steering output must be pinned to one config before it can sit in a single
column: the validation-selected winner (`best_steering_config`) and the
unsteered baseline (`coefficient == 0.0`, same method/layer/token_scope as
the winner) are the two steering columns; QLoRA contributes the third.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .comparison import _quality_key, _read_json, _read_jsonl, _steering_config_key, best_steering_config

ANNOTATION_COLUMNS = [
    "rating_steering_winner", "rating_steering_baseline", "rating_qlora", "preferred_arm", "notes",
]

# Column order for CSV/JSONL -- context first, then the three outputs, then
# each arm's own auto-scores, then the empty annotation columns last (so a
# human importing into a spreadsheet sees prompt/outputs first and doesn't
# have to scroll past scoring internals to find where to write).
_CONTEXT_COLUMNS = [
    "model_id", "recipe_id", "behavior_id", "example_id", "split", "category",
    "benchmark", "is_safe_control", "prompt",
]
_CONFIG_COLUMNS = ["winner_method", "winner_layer_idx", "winner_coefficient", "winner_token_scope"]
_OUTPUT_COLUMNS = ["steering_winner_output", "steering_baseline_output", "qlora_output"]


def _score_keys_present(rows: List[Dict[str, Any]]) -> List[str]:
    """Every extra key beyond the row schema's own fixed fields -- the
    behavior-score keys `_behavior_score` merged in (e.g. `safe_refusal`,
    `leaf_exact_match`), which vary by behavior family. Collected from
    whatever rows are actually present rather than hardcoded, so this
    module doesn't need updating every time a new behavior/scorer is
    added."""
    known = {
        "model_id", "model_name", "arm", "recipe_id", "behavior_id", "example_id", "split", "category",
        "method", "layer_idx", "coefficient", "token_scope", "prompt", "output", "latency_s", "batch_size",
        "batch_wall_time_s", "tokens_generated", "perplexity", "perplexity_ratio_vs_baseline",
        "js_divergence_vs_baseline", "repetition_score", "distinct_2", "benchmark", "is_safe_control",
        "num_train_records", "quantization", "dtype",
    }
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in known and key not in keys:
                keys.append(key)
    return sorted(keys)


def build_human_eval_records(
    steering_run: str | Path, qlora_run: str | Path, splits: Tuple[str, ...] = ("validation", "test"),
) -> List[Dict[str, Any]]:
    """One record per (model_id, recipe_id, example_id, split) present in
    BOTH arms, for each split in `splits`. Uses `best_steering_config` (the
    same function the headline comparison report uses) to pick which
    steering config's rows populate the "winner" column, so a human
    reviewing this package is looking at the exact config the automated
    report calls the winner -- not a second, possibly different pick.

    A (model_id, recipe_id) pair with no validation data (so
    `best_steering_config` can't select a winner) contributes no records --
    mirrors `build_comparison`'s own skip behavior for the same reason.
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

        winner = best_steering_config(all_steer_rows, behavior_id)
        if winner is None:
            continue
        winner_cfg = (winner["method"], winner["layer_idx"], winner["coefficient"], winner["token_scope"])
        baseline_cfg = (winner["method"], winner["layer_idx"], 0.0, winner["token_scope"])

        # QLoRA pinned to the max num_train_records point, matching
        # build_comparison's own "best shot" pinning for the headline table.
        qlora_ns = {r.get("num_train_records") for r in all_qlora_rows if r.get("num_train_records") is not None}
        max_qlora_n = max(qlora_ns) if qlora_ns else None
        qlora_rows_at_max_n = (
            [r for r in all_qlora_rows if r.get("num_train_records") == max_qlora_n]
            if max_qlora_n is not None else all_qlora_rows
        )

        winner_by_key = {(r.get("example_id"), r.get("split")): r for r in all_steer_rows if _steering_config_key(r) == winner_cfg}
        baseline_by_key = {(r.get("example_id"), r.get("split")): r for r in all_steer_rows if _steering_config_key(r) == baseline_cfg}
        qlora_by_key = {(r.get("example_id"), r.get("split")): r for r in qlora_rows_at_max_n}

        example_keys = sorted(
            {k for k in winner_by_key if k[1] in splits} &
            {k for k in qlora_by_key if k[1] in splits}
        )
        for example_id, split in example_keys:
            winner_row = winner_by_key.get((example_id, split))
            baseline_row = baseline_by_key.get((example_id, split))
            qlora_row = qlora_by_key.get((example_id, split))
            if winner_row is None or qlora_row is None:
                continue

            record: Dict[str, Any] = {
                "model_id": model_id, "recipe_id": recipe_id, "behavior_id": behavior_id,
                "example_id": example_id, "split": split,
                "category": winner_row.get("category"), "benchmark": winner_row.get("benchmark"),
                "is_safe_control": winner_row.get("is_safe_control"), "prompt": winner_row.get("prompt"),
                "winner_method": winner["method"], "winner_layer_idx": winner["layer_idx"],
                "winner_coefficient": winner["coefficient"], "winner_token_scope": winner["token_scope"],
                "steering_winner_output": winner_row.get("output"),
                "steering_baseline_output": baseline_row.get("output") if baseline_row else None,
                "qlora_output": qlora_row.get("output"),
            }
            for key in steer_score_keys:
                record[f"steering_winner_{key}"] = winner_row.get(key)
                record[f"steering_baseline_{key}"] = baseline_row.get(key) if baseline_row else None
            for key in qlora_score_keys:
                record[f"qlora_{key}"] = qlora_row.get(key)
            for col in ANNOTATION_COLUMNS:
                record[col] = ""
            records.append(record)

    return records


def _fieldnames(records: List[Dict[str, Any]]) -> List[str]:
    """Stable column order: context, config, outputs, then every
    score-derived column encountered (sorted for determinism), then
    annotation columns last."""
    extra: List[str] = []
    for record in records:
        for key in record:
            if key not in _CONTEXT_COLUMNS and key not in _CONFIG_COLUMNS and key not in _OUTPUT_COLUMNS \
               and key not in ANNOTATION_COLUMNS and key not in extra:
                extra.append(key)
    return _CONTEXT_COLUMNS + _CONFIG_COLUMNS + _OUTPUT_COLUMNS + sorted(extra) + ANNOTATION_COLUMNS


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = _fieldnames(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if v is None else v) for k, v in record.items()})


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")


def _render_review_html(records: List[Dict[str, Any]]) -> str:
    """Self-contained static page -- inline CSS/JS only, no external
    requests (no CDN script/stylesheet/font/image references), so it opens
    correctly from a plain file:// path or any static host. One card per
    example with the three outputs side by side, the auto-scores, and
    rating/notes inputs; a "Download annotations as CSV" button serializes
    the in-page form state via a client-side Blob -- nothing is sent
    anywhere. Theme-aware (prefers-color-scheme) matching the artifact
    conventions used elsewhere in this project."""
    fieldnames = _fieldnames(records)

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    cards = []
    for i, record in enumerate(records):
        score_bits = []
        for key in fieldnames:
            if key.startswith("steering_winner_") and key != "steering_winner_output":
                short = key[len("steering_winner_"):]
                if record.get(key) is not None:
                    score_bits.append(f'<span class="score">winner.{esc(short)}={esc(record[key])}</span>')
            elif key.startswith("qlora_") and key != "qlora_output":
                short = key[len("qlora_"):]
                if record.get(key) is not None:
                    score_bits.append(f'<span class="score">qlora.{esc(short)}={esc(record[key])}</span>')
        cards.append(f"""
<div class="card" data-index="{i}">
  <div class="meta">
    <strong>{esc(record.get('model_id'))} / {esc(record.get('recipe_id'))}</strong>
    &middot; example <code>{esc(record.get('example_id'))}</code>
    &middot; split <code>{esc(record.get('split'))}</code>
    &middot; category <code>{esc(record.get('category'))}</code>
  </div>
  <div class="prompt"><strong>Prompt:</strong> {esc(record.get('prompt'))}</div>
  <div class="outputs">
    <div class="output-col">
      <h4>Steering (winner: {esc(record.get('winner_method'))} L{esc(record.get('winner_layer_idx'))}
        c={esc(record.get('winner_coefficient'))})</h4>
      <div class="output-text">{esc(record.get('steering_winner_output'))}</div>
      <label>Rating <input type="text" class="rating" data-field="rating_steering_winner" placeholder="e.g. 1-5 or pass/fail"></label>
    </div>
    <div class="output-col">
      <h4>Steering baseline (coefficient=0)</h4>
      <div class="output-text">{esc(record.get('steering_baseline_output'))}</div>
      <label>Rating <input type="text" class="rating" data-field="rating_steering_baseline" placeholder="e.g. 1-5 or pass/fail"></label>
    </div>
    <div class="output-col">
      <h4>QLoRA</h4>
      <div class="output-text">{esc(record.get('qlora_output'))}</div>
      <label>Rating <input type="text" class="rating" data-field="rating_qlora" placeholder="e.g. 1-5 or pass/fail"></label>
    </div>
  </div>
  <div class="scores">{' '.join(score_bits)}</div>
  <div class="annotation-row">
    <label>Preferred arm
      <select class="preferred-arm" data-field="preferred_arm">
        <option value=""></option>
        <option value="steering_winner">steering (winner)</option>
        <option value="steering_baseline">steering (baseline)</option>
        <option value="qlora">qlora</option>
        <option value="tie">tie</option>
        <option value="neither">neither</option>
      </select>
    </label>
    <label class="notes-label">Notes
      <textarea class="notes" data-field="notes" rows="2"></textarea>
    </label>
  </div>
</div>""")

    # `<` -> `<` (a standard JSON-in-<script> escape): record content
    # comes from real LLM generations and can contain arbitrary text,
    # including a literal "</script>" sequence -- embedded unescaped, that
    # closes the script tag early and injects the remainder of the JSON
    # blob (and anything an adversarial generation crafted after it) as raw
    # HTML into the page. Escaping every `<` (not just in "</script>")
    # is the safe, standard fix -- JSON's own escaping already handles
    # quotes/backslashes/control characters, this only adds the one
    # character `json.dumps` doesn't escape that's dangerous specifically
    # inside a <script> block.
    records_json = json.dumps(records, default=str).replace("<", "\\u003c")

    return f"""<title>Human evaluation review</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; padding: 1.5rem; }}
h1 {{ font-size: 1.3rem; }}
.toolbar {{ position: sticky; top: 0; background: Canvas; padding: 0.75rem 0; border-bottom: 1px solid color-mix(in srgb, CanvasText 20%, transparent); margin-bottom: 1rem; z-index: 1; }}
button {{ font: inherit; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid color-mix(in srgb, CanvasText 30%, transparent); background: Canvas; color: CanvasText; cursor: pointer; }}
button:hover {{ background: color-mix(in srgb, CanvasText 8%, Canvas); }}
.card {{ border: 1px solid color-mix(in srgb, CanvasText 15%, transparent); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
.meta {{ font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.5rem; }}
.prompt {{ margin-bottom: 0.75rem; white-space: pre-wrap; }}
.outputs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }}
@media (max-width: 800px) {{ .outputs {{ grid-template-columns: 1fr; }} }}
.output-col h4 {{ margin: 0 0 0.35rem 0; font-size: 0.85rem; }}
.output-text {{ white-space: pre-wrap; background: color-mix(in srgb, CanvasText 5%, Canvas); border-radius: 6px; padding: 0.5rem; min-height: 3rem; font-size: 0.9rem; overflow-wrap: break-word; }}
.output-col label {{ display: block; margin-top: 0.4rem; font-size: 0.8rem; }}
.rating {{ width: 100%; box-sizing: border-box; }}
.scores {{ margin-top: 0.5rem; font-size: 0.75rem; opacity: 0.7; }}
.score {{ margin-right: 0.75rem; }}
.annotation-row {{ display: flex; gap: 1rem; margin-top: 0.75rem; flex-wrap: wrap; }}
.notes-label {{ flex: 1; min-width: 200px; }}
.notes {{ width: 100%; box-sizing: border-box; font: inherit; }}
select, input[type=text] {{ font: inherit; }}
</style>
<h1>Human evaluation review</h1>
<p>{len(records)} paired examples. Ratings/notes are saved only in this page's memory until you click
"Download annotations as CSV" -- nothing is sent anywhere.</p>
<div class="toolbar">
  <button id="download-btn">Download annotations as CSV</button>
</div>
<div id="cards">
{''.join(cards)}
</div>
<script>
const RECORDS = {records_json};
const FIELDNAMES = {json.dumps(fieldnames)};

function collectAnnotations() {{
  const cards = document.querySelectorAll('.card');
  const out = RECORDS.map((r) => Object.assign({{}}, r));
  cards.forEach((card) => {{
    const idx = parseInt(card.getAttribute('data-index'), 10);
    card.querySelectorAll('[data-field]').forEach((el) => {{
      out[idx][el.getAttribute('data-field')] = el.value;
    }});
  }});
  return out;
}}

function toCsvValue(v) {{
  if (v === null || v === undefined) v = '';
  v = String(v);
  if (/[",\\n]/.test(v)) v = '"' + v.replace(/"/g, '""') + '"';
  return v;
}}

function downloadCsv() {{
  const rows = collectAnnotations();
  const lines = [FIELDNAMES.map(toCsvValue).join(',')];
  for (const row of rows) {{
    lines.push(FIELDNAMES.map((f) => toCsvValue(row[f])).join(','));
  }}
  const blob = new Blob([lines.join('\\n')], {{ type: 'text/csv' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'human_eval_annotations.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

document.getElementById('download-btn').addEventListener('click', downloadCsv);
</script>
"""


def _render_readme(records: List[Dict[str, Any]]) -> str:
    fieldnames = _fieldnames(records) if records else []
    lines = [
        "# Human evaluation package",
        "",
        f"{len(records)} paired examples across matched (model, recipe) pairs from a steering run and a QLoRA run.",
        "",
        "## Files",
        "",
        "- `pairs.jsonl` / `pairs.csv` -- one row per (model, recipe, example, split). Import directly into a",
        "  spreadsheet, Label Studio, or Argilla.",
        "- `review.html` -- self-contained static page (no server, no external requests) for eyeballing outputs",
        "  side by side and entering ratings/notes; a button downloads your annotations as CSV.",
        "",
        "## Columns",
        "",
        "- `model_id`, `recipe_id`, `behavior_id`, `example_id`, `split`, `category`, `benchmark`, `is_safe_control`,",
        "  `prompt` -- context identifying the example.",
        "- `winner_method`, `winner_layer_idx`, `winner_coefficient`, `winner_token_scope` -- the validation-selected",
        "  steering config populating the `steering_winner_*` columns (same winner the automated comparison report",
        "  picks -- see `comparison.best_steering_config`).",
        "- `steering_winner_output` -- generated text from the best steering config.",
        "- `steering_baseline_output` -- generated text with the SAME method/layer/token_scope but coefficient=0",
        "  (i.e. no steering applied) -- what steering added over doing nothing.",
        "- `qlora_output` -- generated text from the matched-recipe QLoRA adapter, at its largest trained example count.",
        "- `steering_winner_<score>` / `steering_baseline_<score>` / `qlora_<score>` -- each arm's own automated",
        "  scores (e.g. `safe_refusal`, `leaf_exact_match`), for reference next to a human's own judgment.",
        "- `rating_steering_winner`, `rating_steering_baseline`, `rating_qlora`, `preferred_arm`, `notes` -- empty",
        "  annotation columns for a human reviewer to fill in.",
    ]
    if fieldnames:
        lines += ["", "## Full column order", "", "```", ", ".join(fieldnames), "```"]
    return "\n".join(lines) + "\n"


def write_human_eval_package(records: List[Dict[str, Any]], output_root: str | Path) -> Path:
    """Writes `human_eval/{pairs.jsonl,pairs.csv,review.html,README.md}`
    under `output_root` (the same directory `write_comparison_report`
    writes `report.md`/`report.json` to). Returns the `human_eval/`
    directory path."""
    output = Path(output_root) / "human_eval"
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(records, output / "pairs.jsonl")
    _write_csv(records, output / "pairs.csv")
    (output / "review.html").write_text(_render_review_html(records), encoding="utf-8")
    (output / "README.md").write_text(_render_readme(records), encoding="utf-8")
    return output
