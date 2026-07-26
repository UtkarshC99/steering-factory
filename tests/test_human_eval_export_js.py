"""Executes the actual JS embedded in review.html under Node, rather than
grepping the source for patterns -- two real bugs (a null-dereference on
the QLoRA card, and duplicate card keys silently merging two distinct
generations into one selectable target) were found by manually clicking
through a real rendered page and were NOT visible to any Python-side
test, since build_human_eval_records' output was correct; only the
client-side rendering logic was broken. This file is the regression net
for exactly that class of bug.

Requires `node` on PATH; skipped if unavailable rather than failing CI on
an environment that lacks it.
"""
import json
import re
import shutil
import subprocess

import pytest

from steering_factory.human_eval_export import _render_review_html

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _extract_script(html_text: str) -> str:
    match = re.search(r"<script>(.*)</script>", html_text, re.DOTALL)
    assert match, "no <script> block found in rendered HTML"
    return match.group(1)


# Minimal DOM stub -- only what esc()/renderCard()/renderDetail() actually
# touch. Not a general DOM shim; extend only if a specific test needs more.
_DOM_STUB = """
function makeStubElement() {
  return {
    _innerHTML: '',
    set innerHTML(v) { this._innerHTML = v; },
    get innerHTML() { return this._innerHTML; },
    addEventListener: () => {},
    classList: { add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false },
    querySelectorAll: () => [],
    querySelector: () => null,
    setAttribute: () => {},
    getAttribute: () => null,
    appendChild: () => {},
    remove: () => {},
  };
}
const document = {
  createElement: () => {
    let _text = '';
    return {
      set textContent(v) { _text = String(v); },
      get textContent() { return _text; },
      get innerHTML() {
        return _text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                     .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
      },
      appendChild: () => {},
      remove: () => {},
      click: () => {},
      set href(v) {},
      set download(v) {},
    };
  },
  getElementById: () => makeStubElement(),
  querySelectorAll: () => [],
  querySelector: () => null,
  body: { appendChild: () => {} },
};
const URL = { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} };
const Blob = function() {};
"""


def _two_configs_same_key_record():
    """Mirrors a real run's benign_over_refusal_control shape: two
    generation rows with IDENTICAL (method, layer_idx, coefficient,
    token_scope) but different output/scores -- e.g. from two different
    source vectors (N=10 vs N=40) applied via apply_vectors_from, which
    the stored row has no field distinguishing."""
    return {
        "model_id": "m1", "recipe_id": "r1", "behavior_id": "b1",
        "example_id": "e1", "split": "test", "category": "c", "benchmark": "b",
        "is_safe_control": True, "prompt": "p",
        "steering_configs": [
            {"method": "mean_diff", "layer_idx": 5, "coefficient": -2.0, "token_scope": "all",
             "output": "first-generation", "is_baseline": False, "is_winner_config": False, "scores": {}},
            {"method": "mean_diff", "layer_idx": 5, "coefficient": -2.0, "token_scope": "all",
             "output": "second-generation", "is_baseline": False, "is_winner_config": False, "scores": {}},
        ],
        "qlora": {"output": "qlora-out", "scores": {}},
        "selected_best_positive": "", "selected_best_negative": "", "selected_best_lora": "", "notes": "",
    }


def _run_node(script: str) -> str:
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"node script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    return result.stdout.strip()


def test_card_key_for_config_is_unique_across_identical_configs():
    """Regression: two configs with the SAME (method, layer, coefficient,
    token_scope) but different generations must get DIFFERENT card keys --
    the original bug used a value-based key, so clicking "select" on one
    silently also selected the other, and the UI showed both cards
    highlighted despite the user clicking only one."""
    html_text = _render_review_html([_two_configs_same_key_record()], {})
    script = _extract_script(html_text)
    node_script = _DOM_STUB + script + """
const record = RECORDS[0];
const keys = record.steering_configs.map((cfg, i) => cardKeyForConfig(cfg, i));
console.log(JSON.stringify(keys));
"""
    keys = json.loads(_run_node(node_script))
    assert len(keys) == len(set(keys)), f"duplicate card keys for distinct configs: {keys}"


def test_render_card_does_not_crash_for_qlora_with_null_config():
    """Regression: renderCard(record, null, '__qlora__', true) crashed with
    "Cannot read properties of null (reading 'is_winner_config')" because
    the winner-class computation read cfg.is_winner_config unconditionally
    before the isLora branch guarded it -- this silently broke the ENTIRE
    detail view (one uncaught exception in renderDetail aborted the whole
    render, leaving every card's <div id="detail-view"> empty), not just
    the QLoRA card, which is why it was worth a dedicated test."""
    html_text = _render_review_html([_two_configs_same_key_record()], {})
    script = _extract_script(html_text)
    node_script = _DOM_STUB + script + """
const record = RECORDS[0];
const out = renderCard(record, null, '__qlora__', true);
console.log(JSON.stringify({ok: true, hasQlora: out.includes('QLoRA')}));
"""
    result = json.loads(_run_node(node_script))
    assert result["ok"]
    assert result["hasQlora"]


def test_render_detail_does_not_throw_for_a_real_record_shape():
    """End-to-end smoke test: renderDetail() must not throw for the
    ordinary shape build_human_eval_records produces (steering configs +
    a qlora entry), the exact call that was broken."""
    html_text = _render_review_html([_two_configs_same_key_record()], {})
    script = _extract_script(html_text)
    node_script = _DOM_STUB + script + """
try {
  renderDetail();
  console.log(JSON.stringify({ok: true}));
} catch (e) {
  console.log(JSON.stringify({ok: false, error: e.message}));
}
"""
    result = json.loads(_run_node(node_script))
    assert result["ok"], result.get("error")


def test_picked_summary_resolves_by_position_not_value():
    record = _two_configs_same_key_record()
    record["selected_best_negative"] = "cfg1"  # the SECOND config, by position
    html_text = _render_review_html([record], {})
    script = _extract_script(html_text)
    node_script = _DOM_STUB + script + """
console.log(JSON.stringify({summary: pickedSummary(RECORDS[0], 'selected_best_negative')}));
"""
    result = json.loads(_run_node(node_script))
    assert "c=-2" in result["summary"]  # resolves to a real config, not the literal key string
