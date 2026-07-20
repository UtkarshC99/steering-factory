"""Renders `comparison.build_comparison()` output as Markdown, and writes
the accompanying JSON. Kept separate from `comparison.py` so the pure
data-joining logic (easy to unit test without touching disk formatting) is
not tangled with presentation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_comparison_markdown(comparison: Dict[str, Any]) -> str:
    lines = ["# Steering vs QLoRA comparison", "",
             f"- Steering run: `{comparison.get('steering_run')}`",
             f"- QLoRA run: `{comparison.get('qlora_run')}`", ""]

    entries = comparison.get("comparisons", [])
    if not entries:
        lines.append("No matched (model, recipe) pairs found across both runs.")
        return "\n".join(lines)

    lines.append("## Matched quality (held out on test split; config selected on validation)")
    lines.append("")
    lines.append("| model | recipe | steering quality | steering config | qlora quality | winner |")
    lines.append("|---|---|---|---|---|---|")
    for entry in entries:
        steering = entry["steering"]
        qlora = entry["qlora"]
        sq, qq = steering.get("test_quality"), qlora.get("test_quality")
        winner = "n/a"
        if sq is not None and qq is not None:
            winner = "steering" if sq > qq else ("qlora" if qq > sq else "tie")
        config = f"{steering.get('method')} L{steering.get('layer_idx')} c={steering.get('coefficient')} scope={steering.get('token_scope')}"
        lines.append(f"| {entry['model_id']} | {entry['recipe_id']} | {_fmt(sq)} | {config} | {_fmt(qq)} | {winner} |")
    lines.append("")

    lines.append("## Cost (matched-quality configs above)")
    lines.append("")
    lines.append("| model | recipe | arm | wall time (s) | labeled examples | artifact bytes | ms/token |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry in entries:
        for arm_name in ("steering", "qlora"):
            arm = entry[arm_name]
            lines.append(
                f"| {entry['model_id']} | {entry['recipe_id']} | {arm_name} | "
                f"{_fmt(arm.get('wall_time_s'), 2)} | {_fmt(arm.get('labeled_examples'), 0)} | "
                f"{_fmt(arm.get('artifact_bytes'), 0)} | {_fmt(arm.get('per_request_ms_per_token'), 2)} |"
            )
    lines.append("")
    lines.append("_Note: cost is reported for the QLoRA arm's single trained config and the steering "
                  "arm's validation-selected winning config, not summed across the full sweep._")
    return "\n".join(lines)


def write_comparison_report(comparison: Dict[str, Any], output_root: str | Path) -> Path:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    markdown_path = output / "report.md"
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    return markdown_path
