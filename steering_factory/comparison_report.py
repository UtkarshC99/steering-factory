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


def _winner(entry: Dict[str, Any]) -> str:
    steering = entry["steering"]
    qlora = entry["qlora"]
    sq, qq = steering.get("test_quality"), qlora.get("test_quality")
    if sq is None or qq is None:
        return "n/a"
    if steering.get("beat_baseline") is False:
        # The winning steered config didn't even beat doing nothing --
        # declaring it "the winner" over QLoRA would be reporting the base
        # model's quality as if steering achieved it.
        return "inconclusive (steering did not beat baseline)"
    return "steering" if sq > qq else ("qlora" if qq > sq else "tie")


def render_comparison_markdown(comparison: Dict[str, Any]) -> str:
    lines = ["# Steering vs QLoRA comparison", "",
             f"- Steering run: `{comparison.get('steering_run')}`",
             f"- QLoRA run: `{comparison.get('qlora_run')}`",
             f"- Minimum held-out split size to report a result: {comparison.get('min_split_size', 'n/a')}", ""]

    entries = comparison.get("comparisons", [])
    excluded = comparison.get("excluded", [])

    if not entries and not excluded:
        lines.append("No matched (model, recipe) pairs found across both runs.")
        return "\n".join(lines)

    if entries:
        lines.append("## Matched quality (held out on test split; config selected on validation)")
        lines.append("")
        lines.append("| model | recipe | steering quality | baseline quality | steering config | qlora quality | n_test | winner |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for entry in entries:
            steering = entry["steering"]
            qlora = entry["qlora"]
            sq, qq = steering.get("test_quality"), qlora.get("test_quality")
            n_test = min(steering.get("n_test", 0), qlora.get("n_test", 0))
            config = f"{steering.get('method')} L{steering.get('layer_idx')} c={steering.get('coefficient')} scope={steering.get('token_scope')}"
            lines.append(
                f"| {entry['model_id']} | {entry['recipe_id']} | {_fmt(sq)} | {_fmt(steering.get('baseline_quality'))} | "
                f"{config} | {_fmt(qq)} | {n_test} | {_winner(entry)} |"
            )
        lines.append("")

        lines.append("## Cost")
        lines.append("")
        lines.append("| model | recipe | arm | one-time cost (s) | labeled examples | artifact bytes | ms/token |")
        lines.append("|---|---|---|---|---|---|---|")
        for entry in entries:
            steering = entry["steering"]
            qlora = entry["qlora"]
            lines.append(
                f"| {entry['model_id']} | {entry['recipe_id']} | steering | "
                f"{_fmt(steering.get('selected_config_extraction_cost_s'), 2)} | {_fmt(steering.get('labeled_examples'), 0)} | "
                f"{_fmt(steering.get('artifact_bytes'), 0)} | {_fmt(steering.get('per_request_ms_per_token'), 2)} |"
            )
            lines.append(
                f"| {entry['model_id']} | {entry['recipe_id']} | qlora | "
                f"{_fmt(qlora.get('wall_time_s'), 2)} | {_fmt(qlora.get('labeled_examples'), 0)} | "
                f"{_fmt(qlora.get('artifact_bytes'), 0)} | {_fmt(qlora.get('per_request_ms_per_token'), 2)} |"
            )
        lines.append("")
        lines.append(
            "_One-time cost: steering's is its selected vector's estimated extraction share "
            "(the full sweep's wall time, amortized across every config evaluated -- see "
            "`full_sweep_wall_time_s` in report.json for the whole exploration cost, which is "
            "NOT comparable to QLoRA's single training run); QLoRA's is its one training + eval run. "
            "Both arms' `ms/token` is the fair, directly comparable per-request inference cost._"
        )
        lines.append("")

    if excluded:
        lines.append("## Excluded (insufficient data)")
        lines.append("")
        lines.append(
            f"These (model, recipe) pairs had fewer than {comparison.get('min_split_size', 'n/a')} "
            "held-out examples in at least one split for at least one arm -- a quality number computed "
            "on that few examples is noise, not signal, so no winner is reported."
        )
        lines.append("")
        lines.append("| model | recipe | n_test (steering) | n_validation (steering) | n_test (qlora) | n_validation (qlora) |")
        lines.append("|---|---|---|---|---|---|")
        for entry in excluded:
            lines.append(
                f"| {entry['model_id']} | {entry['recipe_id']} | {entry['n_test_steering']} | "
                f"{entry['n_validation_steering']} | {entry['n_test_qlora']} | {entry['n_validation_qlora']} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_comparison_report(comparison: Dict[str, Any], output_root: str | Path) -> Path:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    markdown_path = output / "report.md"
    markdown_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    return markdown_path
