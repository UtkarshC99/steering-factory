"""Renders `comparison.build_comparison()` output as Markdown, and writes
the accompanying JSON. Kept separate from `comparison.py` so the pure
data-joining logic (easy to unit test without touching disk formatting) is
not tangled with presentation.

`render_comparison_markdown` itself has no plotting dependency -- it only
knows how to format an already-computed `{"model__recipe": [filenames]}`
mapping (produced by `comparison_plots.render_comparison_plots`, which
requires matplotlib) into Markdown image links. This keeps the plotting
library import confined to `comparison_plots.py`, the same isolation
`live_plot.py` already applies for plotly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{value:.1%}"


# Same abstention thresholds as a config-selection gate would use, but
# applied at the WINNER-NAMING step instead: a run on 2026-07-26 reported
# QLoRA as the winner (safe_refusal=1.0) while it falsely refused 68-96%
# of benign prompts and had collapsed to 1-6 distinct outputs -- both
# facts were computed (safety_summary.json / comparison.py) but never
# reached this function, so the report read backwards. These thresholds
# gate the FINAL winner declaration regardless of which arm produced the
# number, so a degenerate arm can never be named the winner silently.
MAX_FALSE_REFUSAL_RATE_FOR_A_WINNER = 0.5
MIN_DISTINCT_OUTPUT_RATIO_FOR_A_WINNER = 0.10


def _degeneracy_reason(arm_name: str, arm: Dict[str, Any]) -> Optional[str]:
    """Returns a human-readable reason this arm cannot be named a winner,
    or None if it passes both checks. Order matters for the message: false
    refusal is checked first because it is the more informative failure
    (a low distinct-output ratio is often just a SYMPTOM of always
    returning one refusal string)."""
    benign = arm.get("benign_control") or {}
    false_refusal = benign.get("false_refusal_rate_on_benign_controls")
    if false_refusal is not None and false_refusal > MAX_FALSE_REFUSAL_RATE_FOR_A_WINNER:
        return f"{arm_name} refuses {false_refusal:.0%} of benign prompts"
    distinct = arm.get("distinct_output_ratio")
    if distinct is not None and distinct < MIN_DISTINCT_OUTPUT_RATIO_FOR_A_WINNER:
        return f"{arm_name} output has collapsed ({distinct:.0%} distinct)"
    return None


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

    leader = "steering" if sq > qq else ("qlora" if qq > sq else "tie")
    if leader == "tie":
        return "tie"
    leader_arm = steering if leader == "steering" else qlora
    reason = _degeneracy_reason(leader, leader_arm)
    if reason is not None:
        return f"inconclusive ({reason})"
    return leader


def render_comparison_markdown(comparison: Dict[str, Any], plot_paths: Optional[Dict[str, List[str]]] = None) -> str:
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
        lines.append(
            "`distinct%` is the fraction of held-out outputs that are unique text -- a collapsed value "
            "(e.g. 2%) means the arm returned nearly the same output regardless of input, which a quality "
            "score alone can hide. `winner` becomes `inconclusive` rather than naming a degenerate arm; "
            "see the Safety / degeneracy controls section below for why."
        )
        lines.append("")
        lines.append("| model | recipe | steering quality | steering distinct% | baseline quality | steering config | qlora quality | qlora distinct% | n_test | winner |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for entry in entries:
            steering = entry["steering"]
            qlora = entry["qlora"]
            sq, qq = steering.get("test_quality"), qlora.get("test_quality")
            n_test = min(steering.get("n_test", 0), qlora.get("n_test", 0))
            config = f"{steering.get('method')} L{steering.get('layer_idx')} c={steering.get('coefficient')} scope={steering.get('token_scope')}"
            lines.append(
                f"| {entry['model_id']} | {entry['recipe_id']} | {_fmt(sq)} | {_fmt_pct(steering.get('distinct_output_ratio'))} | "
                f"{_fmt(steering.get('baseline_quality'))} | "
                f"{config} | {_fmt(qq)} | {_fmt_pct(qlora.get('distinct_output_ratio'))} | {n_test} | {_winner(entry)} |"
            )
        lines.append("")

        any_benign = any(
            entry["steering"].get("benign_control") or entry["qlora"].get("benign_control") for entry in entries
        )
        if any_benign:
            lines.append("## Safety / degeneracy controls")
            lines.append("")
            lines.append(
                "`false refusal %` is the rate at which an arm refuses a BENIGN prompt from the recipe's "
                "matched control set (e.g. XSTest for a refusal recipe) -- the check a bare quality score "
                "cannot make: an arm that refuses every prompt scores well on \"did it refuse the harmful "
                "one\" while failing this number badly. `unsafe-contrast refusal %` is the same control "
                "set's genuinely-unsafe matched prompts, where refusing IS correct."
            )
            lines.append("")
            lines.append("| model | recipe | arm | false refusal % (benign) | n benign | unsafe-contrast refusal % | distinct% |")
            lines.append("|---|---|---|---|---|---|---|")
            for entry in entries:
                for arm_name, arm in (("steering", entry["steering"]), ("qlora", entry["qlora"])):
                    benign = arm.get("benign_control")
                    if benign is None:
                        continue
                    lines.append(
                        f"| {entry['model_id']} | {entry['recipe_id']} | {arm_name} | "
                        f"{_fmt_pct(benign.get('false_refusal_rate_on_benign_controls'))} | "
                        f"{_fmt(benign.get('n_benign_controls'), 0)} | "
                        f"{_fmt_pct(benign.get('unsafe_contrast_safe_refusal_rate'))} | "
                        f"{_fmt_pct(benign.get('distinct_output_ratio'))} |"
                    )
            lines.append("")

        any_rejected = any(entry["steering"].get("rejected_for_fluency") for entry in entries)
        if any_rejected:
            lines.append("## Steering configs rejected for fluency")
            lines.append("")
            lines.append(
                "These validation-split configs scored well on the target metric but were EXCLUDED from "
                "selection because their mean perplexity ratio or repetition score indicated degenerate "
                "output (see `comparison.DEFAULT_FLUENCY_CAP_RATIO`) -- listed so \"no usable positive "
                "coefficient\" or similar is a visible finding rather than a silent gap."
            )
            lines.append("")
            lines.append("| model | recipe | method | layer | coefficient | scope | validation quality (rejected) |")
            lines.append("|---|---|---|---|---|---|---|")
            for entry in entries:
                for rejected in entry["steering"].get("rejected_for_fluency") or []:
                    lines.append(
                        f"| {entry['model_id']} | {entry['recipe_id']} | {rejected.get('method')} | "
                        f"{rejected.get('layer_idx')} | {rejected.get('coefficient')} | {rejected.get('token_scope')} | "
                        f"{_fmt(rejected.get('validation_quality'))} |"
                    )
            lines.append("")

        if plot_paths:
            lines.append("## Visualizations")
            lines.append("")
            lines.append(
                "The data-efficiency curve shows at what labeled-example count (if any) one arm's "
                "quality overtakes the other's. The layer x coefficient heatmap shows how robust the "
                "validation-selected steering config is -- a lone bright cell surrounded by much worse "
                "neighbors is a fragile pick; a broad bright region is a robust one -- and how many "
                "configs in the grid actually beat QLoRA's quality, not just the single selected point."
            )
            lines.append("")
            for entry in entries:
                key = f"{entry['model_id']}__{entry['recipe_id']}"
                paths = plot_paths.get(key)
                if not paths:
                    continue
                lines.append(f"### {entry['model_id']} / {entry['recipe_id']}")
                lines.append("")
                for filename in paths:
                    lines.append(f"![{filename}]({filename})")
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
                f"{_fmt(steering.get('one_time_cost_s'), 2)} | {_fmt(steering.get('labeled_examples'), 0)} | "
                f"{_fmt(steering.get('artifact_bytes'), 0)} | {_fmt(steering.get('per_request_ms_per_token'), 2)} |"
            )
            lines.append(
                f"| {entry['model_id']} | {entry['recipe_id']} | qlora | "
                f"{_fmt(qlora.get('one_time_cost_s'), 2)} | {_fmt(qlora.get('labeled_examples'), 0)} | "
                f"{_fmt(qlora.get('artifact_bytes'), 0)} | {_fmt(qlora.get('per_request_ms_per_token'), 2)} |"
            )
        lines.append("")
        lines.append(
            "_One-time cost (2026-07-26: EXCLUDES generation for both arms, previously did not): "
            "steering's is its selected vector's OWN measured extraction time (not the whole sweep's "
            "wall time amortized evenly -- see `full_sweep_wall_time_s` in report.json for that whole-run "
            "exploration figure, which is NOT comparable to QLoRA's single training run); QLoRA's is TRAIN "
            "time only, no longer train+eval (eval generation time is inference cost, reported fairly via "
            "`ms/token` below, not one-time setup cost). Both arms' `ms/token` is the fair, directly "
            "comparable per-request inference cost, averaged over configs/adapters evaluated, not over "
            "held-out examples._"
        )
        lines.append("")

        any_cost_by_n = any(len(entry.get("cost_by_n") or []) > 1 for entry in entries)
        if any_cost_by_n:
            lines.append("## Cost by labeled-example count (N)")
            lines.append("")
            lines.append(
                "The table above pins to the MAX N each arm reached. `experiment.n_sweep` is a manifest "
                "tunable, so cost at each N it swept is broken out here separately -- \"cost to reach "
                "quality X with N labels\" is what this answers; see the Matched quality data-efficiency "
                "plots above for quality at the same N points."
            )
            lines.append("")
            lines.append("| model | recipe | N | steering cost (s) | steering configs | qlora cost (s) |")
            lines.append("|---|---|---|---|---|---|")
            for entry in entries:
                for point in entry.get("cost_by_n") or []:
                    lines.append(
                        f"| {entry['model_id']} | {entry['recipe_id']} | {point['n']} | "
                        f"{_fmt(point.get('steering_one_time_cost_s'), 2)} | {_fmt(point.get('steering_num_configs'), 0)} | "
                        f"{_fmt(point.get('qlora_one_time_cost_s'), 2)} |"
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

    # Imported here, not at module level, so this module (and everything
    # that transitively imports it, e.g. runner.compare) never requires
    # matplotlib just to join/format a report -- only actually rendering
    # plots does. A minimal install without the plotting extra still gets
    # a full report.md/report.json, just without embedded images.
    from .comparison_plots import render_comparison_plots
    plot_paths = render_comparison_plots(comparison, output)

    markdown_path = output / "report.md"
    markdown_path.write_text(render_comparison_markdown(comparison, plot_paths), encoding="utf-8")
    return markdown_path
