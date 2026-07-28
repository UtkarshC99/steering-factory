"""One-off: rescore an existing run's stored generations.jsonl with the
current (widened) refusal_score_substring and rebuild the comparison
report from the rescored rows -- no GPU, no re-generation. Verifies
whether the 2026-07-26 refusal-marker fix changes the headline result,
using the exact stored `output` text so this is not a re-simulation.

Usage: python scripts/rescore_dpo_run.py <run_root>
  where <run_root> is e.g. experiment-outputs/preset_4bit_baseline_260726_0942_dpo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steering_factory.evaluators import refusal_score_substring
from steering_factory import comparison


def _rescore_file(path: Path) -> None:
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if "safe_refusal" not in row or row.get("output") is None:
            continue
        old = row["safe_refusal"]
        new = refusal_score_substring(row["output"])
        row.update(new)
        if old != new["safe_refusal"]:
            changed += 1
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"  {path}: {len(rows)} rows, {changed} safe_refusal flips")


def main() -> None:
    run_root = Path(sys.argv[1])
    steer_dirs = sorted((run_root / "steering").iterdir())
    qlora_dirs = sorted((run_root / "qlora").iterdir())
    assert len(steer_dirs) == 1 and len(qlora_dirs) == 1, "expected exactly one steering/ and one qlora/ run dir"
    steer_dir, qlora_dir = steer_dirs[0], qlora_dirs[0]

    print("Rescoring steering generations...")
    _rescore_file(steer_dir / "results" / "generations.jsonl")
    print("Rescoring QLoRA generations...")
    _rescore_file(qlora_dir / "results" / "generations.jsonl")

    print("\nRebuilding comparison report...")
    report = comparison.build_comparison(str(steer_dir), str(qlora_dir))
    out_dir = run_root / "comparisons_rescored"

    from steering_factory.comparison_report import write_comparison_report
    markdown_path = write_comparison_report(report, str(out_dir))
    print(f"\nRescored report written to {markdown_path}")


if __name__ == "__main__":
    main()
