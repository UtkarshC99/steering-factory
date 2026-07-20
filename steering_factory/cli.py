from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import load_manifest
from .runner import compare, prepare_data, render_report, run_evaluate, run_extract, run_qlora, run_steering


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steering-factory", description="Reproducible activation-steering experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "run", "extract", "finetune"):
        command = sub.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    evaluate = sub.add_parser("evaluate", help="Evaluate vectors saved by a prior `extract` (or `run`) invocation.")
    evaluate.add_argument("manifest")
    evaluate.add_argument("--vectors-run", required=True, help="Artifact directory of a prior extract/run to read vectors/index.jsonl from.")
    evaluate.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    report = sub.add_parser("report", help="Render a Markdown summary from an existing run directory. Loads no model.")
    report.add_argument("run_dir")
    report.add_argument("--output", default=None, help="Output path for the rendered report (default: <run_dir>/report.md).")
    comparison = sub.add_parser("compare")
    comparison.add_argument("runs", nargs="+")
    comparison.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare":
        print(compare(args.runs, args.output_root))
        return 0
    if args.command == "report":
        print(render_report(args.run_dir, args.output))
        return 0
    manifest = load_manifest(args.manifest, args.overrides)
    if args.command == "prepare-data":
        store = prepare_data(manifest, " ".join(sys.argv))
    elif args.command == "finetune":
        store = run_qlora(manifest, " ".join(sys.argv))
    elif args.command == "extract":
        store = run_extract(manifest, " ".join(sys.argv))
    elif args.command == "evaluate":
        store = run_evaluate(manifest, args.vectors_run, " ".join(sys.argv))
    else:
        store = run_steering(manifest, " ".join(sys.argv))
    print(json.dumps({"run_id": store.run_id, "artifact_dir": str(store.path), "status": store.artifact.status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
