from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import load_manifest
from .runner import compare, prepare_data, run_qlora, run_steering


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steering-factory", description="Reproducible activation-steering experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "run", "extract", "evaluate", "finetune", "report"):
        command = sub.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    comparison = sub.add_parser("compare")
    comparison.add_argument("runs", nargs="+")
    comparison.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare":
        print(compare(args.runs, args.output_root))
        return 0
    manifest = load_manifest(args.manifest, args.overrides)
    if args.command == "prepare-data":
        store = prepare_data(manifest, " ".join(sys.argv))
    elif args.command == "finetune":
        store = run_qlora(manifest, " ".join(sys.argv))
    else:
        store = run_steering(manifest, " ".join(sys.argv))
    print(json.dumps({"run_id": store.run_id, "artifact_dir": str(store.path), "status": store.artifact.status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
