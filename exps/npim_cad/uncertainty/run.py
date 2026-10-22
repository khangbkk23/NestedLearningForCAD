# exps/npim_cad/uncertainty/run.py
"""CLI for EXP-NPIM-03 G2/G3.  G4 anomaly evaluation is intentionally absent."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.npim_cad.uncertainty.calibrate import calibrate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-NPIM-03 normal-only calibration (G2/G3)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("calibrate", help="fit/evaluate the frozen normal-only readouts")
    run.add_argument("--run-id", required=True)
    run.add_argument("--allow-normal-only", action="store_true")
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.command != "calibrate":
        parser.error("unknown command")
    if not args.allow_normal_only:
        print("refusing to read normal calibration images without --allow-normal-only", file=sys.stderr)
        return 2
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; refusing to silently fall back", file=sys.stderr)
        return 2
    try:
        result = calibrate(args.run_id, args.device)
    except Exception as exc:  # preserve a machine-readable failure without retrying
        failure = {
            "schema": "npim03_failure_v1",
            "decision": "NPIM03_INCOMPLETE",
            "run_id": args.run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "note": "no anomaly data was accessed; do not retry with altered constants",
        }
        run_dir = ROOT / "results/hope_cad/npim_03" / args.run_id
        if run_dir.exists() and not (run_dir / "E_FAILED.json").exists():
            (run_dir / "E_FAILED.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
