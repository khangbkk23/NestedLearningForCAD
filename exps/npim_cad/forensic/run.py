# exps/npim_cad/forensic/run.py
"""CLI for EXP-NPIM-02 forensic diagnostics.

Subcommands:
  reconstruct  one frozen re-encode of the 60 admitted DEV images + parity
  analyze      per-level/cross-term/defect/capacity/spatial analyses
  all          both, in order

All subcommands are default-deny and require --allow-consumed-dev.  No training,
no parameter update, no mask enters a predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.npim_cad.forensic.analysis import analyze  # noqa: E402
from exps.npim_cad.forensic.reconstruct import finalize, reconstruct  # noqa: E402


def require_authorization(args: argparse.Namespace) -> bool:
    if not args.allow_consumed_dev:
        print(json.dumps({
            "decision": "BLOCKED_CONSUMED_DEV_NOT_AUTHORIZED",
            "reason": "explicit --allow-consumed-dev is required; no DEV image or mask was opened",
            "opened_anomaly_data": False,
        }, indent=2))
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-NPIM-02 forensic diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reconstruct", "finalize", "analyze", "all"):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--allow-consumed-dev", action="store_true")
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not require_authorization(args):
        return 2
    if args.command == "reconstruct":
        result = reconstruct(args.run_id, args.device)
        print(json.dumps({
            "decision": "RECONSTRUCTION_SEALED",
            "run_id": args.run_id,
            "parity_all_pass": result["parity"]["all_pass"],
            "max_abs": result["parity"]["max_abs_global"],
            "max_relative_l2": result["parity"]["max_relative_l2_global"],
            "run_dir": result["run_dir"],
            "resource": result["resource"],
        }, indent=2))
        return 0
    if args.command == "finalize":
        result = finalize(args.run_id)
        print(json.dumps({
            "decision": "RECONSTRUCTION_FINALIZED",
            "run_id": args.run_id,
            "parity_all_pass": result["parity"]["all_pass"],
            "run_dir": result["run_dir"],
        }, indent=2))
        return 0
    if args.command == "analyze":
        summary = analyze(args.run_id)
        print(json.dumps({
            "decision": "ANALYSIS_COMPLETE",
            "run_id": args.run_id,
            "levels": summary["levels"],
            "cross_groups": summary["cross_groups"],
            "reproduction_all": summary["reproduction"]["all_reproduced"],
            "spatial_geometric_ok": summary["spatial"]["geometric_all_ok"],
            "spatial_map_parity_ok": summary["spatial"]["qualitative"]["sealed_map_parity_ok"],
            "resource": summary["resource"],
        }, indent=2))
        return 0
    if args.command == "all":
        result = reconstruct(args.run_id, args.device)
        summary = analyze(args.run_id)
        print(json.dumps({
            "decision": "FORENSIC_ANALYSIS_COMPLETE",
            "run_id": args.run_id,
            "parity_all_pass": result["parity"]["all_pass"],
            "levels": summary["levels"],
            "reproduction_all": summary["reproduction"]["all_reproduced"],
            "run_dir": result["run_dir"],
        }, indent=2))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
