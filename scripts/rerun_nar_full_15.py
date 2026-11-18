"""Launch the unchanged NAR full-15 runner in a fresh output directory.

This helper changes only the output destination. All original model, data,
source-lock, owner-authorization and resource checks remain in effect.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("lock", "smoke", "train", "score", "metrics", "cost"))
    parser.add_argument("--out", required=True, type=Path,
                        help="new directory named nar_full_15_rerun_<name> under results/hope_cad")
    parser.add_argument("--owner-authorized", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out if args.out.is_absolute() else root / args.out
    output = output.resolve()
    allowed_parent = (root / "results/hope_cad").resolve()
    if output.parent != allowed_parent or not output.name.startswith("nar_full_15_rerun_"):
        parser.error("--out must be results/hope_cad/nar_full_15_rerun_<name>; preserve historical results")
    if args.phase == "lock" and output.exists():
        parser.error("lock requires a new output directory; preserve existing outputs")
    if args.phase != "lock" and not (output / "SOURCE_LOCK.json").is_file():
        parser.error("run lock first for this output directory")
    if args.phase in ("smoke", "train", "score", "cost") and not args.owner_authorized:
        parser.error("this GPU phase requires --owner-authorized")
    sys.path.insert(0, str(root))
    from exps.nar_pilot_00 import full

    full.OUT = output
    if args.phase == "cost":
        from exps.nar_pilot_00 import cost
        cost.main()
    else:
        # Use the original CLI so its phase-specific safeguards remain active.
        sys.argv = [sys.argv[0], args.phase]
        if args.owner_authorized:
            sys.argv.append("--owner-authorized")
        full.main()


if __name__ == "__main__":
    main()
