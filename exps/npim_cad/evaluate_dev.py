# exps/npim_cad/evaluate_dev.py
"""EXP-NPIM-00 Stage E authorization gate (no evaluator in NPIM-00).

The anomaly-development evaluator is deliberately NOT implemented in
EXP-NPIM-00.  Anomaly labels and masks are protected evidence: reading them
requires a separate, explicit authorization and a frozen score lock.  This
module exists only to enforce that gate and to make an accidental run fail
closed.

A future authorized evaluator must, before reading any mask:

1. name the formally admitted exposed-DEV cohort and its raw-image/checkpoint
   lineage;
2. freeze periods, lambda, rank, projection/seed, mixture, score type, image
   score, writer checkpoint, controls and the exact AP implementation;
3. generate and SHA-256-lock prediction arrays *without* labels;
4. only then open numerical masks, and never rerun/rescore a consumed cohort.

Required authorization schema (all fields mandatory)::

    {
      "experiment": "EXP-NPIM-00",
      "authorized": true,
      "authorized_by": "<human owner>",
      "authorized_utc": "YYYY-MM-DDTHH:MM:SSZ",
      "protocol_sha256": "<current exps/npim_cad/protocol.md sha256>",
      "cohort_manifest_path": "<path>",
      "cohort_manifest_sha256": "<sha256>",
      "score_lock_sha256": "<sha256 of the label-free prediction lock>",
      "purpose": "<one sentence>"
    }

Nothing is written and no image is opened by this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hope_cad.npim_context import sha256_file  # noqa: E402

PROTOCOL = ROOT / "exps/npim_cad/protocol.md"
REQUIRED = (
    "experiment",
    "authorized",
    "authorized_by",
    "authorized_utc",
    "protocol_sha256",
    "cohort_manifest_path",
    "cohort_manifest_sha256",
    "score_lock_sha256",
    "purpose",
)


def refuse(reason: str) -> Dict[str, Any]:
    return {
        "decision": "BLOCKED_STAGE_E_NOT_AUTHORIZED",
        "reason": reason,
        "opened_anomaly_data": False,
        "anomaly_labels_used": False,
        "note": "NPIM-00 implements no anomaly evaluator; Stage E is NOT_RUN.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-NPIM-00 Stage E gate (refuses by design)")
    parser.add_argument("--authorization", help="path to a signed authorization JSON")
    parser.add_argument("--score-lock", help="path to the label-free prediction lock")
    args = parser.parse_args()

    if not args.authorization:
        print(json.dumps(refuse("no --authorization manifest supplied"), indent=2))
        return 2
    authorization_path = Path(args.authorization)
    if not authorization_path.is_file():
        print(json.dumps(refuse(f"authorization manifest not found: {authorization_path}"), indent=2))
        return 2
    try:
        authorization = json.loads(authorization_path.read_text())
    except Exception as exc:
        print(json.dumps(refuse(f"authorization manifest is not valid JSON: {exc}"), indent=2))
        return 2
    missing = [field for field in REQUIRED if field not in authorization]
    if missing:
        print(json.dumps(refuse(f"authorization manifest missing fields: {missing}"), indent=2))
        return 2
    problems = []
    if authorization["experiment"] != "EXP-NPIM-00":
        problems.append("experiment mismatch")
    if authorization["authorized"] is not True:
        problems.append("authorized is not true")
    if not isinstance(authorization["authorized_by"], str) or not authorization["authorized_by"].strip():
        problems.append("empty authorized_by")
    if authorization["protocol_sha256"] != sha256_file(PROTOCOL):
        problems.append("protocol SHA-256 does not match the frozen protocol")
    cohort = Path(authorization["cohort_manifest_path"])
    if not cohort.is_file() or sha256_file(cohort) != authorization["cohort_manifest_sha256"]:
        problems.append("cohort manifest missing or SHA-256 mismatch")
    if not args.score_lock:
        problems.append("no --score-lock supplied")
    else:
        lock = Path(args.score_lock)
        if not lock.is_file() or sha256_file(lock) != authorization["score_lock_sha256"]:
            problems.append("score lock missing or SHA-256 mismatch")
    if problems:
        print(json.dumps(refuse("; ".join(problems)), indent=2))
        return 2

    print(json.dumps({
        "decision": "BLOCKED_STAGE_E_NOT_IMPLEMENTED",
        "reason": "authorization checked, but EXP-NPIM-00 intentionally contains no anomaly evaluator; "
                  "a separately reviewed Stage E implementation is required",
        "opened_anomaly_data": False,
        "anomaly_labels_used": False,
        "note": "This gate never opens labels or masks. No rerun or rescore was performed.",
    }, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
