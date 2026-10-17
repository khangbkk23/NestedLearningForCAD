"""P2d paths, arms, and pinned read-only reuse of P2/P1 machinery.

P2d adds no new data access: NormalReader, Replay, evaluation markers, stream
construction, protocol verification and hashing are imported unchanged from the
locked P2 module. On top of that, `p2_reference_pins` proves that every P2
reference artifact P2d consumes (arm-B/D predictions, point metrics, bootstrap
samples, per-run traces) still matches the SHA-256 recorded by the P2 phase.
"""
from __future__ import annotations

import json

from exps.task_p.p1_data import CP_SHA, sha, write_json  # noqa: F401  (re-export)
from exps.task_p.p2 import data as p2data
from exps.task_p.p2.data import (FEATURES, NormalReader, P1, P1_SHA, Replay,  # noqa: F401
                                 digest_tensor, evaluation_features, p1_protocol,
                                 require, stream, tensor_bytes, tree_digest,
                                 verify_inheritance)

ROOT = p2data.ROOT
P2 = ROOT / "results/hope_cad/task_p/p2"
OUT = ROOT / "results/hope_cad/task_p/p2d"
P2_PROTOCOL_SHA = "be3a0fa0a351f4acf0e087667090d25b3257e5b5cfccc5dac8b0daa3b1ea0fa2"
ARMS = ("C2", "D2")
SEEDS = p2data.SEEDS
ORDERS = p2data.ORDERS


def p2_reference_pins():
    """Fail closed if any reused P2 artifact or locked P2 source changed."""
    require(sha(P2 / "protocol_resolved.json") == P2_PROTOCOL_SHA, "P2 protocol changed")
    protocol = json.loads((P2 / "protocol_resolved.json").read_text())
    for name, expected in protocol["code_sha256"].items():
        require(sha(ROOT / name) == expected, f"P2 locked source changed: {name}")
    integrity = json.loads((P2 / "artifact_integrity.json").read_text())
    files = integrity["files"]
    references = ["point_metrics.json", "bootstrap_samples.json", "score_separation.json", "summary.json",
                  "training_diagnostics_summary.json"]
    for order in ORDERS:
        for seed in SEEDS:
            references.append(f"{order}_seed{seed}/predictions.json")
            references.append(f"{order}_seed{seed}/trace.json")
            references.append(f"{order}_seed{seed}/budgets.json")
            references.append(f"{order}_seed{seed}/state_t100.pt")
    checks = {}
    for name in references:
        record = files.get(name)
        require(record is not None, f"P2 artifact missing from integrity record: {name}")
        actual = sha(P2 / name)
        require(actual == record["sha256"], f"P2 artifact changed: {name}")
        checks[name] = actual
    return {"passed": True, "p2_protocol_sha256": P2_PROTOCOL_SHA, "checks": checks}
