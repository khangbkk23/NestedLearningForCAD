"""Record the ORIGINAL full-arm NL-SGM-01A protocol as SGM01A_BLOCKED / PREFLIGHT_BLOCKED (owner decision 2026-11-06).

The CPU preflight (bench.py) was stopped by the operator during BM2b after measured LOWER BOUNDS already exceeded the
per-arm cap, so BENCH.json / RESOURCE_PROJECTION.json were never written. This script transcribes the measured
evidence verbatim from the preserved logs (parsed, not retyped) and writes, create-exclusive:
  RESOURCE_LEDGER.json, MATCHED_CONTROL_RESULTS.csv, PER_CATEGORY_TABLE.csv, FINAL_VERDICT.json
No outcome (distance, score, metric) is computed. No LOCK.json existed (the gate never passed).
"""
from __future__ import annotations

import ast
import csv
import io
import re

from exps.nl_sgm_01a import common as C

H = 3600.0
CAPS_ORIGINAL = {"total_wall_h": 16, "per_arm_h": 3, "rss_gb": 8, "disk_gb": 5, "device": "cpu"}
CAPS_OWNER_REVISION = {"total_wall_h": 16, "per_arm_h": 3, "rss_gb": 12, "disk_gb": 5, "device": "cuda allowed, TF32 off"}


def parse_log(path):
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] (BM\w+) (.*)$", line)
        if not m:
            continue
        tag, rest = m.group(2), m.group(3)
        key, _, payload = rest.partition(" {")
        if not payload:
            continue
        name = f"{tag} {key}".strip() if key and not key.startswith("{") else tag
        out[name] = ast.literal_eval("{" + payload)
    return out


def main():
    out = C.OUT
    log = out / "logs" / "bench_cpu_preflight.log"
    cuda = out / "logs" / "cuda_microbench_2026-11-06.txt"
    ev = parse_log(log)
    reasons = []
    for name, v in ev.items():
        if not name.startswith(("BM2 ", "BM2b ")):
            continue
        lb = v["lower_bound_s"] / H
        if lb > 3:
            reasons.append(f"{name}: measured LOWER BOUND {lb:.1f} h > per-arm cap 3 h")
        elif v["projected_s"] / H > 3:
            reasons.append(f"{name}: projection {v['projected_s'] / H:.1f} h (x1.3) > per-arm cap 3 h")
    reasons.append("CUDA microbench (owner allowed CUDA, RAM 12 GB): canonical closest-pair rule at 75,121 rows still "
                   "22-34 s/image (CPU-side bookkeeping ~40 ms/replacement); A9 ~2.5-4.0 s/image -> ~3.2 h projected")
    reasons.append("full A0-A10/B1-B8 matrix cannot fit 16 h total / 3 h per arm on CPU or CUDA")

    ledger = {"task": "NL-SGM-01A (original full-arm protocol)", "stage": "P1-C preflight, stopped",
              "caps_original": CAPS_ORIGINAL, "caps_owner_revision_2026_11_06": CAPS_OWNER_REVISION,
              "evidence_logs": {str(log.relative_to(C.ROOT)): C.sha(log), str(cuda.relative_to(C.ROOT)): C.sha(cuda)},
              "measured": ev,
              "bench_completion": "stopped by operator during BM2b (X=A3) after lower bounds exceeded caps; "
                                  "BM2b for A4/A5/A6 not measured; BENCH.json and RESOURCE_PROJECTION.json not written",
              "lock": "none (LOCK.json is written only after a passed gate; the gate never passed)",
              "outcome_compute_performed": False}
    C.write_json(out / "RESOURCE_LEDGER.json", ledger)

    arms = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
            "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["arm", "status", "note"])
    for a in arms:
        w.writerow([a, "NOT_RUN_PREFLIGHT_BLOCKED", "no outcome computed; see RESOURCE_LEDGER.json"])
    with (out / "MATCHED_CONTROL_RESULTS.csv").open("x") as f:
        f.write(buf.getvalue())

    co = C.cohorts()
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["a(c)", "category", "group", "writer_images", "writer_patches", "sel_probes", "eval_probes", "status"])
    for k, c in enumerate(C.ORDER):
        v = co[c]
        w.writerow([k, c, "old" if k in C.OLD else "late", len(v["writers"]), 784 * len(v["writers"]),
                    len(v["sel"]), len(v["eval"]), "NOT_RUN_PREFLIGHT_BLOCKED"])
    with (out / "PER_CATEGORY_TABLE.csv").open("x") as f:
        f.write(buf.getvalue())

    verdict = {"task": "NL-SGM-01A", "protocol": "original full-arm (plan §1-§12)", "phase": "P1", "utc": C.now(),
               "verdict": "SGM01A_BLOCKED", "subreason": "PREFLIGHT_BLOCKED", "gate_reasons": reasons,
               "gate_inputs": {"tests": "24/24 passed (tests/pytest_v3.txt)",
                               "exactness": "certified band argument + brute-force tests; BM1 screen error <= 1.1% of bound"},
               "successor": "NL-SGM-01A-AM02 minimal historical-support retention (separate, versioned amendment; "
                            "agents/plans/nl_sgm_01a_am02_plan.md; results/hope_cad/nl_sgm_01a_am02/)",
               "not_performed": ["LOCK.json", "A0/A1 oracle", "Stage A/B arms", "X selection", "EVAL", "seal"],
               "scientific_status": "No outcome was computed; no headroom, retention or NL claim of any kind."}
    C.write_json(out / "FINAL_VERDICT.json", verdict)
    print("SGM01A_BLOCKED", *reasons, sep="\n")


if __name__ == "__main__":
    main()
