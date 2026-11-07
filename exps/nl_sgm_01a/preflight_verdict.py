"""Apply the NL-SGM-01A execution gate to the measured preflight and write the gate artifacts.

Reads BENCH.json / RESOURCE_PROJECTION.json (timings only). Writes, create-exclusive:
  RESOURCE_LEDGER.json, MATCHED_CONTROL_RESULTS.csv, PER_CATEGORY_TABLE.csv, FINAL_VERDICT.json
No outcome (distance, score, metric) is computed here.
"""
from __future__ import annotations

import csv
import io

from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a.bench import CAPS

H = 3600.0


def main():
    out = C.OUT
    bench = C.read_json(out / "BENCH.json")
    proj = C.read_json(out / "RESOURCE_PROJECTION.json")
    a = bench["BM2_stage_a_rules"]
    reasons = []
    for name, sm in a.items():
        lb, pr = sm["lower_bound_s"] / H, sm["projected_s"] / H * (3 if name.startswith("A6") else 1)
        if lb > CAPS["per_arm_h"]:
            reasons.append(f"{name}: measured LOWER BOUND {lb:.2f} h > per-arm cap {CAPS['per_arm_h']} h")
        elif pr > CAPS["per_arm_h"]:
            reasons.append(f"{name}: projection {pr:.2f} h (x{proj['safety_factor']}) > per-arm cap")
    for name, v in proj["stage_b_by_candidate_X"].items():
        if v["max_single_arm_s"] / H > CAPS["per_arm_h"]:
            reasons.append(f"Stage B with {name}: largest arm {v['max_single_arm_s'] / H:.1f} h > per-arm cap "
                           f"(B5 lower bound {v['B5_lower_bound_s'] / H:.1f} h)")
    g = proj["gate"]
    if g["total_wall_h_worst_X"] > CAPS["total_wall_h"]:
        reasons.append(f"total projected {g['total_wall_h_worst_X']:.1f} h (worst X = {g['worst_X']}) > {CAPS['total_wall_h']} h")
    if bench["peak_rss_gb"] > CAPS["rss_gb"]:
        reasons.append(f"benchmark process peak RSS {bench['peak_rss_gb']:.2f} GB > {CAPS['rss_gb']} GB "
                       "(includes file-backed mmap pages and several surrogate tasks held at once; the run's peak "
                       "is therefore not demonstrated within the cap)")
    blocked = bool(reasons)

    ledger = {
        "task": "NL-SGM-01A", "stage": "P1-C preflight (timings only)", "device": "cpu", "cuda_visible": "",
        "bench_wall_s": bench["wall_s"], "bench_peak_rss_gb": bench["peak_rss_gb"], "threads": bench["threads"],
        "cpu": bench["cpu"], "caps": CAPS, "projection": proj,
        "persistent_bytes_by_arm_formula": {
            "A2/A3/A5/A6 FP32 2,500 rows": 2500 * 3072, "A4/A7 FP32 2,499 rows + owner": 2499 * 3073,
            "A8 FP32 10,000 rows": 10000 * 3072, "A9 CALM-L1 9,684 rows (calm01 ledger)": None,
            "B4/B8 UINT8 9,884 rows + owner": 9884 * (K.U8_ROW_BYTES + 1),
            "B5/B6 PQ 75,121 rows + owner + 1 FP16 codebook": 75121 * 97 + K.CODEBOOK_BYTES,
            "B7 PQ at k=14: 18,368 rows + owner + 15 FP16 codebooks": 18368 * 97 + 15 * K.CODEBOOK_BYTES,
        },
        "outcome_compute_performed": False,
    }
    C.write_json(out / "RESOURCE_LEDGER.json", ledger)

    rows = [
        ("A0", "full-writer exact oracle (offline)", "uncharged", "oracle_s"),
        ("A1", "category oracle (offline, H4 normal side)", "uncharged", "oracle_s"),
        ("A2", "CADIC closest-pair FP32", "B1 / 2,500", "A2_cadic_2500"), ("A3", "P0 stream greedy FP32", "B1 / 2,500", "A3_greedy_2500"),
        ("A4", "equal-quota CADIC FP32", "B1 / 2,499", "A4_quota_2499"), ("A5", "AD-02 S4 partition", "B1 / 2,500", "A5_S4_2500"),
        ("A6", "AD-02 R4 partition x3 draws", "B1 / 2,500", "A6_R4_2500"), ("A7", "boundary FPS", "B1 / 2,499", "A7_fps_2499"),
        ("A8", "CADIC FP32 at B2", "B2 / 10,000", "A8_cadic_10000"), ("A9", "CALM-L1 UINT8", "B1 / 9,684", "A9_calm_l1_9684"),
        ("A10", "X under 49-bin scorer (report-only)", "B1", None),
        ("B1", "X FP32", "B1", None), ("B2", "same IDs UINT8", "<B1", None), ("B3", "same IDs PQ fixed C0", "<B1", None),
        ("B4", "X UINT8 matched bytes", "B1 / 9,884", "B4"), ("B5", "X PQ fixed C0", "B1 / 75,121", "B5"),
        ("B6", "X PQ refit + transport", "B1 / 75,121", "B6"), ("B7", "X PQ append-only codebooks", "B1 / 75,121->18,368", "B7"),
        ("B8", "hash-random UINT8", "B1 / 9,884", "B8"), ("DIAG", "same-ID codebook policy (X at 18,368 FP32)", "diagnostic", "X18k"),
        ("R1", "B5/B6 capped at 18,368 vs B7", "diagnostic", "R1_B5"), ("R2", "B5 at 9,884 vs B4", "diagnostic", "R2_B5"),
    ]
    worst = proj["stage_b_by_candidate_X"][g["worst_X"]]["B_rules_s"]
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["arm", "definition", "budget_rows", "status", "projected_update_h", "note"])
    for arm, d, b, key in rows:
        if key == "oracle_s":
            h = proj["oracle_s"] / H
        elif key in proj["stage_a_update_s_by_arm"]:
            h = proj["stage_a_update_s_by_arm"][key] / H
        elif key in worst:
            h = worst[key] / H
        else:
            h = ""
        w.writerow([arm, d, b, "NOT_RUN_PREFLIGHT_BLOCKED" if blocked else "NOT_RUN", f"{h:.2f}" if h != "" else "",
                    "no outcome computed; Stage-B projection for worst candidate X" if arm.startswith(("B", "R", "D")) else "no outcome computed"])
    p = out / "MATCHED_CONTROL_RESULTS.csv"
    with p.open("x") as f:
        f.write(buf.getvalue())

    co = C.cohorts()
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["a(c)", "category", "group", "writer_images", "writer_patches", "sel_probes", "eval_probes", "status"])
    for k, c in enumerate(C.ORDER):
        v = co[c]
        w.writerow([k, c, "old" if k in C.OLD else "late", len(v["writers"]), 784 * len(v["writers"]),
                    len(v["sel"]), len(v["eval"]), "NOT_RUN_PREFLIGHT_BLOCKED" if blocked else "NOT_RUN"])
    with (out / "PER_CATEGORY_TABLE.csv").open("x") as f:
        f.write(buf.getvalue())

    verdict = {
        "task": "NL-SGM-01A", "phase": "P1", "utc": C.now(),
        "verdict": "SGM01A_BLOCKED" if blocked else "PREFLIGHT_PASSED",
        "subreason": "PREFLIGHT_BLOCKED" if blocked else None,
        "gate_reasons": reasons,
        "gate_inputs": {"tests": "24/24 passed (tests/pytest_v3.txt)", "exactness": "certified band argument + brute-force tests; BM1 screen error <= 1.1% of bound",
                        "caps": CAPS, "projection": "RESOURCE_PROJECTION.json"},
        "not_performed": ["LOCK.json (only after a passed gate)", "A0/A1 oracle", "Stage A/B arms", "X selection",
                          "EVAL evaluation", "seal"],
        "scientific_status": "No headroom, consolidation or NL claim of any kind: no outcome was computed.",
        "not_claimed": ["anomaly-detection improvement", "Pixel AP retention", "Nested Learning efficacy",
                        "that conventional PQ is or is not sufficient"],
    }
    C.write_json(out / "FINAL_VERDICT.json", verdict)
    print(verdict["verdict"], *reasons, sep="\n")


if __name__ == "__main__":
    main()
