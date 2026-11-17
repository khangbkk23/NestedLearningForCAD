"""NL-SGM-01A-C2-DOM sweep — locked analysis: metrics, matched comparisons, Pareto, recommendation, verdict.

    python -m exps.nl_sgm_01a_c2dom.analyze

Every rule below is part of the LOCK (RULES is hashed into PROTOCOL.json). Nothing here is tuned on results.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a_c2dom import common as D
from exps.task_p.p0_normal_support import require

ORDER = D.ORDER
HIST = {"C1_FPS_FP32": 7_679_427, "C2_FPS_PQ": 7_679_953}       # AM02/AM03 ledgers (ids not charged)
BOOT = {"B": 2000, "seed": 20261108, "unit": "category (paired, resample categories with replacement)"}

RULES = {
    "metrics": {
        "acquisition": "cell at boundary a(c) (c's own boundary), macro = mean over 15 categories",
        "final": "cell at boundary 15, macro = mean over 15 categories",
        "FM_Eq8": "per category with a(c) <= 14 (1-based): max_{t in [a(c), 14]} m_t - m_15; macro = mean over the 14 "
                  "categories (identical to AM03 code)",
        "pixel": "native Pixel AP (exact_ap)", "image": "Image AUROC (max-of-map image score)",
    },
    "bootstrap": BOOT,
    "comparisons": {
        "fixed_budget": "DOM_PQ_Bx - SEL_FP32_Bx (byte-matched, same ledger, same SEL pool)",
        "original_baseline": "DOM_PQ_Bx - C1_FPS_FP32 and SEL_FP32_Bx - C1_FPS_FP32 (NOT byte-matched for x > 1; C1 "
                             "ledger does not charge ids; C1 selects from the full writer pool)",
        "capacity": "DOM_PQ_B2 - DOM_PQ_B1, DOM_PQ_B4 - DOM_PQ_B2, DOM_PQ_B4 - DOM_PQ_B1 (and the same for SEL_FP32)",
        "domain_at_B1": "DOM_PQ_B1 - C2_FPS_PQ (bottle-only codebook; C2 ledger does not charge ids)",
        "cross_budget": "DOM_PQ_B1 - SEL_FP32_B2, DOM_PQ_B1 - SEL_FP32_B4 (PQ at 1/2 and 1/4 of the FP32 bytes)",
        "partition_cost": "SEL_FP32_B1 - C1_FPS_FP32 (half candidate pool vs full pool, same FP32 rule)",
        "descriptive_only": "DOM_PQ acquisition vs F2 acquisition (different identities/pool; no inference)",
    },
    "recommendation": {
        "candidates": "all registered configurations with complete evaluation (DOM_PQ_Bx, SEL_FP32_Bx, C1_FPS_FP32, "
                      "C2_FPS_PQ); bytes = final persistent bytes",
        "best_accuracy": "argmax final macro Pixel AP (ties -> fewer bytes)",
        "best_retention": "argmin macro Pixel FM among candidates with acquisition Pixel AP >= max_acq - 0.02 and final "
                          "Pixel AP >= max_final - 0.02 (reduced FM never counts when accuracy is substantially lower)",
        "near_best": "final Pixel AP >= best_final - 0.010 AND final Image AUROC >= max_final_auroc - 0.010 AND Pixel FM "
                     "<= FM(best_accuracy) + 0.010",
        "smallest_near_best": "argmin bytes within near_best (ties -> higher final Pixel AP)",
        "pareto_recommendation": "smallest_near_best",
        "pareto_frontier": "non-dominated on (bytes min, final Pixel AP max, Pixel FM min); also 2-D (bytes, final AP)",
        "status": "exploratory: MVTec test split historically exposed; not independently confirmed",
    },
    "verdict": {
        "V1_domain_resolved_in_stream": "DOM_PQ_B1 - C2_FPS_PQ acquisition Pixel AP CI lower > 0 AND DOM_PQ_B1 "
                                        "acquisition Pixel AP >= C1 acquisition - 0.02",
        "V2_fixed_budget_x": "final Pixel AP (DOM_PQ_Bx - SEL_FP32_Bx): ADVANTAGE if CI lower > 0 and final Image AUROC "
                             "difference point >= -0.01; INFERIOR if CI upper < 0; else PARITY",
        "V3_capacity": "DOM_PQ_B4 - DOM_PQ_B1 final Pixel AP: USEFUL if CI lower > 0 and point >= 0.01; SATURATED_BY_B2 "
                       "if additionally DOM_PQ_B4 - DOM_PQ_B2 point < 0.005; NOT_USEFUL otherwise",
        "V4_credible_detector": "some registered DOM_PQ_Bx has acquisition Pixel AP >= C1 - 0.02, final Pixel AP >= C1 "
                                "- 0.01 and final Image AUROC >= C1 - 0.02 (C1 = original B1 FP32 baseline)",
        "overall": "PQ_INFERIOR_STOP_COMPRESSION if V2 = INFERIOR at every registered budget; else PQ_ADVANTAGE if V2 "
                   "= ADVANTAGE at >= 1 registered budget (report the smallest); else PQ_PARITY_NO_ADVANTAGE",
        "limitation_attribution": "descriptive: acquisition gap to C1, FM, final gap, codec (DOM_PQ vs F2/C1 "
                                  "acquisition), update cost (FPS seconds, codebook fit seconds)",
    },
}


def boot(a, b, seed=BOOT["seed"], B=BOOT["B"]):
    return A3.bootstrap_delta(a, b, seed=seed, B=B)


def load_cells(arm):
    if arm in HIST:
        pc = json.loads((A3.OUT / "METRICS.json").read_text())["per_cell"]
        return {(t, c): pc[f"{arm}|{t}|{c}"] for t in range(15) for c in ORDER[:t + 1]}
    return {(t, c): D.read_json(D.OUT / "eval" / arm / f"stage{t + 1:02d}_{c}.json")
            for t in range(15) for c in ORDER[:t + 1]}


def summarize(cells):
    acq = {c: cells[(i, c)] for i, c in enumerate(ORDER)}
    fin = {c: cells[(14, c)] for c in ORDER}
    fm = {}
    for i, c in enumerate(ORDER[:14]):
        fm[c] = {m: max(cells[(t, c)][m] for t in range(i, 14)) - fin[c][m] for m in ("pixel_ap", "image_auroc")}
    out = {"per_category": {c: {"acq_pixel_ap": acq[c]["pixel_ap"], "final_pixel_ap": fin[c]["pixel_ap"],
                                "acq_image_auroc": acq[c]["image_auroc"], "final_image_auroc": fin[c]["image_auroc"],
                                "pixel_fm": fm.get(c, {}).get("pixel_ap"), "image_fm": fm.get(c, {}).get("image_auroc"),
                                "pixel_change_acq_to_final": fin[c]["pixel_ap"] - acq[c]["pixel_ap"],
                                "image_change_acq_to_final": fin[c]["image_auroc"] - acq[c]["image_auroc"]}
                            for c in ORDER}}
    out["acq_pixel_ap"] = float(np.mean([acq[c]["pixel_ap"] for c in ORDER]))
    out["acq_image_auroc"] = float(np.mean([acq[c]["image_auroc"] for c in ORDER]))
    out["final_pixel_ap"] = float(np.mean([fin[c]["pixel_ap"] for c in ORDER]))
    out["final_image_auroc"] = float(np.mean([fin[c]["image_auroc"] for c in ORDER]))
    out["pixel_fm"] = float(np.mean([fm[c]["pixel_ap"] for c in ORDER[:14]]))
    out["image_fm"] = float(np.mean([fm[c]["image_auroc"] for c in ORDER[:14]]))
    out["stage_macro"] = {t + 1: {"pixel_ap": float(np.mean([cells[(t, c)]["pixel_ap"] for c in ORDER[:t + 1]])),
                                  "image_auroc": float(np.mean([cells[(t, c)]["image_auroc"] for c in ORDER[:t + 1]]))}
                          for t in range(15)}
    return out


def vectors(S, arm, phase, metric):
    pc = S[arm]["per_category"]
    if phase == "FM":
        key = "pixel_fm" if metric == "pixel" else "image_fm"
        return [pc[c][key] for c in ORDER[:14]]
    key = f"{phase}_{'pixel_ap' if metric == 'pixel' else 'image_auroc'}"
    return [pc[c][key] for c in ORDER]


def compare(S, left, right, family):
    rows = []
    for metric in ("pixel", "image"):
        for phase in ("acq", "final", "FM"):
            ci = boot(vectors(S, left, phase, metric), vectors(S, right, phase, metric))
            per = dict(zip(ORDER if phase != "FM" else ORDER[:14],
                           np.subtract(vectors(S, left, phase, metric), vectors(S, right, phase, metric)).tolist()))
            rows.append({"family": family, "comparison": f"{left} - {right}", "metric": metric, "phase": phase,
                         **ci, "n_categories_positive": int(sum(v > 0 for v in per.values())), "per_category": per})
    return rows


def pareto(points, keys):
    """keys: list of (field, sense) with sense +1 = maximize, -1 = minimize."""
    front = []
    for a, pa in points.items():
        dom = False
        for b, pb in points.items():
            if a == b:
                continue
            ge = all((pb[f] - pa[f]) * s >= 0 for f, s in keys)
            gt = any((pb[f] - pa[f]) * s > 0 for f, s in keys)
            if ge and gt:
                dom = True; break
        if not dom:
            front.append(a)
    return sorted(front, key=lambda a: points[a]["bytes"])


def main():
    D.verify_lock()
    dec = D.read_json(D.OUT / "B8_DECISION.json")
    budgets = list(D.REGISTERED) + (["B8"] if dec["run_B8"] else [])
    new = [f"{k}_{b}" for b in budgets for k in ("DOM_PQ", "SEL_FP32")]
    arms = new + list(HIST)
    cells = {a: load_cells(a) for a in arms}
    S = {a: summarize(cells[a]) for a in arms}
    bytes_ = {a: (D.read_json(D.OUT / "streams" / a / "STREAM_DONE.json")["final_persistent_bytes"] if a in new
                  else HIST[a]) for a in arms}
    for a in arms:
        S[a]["bytes"] = bytes_[a]
    comps = []
    for b in budgets:
        comps += compare(S, f"DOM_PQ_{b}", f"SEL_FP32_{b}", "fixed_budget")
        comps += compare(S, f"DOM_PQ_{b}", "C1_FPS_FP32", "original_baseline")
        comps += compare(S, f"SEL_FP32_{b}", "C1_FPS_FP32", "original_baseline")
    for k in ("DOM_PQ", "SEL_FP32"):
        seq = budgets
        for i in range(1, len(seq)):
            comps += compare(S, f"{k}_{seq[i]}", f"{k}_{seq[i - 1]}", "capacity")
        comps += compare(S, f"{k}_B4", f"{k}_B1", "capacity")
    comps += compare(S, "DOM_PQ_B1", "C2_FPS_PQ", "domain_at_B1")
    comps += compare(S, "DOM_PQ_B1", "SEL_FP32_B2", "cross_budget")
    comps += compare(S, "DOM_PQ_B1", "SEL_FP32_B4", "cross_budget")
    comps += compare(S, "SEL_FP32_B1", "C1_FPS_FP32", "partition_cost")
    get = lambda l, r, m, p: next(x for x in comps if x["comparison"] == f"{l} - {r}" and x["metric"] == m and x["phase"] == p)

    # ---------------- recommendation
    pts = {a: {"bytes": S[a]["bytes"], "final_pixel_ap": S[a]["final_pixel_ap"], "pixel_fm": S[a]["pixel_fm"],
               "final_image_auroc": S[a]["final_image_auroc"], "acq_pixel_ap": S[a]["acq_pixel_ap"]} for a in arms}
    best_acc = sorted(arms, key=lambda a: (-pts[a]["final_pixel_ap"], pts[a]["bytes"]))[0]
    mx_acq = max(p["acq_pixel_ap"] for p in pts.values()); mx_fin = max(p["final_pixel_ap"] for p in pts.values())
    mx_auc = max(p["final_image_auroc"] for p in pts.values())
    elig = [a for a in arms if pts[a]["acq_pixel_ap"] >= mx_acq - 0.02 and pts[a]["final_pixel_ap"] >= mx_fin - 0.02]
    best_ret = sorted(elig, key=lambda a: (pts[a]["pixel_fm"], pts[a]["bytes"]))[0]
    near = [a for a in arms if pts[a]["final_pixel_ap"] >= pts[best_acc]["final_pixel_ap"] - 0.010
            and pts[a]["final_image_auroc"] >= mx_auc - 0.010 and pts[a]["pixel_fm"] <= pts[best_acc]["pixel_fm"] + 0.010]
    # amendment 06: the locked rule can yield an empty near-best set (no crash; reported as None, rule unchanged)
    small = sorted(near, key=lambda a: (pts[a]["bytes"], -pts[a]["final_pixel_ap"]))[0] if near else None
    blocking = {a: [k for k, ok in (("pixel_ap", pts[a]["final_pixel_ap"] >= pts[best_acc]["final_pixel_ap"] - 0.010),
                                    ("image_auroc", pts[a]["final_image_auroc"] >= mx_auc - 0.010),
                                    ("pixel_fm", pts[a]["pixel_fm"] <= pts[best_acc]["pixel_fm"] + 0.010)) if not ok]
                for a in arms}
    rec = {"best_accuracy": best_acc, "best_retention": best_ret, "best_retention_eligible": elig,
           "near_best_set": near, "smallest_near_best": small, "pareto_recommendation": small,
           "near_best_failed_constraints": blocking, "max_final_image_auroc": mx_auc,
           "pareto_frontier_3d": pareto(pts, [("bytes", -1), ("final_pixel_ap", 1), ("pixel_fm", -1)]),
           "pareto_frontier_2d": pareto(pts, [("bytes", -1), ("final_pixel_ap", 1)]), "points": pts,
           "status": RULES["recommendation"]["status"]}

    # ---------------- verdict
    dm = get("DOM_PQ_B1", "C2_FPS_PQ", "pixel", "acq")
    v1 = dm["lo"] > 0 and S["DOM_PQ_B1"]["acq_pixel_ap"] >= S["C1_FPS_FP32"]["acq_pixel_ap"] - 0.02
    v2 = {}
    for b in budgets:
        fp, fi = get(f"DOM_PQ_{b}", f"SEL_FP32_{b}", "pixel", "final"), get(f"DOM_PQ_{b}", f"SEL_FP32_{b}", "image", "final")
        v2[b] = "ADVANTAGE" if fp["lo"] > 0 and fi["point"] >= -0.01 else "INFERIOR" if fp["hi"] < 0 else "PARITY"
    c41, c42 = get("DOM_PQ_B4", "DOM_PQ_B1", "pixel", "final"), get("DOM_PQ_B4", "DOM_PQ_B2", "pixel", "final")
    v3 = ("USEFUL" + ("_SATURATED_BY_B2" if c42["point"] < 0.005 else "")) if c41["lo"] > 0 and c41["point"] >= 0.01 \
        else "NOT_USEFUL"
    c1 = S["C1_FPS_FP32"]
    v4 = [b for b in D.REGISTERED if S[f"DOM_PQ_{b}"]["acq_pixel_ap"] >= c1["acq_pixel_ap"] - 0.02
          and S[f"DOM_PQ_{b}"]["final_pixel_ap"] >= c1["final_pixel_ap"] - 0.01
          and S[f"DOM_PQ_{b}"]["final_image_auroc"] >= c1["final_image_auroc"] - 0.02]
    reg = [v2[b] for b in D.REGISTERED]
    overall = ("PQ_INFERIOR_STOP_COMPRESSION" if all(v == "INFERIOR" for v in reg) else
               "PQ_ADVANTAGE" if any(v == "ADVANTAGE" for v in reg) else "PQ_PARITY_NO_ADVANTAGE")
    verdict = {"overall": overall, "smallest_advantage_budget": next((b for b in D.REGISTERED if v2[b] == "ADVANTAGE"), None),
               "V1_domain_resolved_in_stream": bool(v1), "V2_fixed_budget": v2, "V3_capacity": v3,
               "V4_credible_detector_budgets": v4, "V4_credible_detector": bool(v4), "rules": RULES["verdict"]}

    out = {"schema": "c2dom_sweep_analysis_v1", "lock_sha256": D.C.sha(D.LOCK), "budgets": budgets,
           "summary": {a: {k: v for k, v in S[a].items() if k not in ("per_category", "stage_macro")} for a in arms},
           "per_category": {a: S[a]["per_category"] for a in arms},
           "stage_macro": {a: S[a]["stage_macro"] for a in arms},
           "comparisons": comps, "recommendation": rec, "verdict": verdict}
    D.write_json(D.OUT / "ANALYSIS.json", out)
    D.write_json(D.OUT / "FINAL_VERDICT.json", {"verdict": verdict, "recommendation": {k: v for k, v in rec.items()
                                                                                        if k != "points"},
                                                "lock_sha256": out["lock_sha256"]})
    with (D.OUT / "STAGE_METRIC_MATRIX.csv").open("x", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "stage", "category", "pixel_ap", "image_auroc", "rows", "reused_from"])
        for a in arms:
            for (t, c), v in sorted(cells[a].items()):
                w.writerow([a, t + 1, c, v["pixel_ap"], v["image_auroc"], v.get("rows", ""), v.get("reused_from", "")])
    with (D.OUT / "PER_CATEGORY_TABLE.csv").open("x", newline="") as f:
        fields = list(next(iter(S[arms[0]]["per_category"].values())))
        w = csv.writer(f); w.writerow(["arm", "category"] + fields)
        for a in arms:
            for c in ORDER:
                w.writerow([a, c] + [S[a]["per_category"][c][k] for k in fields])
    with (D.OUT / "MATCHED_COMPARISONS.csv").open("x", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "comparison", "metric", "phase", "point", "lo", "hi", "n_pos"])
        for x in comps:
            w.writerow([x["family"], x["comparison"], x["metric"], x["phase"], x["point"], x["lo"], x["hi"],
                        x["n_categories_positive"]])
    print(json.dumps(verdict, indent=1)); print(json.dumps({k: v for k, v in rec.items() if k != "points"}, indent=1))
    for a in arms:
        s = S[a]
        print(f"{a:14s} bytes {s['bytes']:>11,d} acqAP {s['acq_pixel_ap']:.4f} finAP {s['final_pixel_ap']:.4f} "
              f"acqAU {s['acq_image_auroc']:.4f} finAU {s['final_image_auroc']:.4f} FMpx {s['pixel_fm']:.4f} "
              f"FMim {s['image_fm']:.4f}")


if __name__ == "__main__":
    main()
