"""NL-RET-00B summarize: applies the rules frozen in AUDIT_LOCK.json to CELLS.jsonl / TRANSITIONS.jsonl."""
from __future__ import annotations

import csv
import json

import numpy as np

from exps.cadfinal import common as K
from exps.nl_ret_00b import decomp as D
from exps.task_p.p0_normal_support import require

OUT = D.OUT


def _cells():
    rows = [json.loads(x) for x in D.CELLS.read_text().splitlines()]
    by = {(r["arm"], r["seed"], r["category"]): r for r in rows}
    require(len(by) == len(rows), "duplicate cells")
    return by


def signatures(cs):
    """cs: the 3 seed cells of one (arm, category). Returns the per-category reading under the frozen rules."""
    d = np.array([c["ap"]["D"] for c in cs]); dp = np.array([c["ap"]["D_P"] for c in cs])
    dn = np.array([c["ap"]["D_N"] for c in cs])
    wd = np.array([-c["within"]["mean_delta"] for c in cs])          # within-image AP deterioration
    nd = np.array([c["normalised"]["D"] for c in cs])
    m, mp, mn, mw, mnd = d.mean(), dp.mean(), dn.mean(), wd.mean(), nd.mean()
    out = {"D": m, "D_P": mp, "D_N": mn, "within_det": mw, "norm_D": nd.mean(),
           "seed_D": d.tolist(), "seed_D_P": dp.tolist(), "seed_D_N": dn.tolist(), "seed_within_det": wd.tolist(),
           "seed_norm_D": nd.tolist(), "n_seeds_D_pos": int((d > 0).sum())}
    if m < 0.02:
        out.update({"material": False, "signatures": ["STABLE_OR_IMPROVING"], "consistency": {}})
        return out
    def per_seed(rule):
        return int(sum(rule(d[i], dp[i], dn[i], wd[i], nd[i]) for i in range(len(d))))
    rules = {"A": lambda D_, P, N, W, Z: D_ > 0 and P >= 0.6 * D_,
             "B": lambda D_, P, N, W, Z: D_ > 0 and N >= 0.6 * D_,
             "C": lambda D_, P, N, W, Z: D_ > 0 and P >= 0.25 * D_ and N >= 0.25 * D_,
             "D": lambda D_, P, N, W, Z: D_ > 0 and (W <= 0.5 * D_ or Z <= 0.5 * D_)}
    sig = []
    if mp >= 0.6 * m: sig.append("A")
    if mn >= 0.6 * m: sig.append("B")
    if mp >= 0.25 * m and mn >= 0.25 * m: sig.append("C")
    if mw <= 0.5 * m or mnd <= 0.5 * m: sig.append("D")
    dom = "P" if mp >= mn else "N"
    agree = int(sum((dp[i] >= dn[i]) == (dom == "P") for i in range(len(d))))
    if not sig or agree < 2:
        sig.append("E")
    out.update({"material": True, "signatures": sig, "dominant_component": dom, "dominance_seed_agreement": agree,
                "consistency": {k: per_seed(r) for k, r in rules.items()}})
    return out


def trajectory_summary():
    """Descriptive summary of STAGE_TRAJECTORY.json (every consecutive transition, stages a(c)..15)."""
    from exps.nl_ret_00b import early as EA
    if not EA.TRAJ_OUT.exists():
        return {"status": "UNAVAILABLE"}
    rec = K.read(EA.TRAJ_OUT)["records"]
    out = {"per_category": {}, "by_transition": {}}
    for a in D.ARMS:
        for c in D.ORDER:
            rs = [r for r in rec if r["arm"] == a and r["category"] == c]
            if not rs:
                continue
            tot = [r["stages"][0]["ap"] - r["stages"][-1]["ap"] for r in rs]
            first = [r["transitions"][0]["D"] for r in rs]
            worst = [max(r["transitions"], key=lambda t: t["D"]) for r in rs]
            loss = [sum(max(t["D"], 0) for t in r["transitions"]) for r in rs]
            gain = [sum(min(t["D"], 0) for t in r["transitions"]) for r in rs]
            trs = [t for r in rs for t in r["transitions"] if t["D"] > 0]
            out["per_category"][f"{a}|{c}"] = {
                "net_D_seed": tot, "first_transition_D_seed": first,
                "gross_loss_seed": loss, "gross_recovery_seed": gain,
                "worst_transition_seed": [(w["from"], w["to"], w["D"], w["D_P"], w["D_N"]) for w in worst],
                "worst_transition_same_in_seeds": max(sum((w["from"], w["to"]) == (v["from"], v["to"]) for v in worst)
                                                      for w in worst),
                "loss_transitions_share_DN_dominant": float(np.mean([t["D_N"] > t["D_P"] for t in trs])) if trs else None,
                "loss_transitions_n": len(trs),
                "trajectory_seedmean": {s["stage"]: float(np.mean([r["stages"][i]["ap"] for r in rs]))
                                        for i, s in enumerate(rs[0]["stages"])},
                "pos_q50_seedmean": {s["stage"]: float(np.mean([r["stages"][i]["pos_q50"] for r in rs]))
                                     for i, s in enumerate(rs[0]["stages"])},
                "neg_q99_seedmean": {s["stage"]: float(np.mean([r["stages"][i]["neg_q99"] for r in rs]))
                                     for i, s in enumerate(rs[0]["stages"])}}
        for to in range(2, 16):                     # old-category damage at each task arrival
            v = [t for r in rec if r["arm"] == a for t in r["transitions"] if t["to"] == to]
            if v:
                out["by_transition"][f"{a}|{to}"] = {
                    "new_task": D.ORDER[to - 1], "n_old_cells": len(v), "mean_D": float(np.mean([t["D"] for t in v])),
                    "mean_D_P": float(np.mean([t["D_P"] for t in v])), "mean_D_N": float(np.mean([t["D_N"] for t in v])),
                    "frac_cells_D_pos": float(np.mean([t["D"] > 0 for t in v]))}
    return out


def main():
    lk = D.verify_lock()
    by = _cells()
    missing = [(a, s, c) for a in D.ARMS for s in D.SEEDS for c in D.ORDER if (a, s, c) not in by]
    m4 = K.read(D.A2 / "M4_METRICS.json")["forgetting_acq_to_final"]
    high = [c for c in D.ORDER if c != "zipper" and m4["ID"]["per_category"][c] >= 0.05]
    cat = {a: {c: signatures([by[(a, s, c)] for s in D.SEEDS]) for c in D.ORDER} for a in D.ARMS} if not missing else {}
    ident = max(abs(r["ap"]["identity_err"]) for r in by.values())

    # ---- artifacts ------------------------------------------------------------------------------------
    K.write_json(OUT / "SCORE_DISTRIBUTIONS.json", {
        "definition": "positives = native mask==1 pixels of anomalous test images; neg_good = all pixels of test/good "
                      "images; neg_bg_anom = mask==0 pixels of anomalous images; quantiles lower-type; tau99/tau999 = "
                      "acquisition-era quantiles of all negatives",
        "cells": {f"{a}|{s}|{c}": {k: r[k] for k in ("acq_stage", "dist_acq", "dist_fin", "thresholds_acq",
                                                     "threshold_mass")} for (a, s, c), r in by.items()}})
    K.write_json(OUT / "FACTORIAL_AP_DECOMPOSITION.json", {
        "definition": "AP(P_x,N_y), x,y in {A=acquisition, F=final}, same native pixels; D = AA - FF; "
                      "D_P = 1/2[(AA-FA)+(AF-FF)], D_N = 1/2[(AA-AF)+(FA-FF)]; positive = deterioration",
        "limits": lk["rules"]["limits"], "max_identity_error": ident,
        "cells": {f"{a}|{s}|{c}": {"raw": r["ap"], "percentile_rank_normalised": r["normalised"]}
                  for (a, s, c), r in by.items()},
        "category_seed_means": {a: {c: {k: v for k, v in cat[a][c].items() if k in
                                        ("D", "D_P", "D_N", "seed_D", "seed_D_P", "seed_D_N")}
                                    for c in D.ORDER} for a in cat}})
    K.write_json(OUT / "WITHIN_CROSS_IMAGE_ANALYSIS.json", {
        "within_weighting": lk["within_weighting"],
        "normalisation": "per-image empirical-CDF (percentile rank) transform; preserves within-image order and ties; "
                         "retrospective diagnostic only, not a deployable scorer",
        "cells": {f"{a}|{s}|{c}": {"within": r["within"], "calibration": r["calibration"],
                                   "normalised": r["normalised"]} for (a, s, c), r in by.items()}})
    # mechanism matrix
    rows = []
    for a in D.ARMS:
        for c in D.ORDER:
            cs = [by[(a, s, c)] for s in D.SEEDS]
            g = cat[a][c]
            mean = lambda f: float(np.mean([f(x) for x in cs]))
            rows.append({
                "arm": a, "category": c, "acq_stage": cs[0]["acq_stage"],
                "acq_ap": mean(lambda x: x["ap"]["AA"]), "final_ap": mean(lambda x: x["ap"]["FF"]),
                "signed_forgetting_D": g["D"], "D_P": g["D_P"], "D_N": g["D_N"],
                "within_ap_acq": mean(lambda x: x["within"]["mean_acq"]),
                "within_ap_delta": mean(lambda x: x["within"]["mean_delta"]),
                "within_frac_images_worse": mean(lambda x: x["within"]["frac_worse"]),
                "norm_pooled_D": g["norm_D"],
                "calibration_sensitivity_D_minus_normD": g["D"] - g["norm_D"],
                "norm_D_P": mean(lambda x: x["normalised"]["D_P"]), "norm_D_N": mean(lambda x: x["normalised"]["D_N"]),
                "pos_median_delta": mean(lambda x: x["dist_fin"]["pos"]["q50"] - x["dist_acq"]["pos"]["q50"]),
                "good_q99_delta": mean(lambda x: x["dist_fin"]["neg_good"]["q99"] - x["dist_acq"]["neg_good"]["q99"]),
                "bg_anom_q99_delta": mean(lambda x: x["dist_fin"]["neg_bg_anom"]["q99"] - x["dist_acq"]["neg_bg_anom"]["q99"]),
                "pos_below_tau99acq_acq": mean(lambda x: x["threshold_mass"]["pos_below_tau99_acq"]),
                "pos_below_tau99acq_fin": mean(lambda x: x["threshold_mass"]["pos_below_tau99_fin"]),
                "neg_above_tau99acq_fin": mean(lambda x: x["threshold_mass"]["neg_above_tau99_fin"]),
                "good_above_tau99acq_fin": mean(lambda x: x["threshold_mass"]["good_above_tau99_fin"]),
                "bg_above_tau99acq_fin": mean(lambda x: x["threshold_mass"]["bg_above_tau99_fin"]),
                "offset_delta_mean": mean(lambda x: x["calibration"]["offset_delta_mean"]),
                "q99_cv_acq": mean(lambda x: x["calibration"]["acq"]["q99_cv_all"]),
                "q99_cv_fin": mean(lambda x: x["calibration"]["fin"]["q99_cv_all"]),
                "seeds_D_positive": g["n_seeds_D_pos"],
                "dominant": g.get("dominant_component", ""), "dominance_seed_agreement": g.get("dominance_seed_agreement", ""),
                "signatures": "+".join(g["signatures"]),
                "consistency_A_B_C_D": "/".join(str(g["consistency"].get(k, "")) for k in "ABCD"),
                "high_forgetting": c in high})
    with open(OUT / "CATEGORY_MECHANISM_MATRIX.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

    # transitions
    trans = [json.loads(x) for x in D.TRANS.read_text().splitlines()] if D.TRANS.exists() else []
    traj = trajectory_summary()

    # statistical summary
    def agg(a, cats):
        g = [cat[a][c] for c in cats]
        return {k: float(np.mean([x[k] for x in g])) for k in ("D", "D_P", "D_N", "within_det", "norm_D")}
    fs_vs_id = {c: {k: cat["FS_B"][c][k] - cat["ID"][c][k] for k in ("D", "D_P", "D_N", "within_det", "norm_D")}
                for c in D.ORDER}
    summ = {"n_cells": len(by), "missing_cells": missing, "max_identity_error": ident,
            "high_forgetting_set": high,
            "macro_all_14": {a: agg(a, [c for c in D.ORDER if c != "zipper"]) for a in D.ARMS},
            "macro_high_forgetting": {a: agg(a, high) for a in D.ARMS},
            "per_category": cat, "FS_B_minus_ID": fs_vs_id,
            "FS_B_minus_ID_macro_14": {k: float(np.mean([fs_vs_id[c][k] for c in D.ORDER if c != "zipper"]))
                                       for k in ("D", "D_P", "D_N", "within_det", "norm_D")},
            "worst_transition_focus": trans, "trajectory": traj,
            "uncertainty_note": "seed-level spread only (3 seeds); no image bootstrap of the factorial terms"}
    K.write_json(OUT / "STATISTICAL_SUMMARY.json", summ)
    prof = {"cells": {f"{a}|{s}|{c}": {"seconds": r["seconds"], "rss_gib_end": r["rss_gib_end"],
                                       "peak_rss_gib_so_far": r["peak_rss_gib_so_far"]} for (a, s, c), r in by.items()},
            "max_peak_rss_gib": max(r["peak_rss_gib_so_far"] for r in by.values()),
            "parity_peak_rss_gib": K.read(D.PARITY)["peak_rss_gib"], "target_gib": lk["rss_target_gib"],
            "total_seconds": sum(r["seconds"] for r in by.values()),
            "method": "image-by-image streaming; one category's bool masks resident; positives/negatives preallocated "
                      "float32 per state; negatives sorted in place; AP by searchsorted counts; no merged arrays"}
    K.write_json(OUT / "MEMORY_PROFILE.json", prof)

    # verdict (frozen rule)
    blocked = bool(missing) or not K.read(D.PARITY)["pass"] or ident > 1e-12
    hs = {c: cat["ID"][c] for c in high} if not blocked else {}
    prim = {c: [s for s in g["signatures"] if s in ("A", "B") and g["consistency"].get(s, 0) >= 2] for c, g in hs.items()}
    anyc = {c: [s for s in g["signatures"] if s in "ABCD" and g["consistency"].get(s, 0) >= 2] for c, g in hs.items()}
    if blocked:
        label = "INTEGRITY_OR_RESOURCE_BLOCKED"
    else:
        counts = {s: sum(s in v for v in prim.values()) for s in ("A", "B")}
        best = max(counts, key=counts.get)
        if counts[best] >= 0.75 * len(high):
            label = "NATIVE_FORGETTING_MECHANISM_IDENTIFIED_DESCRIPTIVE"
        elif sum(bool(v) for v in anyc.values()) >= 0.75 * len(high):
            label = "MIXED_MECHANISMS_IDENTIFIED_DESCRIPTIVE"
        else:
            label = "MECHANISM_STILL_UNRESOLVED"
    K.write_json(OUT / "VERDICT.json", {"label": label, "high_forgetting_set": high, "primary_consistent": prim,
                                        "any_consistent": anyc, "rule": lk["rules"]["verdict"],
                                        "lock_sha256": K.sha(D.LOCK), "utc": K.now()})
    print("verdict", label)
    for r in rows:
        print(f"{r['arm']:5s} {r['category']:11s} D={r['signed_forgetting_D']:+.3f} DP={r['D_P']:+.3f} "
              f"DN={r['D_N']:+.3f} wD={r['within_ap_delta']:+.3f} nD={r['norm_pooled_D']:+.3f} "
              f"gq99={r['good_q99_delta']:+.4f} bq99={r['bg_anom_q99_delta']:+.4f} pmed={r['pos_median_delta']:+.4f} "
              f"{r['signatures']} {r['consistency_A_B_C_D']} dom={r['dominant']}{r['dominance_seed_agreement']}")


if __name__ == "__main__":
    main()
