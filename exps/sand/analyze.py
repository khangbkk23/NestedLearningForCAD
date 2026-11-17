"""SAND-FINAL locked analysis (protocol sections 4-6). Rules are fixed in agents/plans/sand_final_protocol.md.

    python -m exps.sand.analyze phase0
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from exps.nl_sgm_01a import am03_eval as A3
from exps.sand import common as S
from exps.sand import score as SC
from exps.task_p.p0_normal_support import require

ORDER = S.ORDER
KEYS = [(t, c) for t in range(15) for c in ORDER[:t + 1]]


def boot(a, b):
    return A3.bootstrap_delta(a, b, seed=S.BOOT["seed"], B=S.BOOT["B"])


def summarize(cells):
    """cells[(t, c)] = {'pixel_ap', 'image_auroc'}; acquisition, final, Eq.8 FM (as the C2-DOM / AM03 analysis)."""
    acq = {c: cells[(i, c)] for i, c in enumerate(ORDER)}
    fin = {c: cells[(14, c)] for c in ORDER}
    fm = {c: {m: max(cells[(t, c)][m] for t in range(i, 14)) - fin[c][m] for m in ("pixel_ap", "image_auroc")}
          for i, c in enumerate(ORDER[:14])}
    out = {"per_category": {c: {"acq_pixel_ap": acq[c]["pixel_ap"], "final_pixel_ap": fin[c]["pixel_ap"],
                                "acq_image_auroc": acq[c]["image_auroc"], "final_image_auroc": fin[c]["image_auroc"],
                                "pixel_fm": fm.get(c, {}).get("pixel_ap"), "image_fm": fm.get(c, {}).get("image_auroc")}
                            for c in ORDER}}
    for k in ("acq_pixel_ap", "final_pixel_ap", "acq_image_auroc", "final_image_auroc"):
        out[k] = float(np.mean([out["per_category"][c][k] for c in ORDER]))
    out["pixel_fm"] = float(np.mean([fm[c]["pixel_ap"] for c in ORDER[:14]]))
    out["image_fm"] = float(np.mean([fm[c]["image_auroc"] for c in ORDER[:14]]))
    out["acq_to_final_pixel_ap_positive_part"] = float(np.mean(
        [max(acq[c]["pixel_ap"] - fin[c]["pixel_ap"], 0) for c in ORDER[:14]]))
    out["stage_macro"] = {t + 1: {m: float(np.mean([cells[(t, c)][m] for c in ORDER[:t + 1]]))
                                  for m in ("pixel_ap", "image_auroc")} for t in range(15)}
    return out


def edge(cells):
    """Acquisition / final macro only (variants evaluated at those two boundaries)."""
    return {"acq_pixel_ap": float(np.mean([cells[(i, c)]["pixel_ap"] for i, c in enumerate(ORDER)])),
            "acq_image_auroc": float(np.mean([cells[(i, c)]["image_auroc"] for i, c in enumerate(ORDER)])),
            "final_pixel_ap": float(np.mean([cells[(14, c)]["pixel_ap"] for c in ORDER])),
            "final_image_auroc": float(np.mean([cells[(14, c)]["image_auroc"] for c in ORDER])),
            "per_category": {c: {"acq_pixel_ap": cells[(i, c)]["pixel_ap"], "final_pixel_ap": cells[(14, c)]["pixel_ap"],
                                 "acq_image_auroc": cells[(i, c)]["image_auroc"],
                                 "final_image_auroc": cells[(14, c)]["image_auroc"]} for i, c in enumerate(ORDER)}}


def compare(Sm, left, right):
    rows = []
    for phase, key, cats in (("acq", "acq_pixel_ap", ORDER), ("final", "final_pixel_ap", ORDER), ("FM", "pixel_fm", ORDER[:14]),
                             ("acq", "acq_image_auroc", ORDER), ("final", "final_image_auroc", ORDER),
                             ("FM", "image_fm", ORDER[:14])):
        a = [Sm[left]["per_category"][c][key] for c in cats]
        b = [Sm[right]["per_category"][c][key] for c in cats]
        rows.append({"comparison": f"{left} - {right}", "phase": phase, "metric": "pixel" if "pixel" in key else "image",
                     **boot(a, b), "n_categories_positive": int(sum(x > y for x, y in zip(a, b))),
                     "per_category": {c: x - y for c, x, y in zip(cats, a, b)}})
    return rows


def decoder_cells(tag):
    arm = SC.arms()[tag]
    src = arm["decoder_ref"] if arm["decoder_ref"] is not None else S.OUT / tag / "decoder_canonical"
    ev = {s: S.read_json(src / f"EVAL_STAGE{s:02d}.json") for s in range(1, 16)}
    return {(t, c): {m: ev[t + 1]["per_category"][c][m] for m in ("pixel_ap", "image_auroc")} for t, c in KEYS}


def fused_cells(tag):
    return {(t, c): S.read_json(S.OUT / tag / "cells" / f"stage{t + 1:02d}_{c}.json") for t, c in KEYS}


def phase0(tag="A3_s2711"):
    S.verify_lock()
    target = S.OUT / "PHASE0_VERDICT.json"
    require(not target.exists(), "one-shot")
    raw = fused_cells(tag)
    A1 = decoder_cells(tag)
    A2 = {k: {m: v["support_canonical"][m] for m in ("pixel_ap", "image_auroc")} for k, v in raw.items()}
    A3c = {k: {m: v["fusion_P"][m] for m in ("pixel_ap", "image_auroc")} for k, v in raw.items()}
    Sm = {"A1": summarize(A1), "A2": summarize(A2), "A3": summarize(A3c)}
    ek = [k for k in KEYS if k[0] == 14 or ORDER[k[0]] == k[1]]
    var = {name: edge({k: raw[k][key] for k in ek}) for name, key in
           (("A3_S1_dinomaly_readout", "fusion_S1"), ("A3_D1_refreshed_calibrators", "fusion_D1"),
            ("TH_theta_channel_under_P", "theta_P"))}
    gate = S.gate0(Sm["A1"], Sm["A2"], Sm["A3"])
    comps = compare(Sm, "A3", "A1") + compare(Sm, "A3", "A2") + compare(Sm, "A1", "A2")
    checks = {"a1_image_score_max_abs_diff": None, "a2_image_score_max_abs_diff": max(
        v["support_canonical"]["image_score_max_abs_diff"] for v in raw.values()),
        "a1_auroc_recomputed_max_abs_diff": max(abs(raw[k]["decoder_image_auroc"] - A1[k]["image_auroc"]) for k in KEYS),
        "decoder_check": S.read_json(S.OUT / tag / "DECODER_CHECK.json")}
    cal = {}
    for i, c in enumerate(ORDER):
        a, f = raw[(i, c)]["calibration"], raw[(14, c)]["calibration"]
        cal[c] = {"frozen": a["frozen"], "refreshed_at_final": f["refreshed"],
                  "eval_probe_z_theta_mean_acq": a["eval_probe_z_theta"]["mean"],
                  "eval_probe_z_m_mean_acq": a["eval_probe_z_m"]["mean"],
                  "eval_probe_z_theta_mean_final": f["eval_probe_z_theta"]["mean"],
                  "eval_probe_z_m_mean_final": f["eval_probe_z_m"]["mean"],
                  "eval_probe_z_theta_std_acq": a["eval_probe_z_theta"]["std"],
                  "eval_probe_z_m_std_acq": a["eval_probe_z_m"]["std"],
                  "heldout_theta_z_mean_acq": a.get("heldout_theta", {}).get("z_mean_under_frozen"),
                  "theta_is_min_fraction_acq": raw[(i, c)]["fusion_P"]["theta_is_min_fraction"],
                  "theta_is_min_fraction_final": raw[(14, c)]["fusion_P"]["theta_is_min_fraction"]}
    res = {"schema": "sand_phase0_v1", "tag": tag, "lock_sha256": S.sha(S.LOCK), "summary": Sm, "variants_non_gating": var,
           "comparisons": comps, "gate_G0": gate, "oracle_checks": checks, "calibration": cal, "bootstrap": S.BOOT,
           "exposure": "MVTec test split historically exposed; exploratory", "utc": S.now()}
    S.write_json(S.OUT / "PHASE0_RESULTS.json", res)
    with (S.OUT / "PHASE0_STAGE_MATRIX.csv").open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "category", "A1_pixel_ap", "A1_image_auroc", "A2_pixel_ap", "A2_image_auroc",
                    "A3_pixel_ap", "A3_image_auroc", "theta_is_min_fraction"])
        for t, c in KEYS:
            w.writerow([t + 1, c, A1[(t, c)]["pixel_ap"], A1[(t, c)]["image_auroc"], A2[(t, c)]["pixel_ap"],
                        A2[(t, c)]["image_auroc"], A3c[(t, c)]["pixel_ap"], A3c[(t, c)]["image_auroc"],
                        raw[(t, c)]["fusion_P"]["theta_is_min_fraction"]])
    S.write_json(target, {"phase": 0, "gate_G0": gate, "results_sha256": S.sha(S.OUT / "PHASE0_RESULTS.json"),
                          "A1": {k: Sm["A1"][k] for k in ("final_pixel_ap", "final_image_auroc")},
                          "A2": {k: Sm["A2"][k] for k in ("final_pixel_ap", "final_image_auroc")},
                          "A3": {k: Sm["A3"][k] for k in ("final_pixel_ap", "final_image_auroc")}, "utc": S.now()})
    print(gate)
    for a in ("A1", "A2", "A3"):
        print(a, {k: round(v, 4) for k, v in Sm[a].items() if isinstance(v, float)})
    for a, v in var.items():
        print(a, {k: round(x, 4) for k, x in v.items() if isinstance(x, float)})


if __name__ == "__main__":
    {"phase0": phase0}[sys.argv[1]]()
