"""NL-ALIGN-02-MVTec post-hoc DESCRIPTIVE analyses (written after FINAL_VERDICT; not part of any gate, tunes nothing).

1. Effective auxiliary update mass from the per-step training logs (G1 stages 7-9, M3 stages 10-15):
   r_t = scale_t * ||g_aux|| / ||g_N|| = rho * ||g_aux|| / (||g_aux|| + EPS), i.e. the injected auxiliary norm as a fraction
   of the normal gradient norm (nominal 0.25).
2. Leave-category-out sensitivity of the final stage-15 macro contrasts. The bootstrap is stratified by category x label,
   so category deltas are independent across categories within a replicate; the macro CI is approximated by a normal
   interval from the per-category bootstrap CIs (validated against the full reported CI).
3. Acquisition-vs-retention decomposition: acquisition AP deltas for categories 7-15 and acquisition->final forgetting.
Reads only sealed outputs (LOG*.pt, M4_METRICS.json). Usage: python -m exps.nl_align_02.posthoc
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
G1 = ROOT / "results/hope_cad/nl_align_01/g1/runs"
OUT = ROOT / "results/hope_cad/nl_align_02_mvtec"
M3 = OUT / "m3/runs"
SEEDS = (2711, 2712, 2713)
ORDER = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile",
         "toothbrush", "transistor", "wood", "zipper"]


def aux_mass():
    res = {}
    for arm, stages in (("FS_B", range(7, 16)), ("CN", range(7, 16)), ("FS_A", range(7, 10))):
        for st in stages:
            rows = []
            for s in SEEDS:
                base = G1 if st <= 9 else M3
                d = torch.load(base / f"{arm}_seed{s}/LOG{st}.pt", weights_only=False)
                nA, nN = np.asarray(d["nA"]), np.asarray(d["nN"])
                r = np.asarray(d["scale"]) * nA / nN
                L = np.asarray(d["aux_loss"])
                rows.append([r.mean(), (r >= 0.2).mean(), (L < 1e-5).mean(), float(np.nanmean(d["cos"]))])
            m = np.mean(rows, 0)
            res[f"{arm}|{st}"] = {"eff_ratio_mean": m[0], "frac_steps_ratio_ge_0.2": m[1], "frac_aux_loss_lt_1e-5": m[2],
                                  "cos_gN_gaux_mean": m[3]}
    return res


def leave_out(metrics):
    out = {}
    for con in ("FS_B-ID", "FS_B-CN", "CN-ID"):
        pc = metrics["contrasts"][con]["per_category_final"]
        var = {c: ((v["ci95"][1] - v["ci95"][0]) / 3.92) ** 2 for c, v in pc.items()}
        rows = {}
        for drop in ([], ["grid"], ["zipper"], ["toothbrush"], ["grid", "zipper"], ["grid", "zipper", "toothbrush"]):
            ks = [c for c in pc if c not in drop]
            m = sum(pc[c]["delta"] for c in ks) / len(ks)
            se = math.sqrt(sum(var[c] for c in ks)) / len(ks)
            rows["-".join(drop) or "none"] = {"delta": m, "ci95_normal_approx": [m - 1.96 * se, m + 1.96 * se]}
        out[con] = {"reported_full_ci95": metrics["contrasts"][con]["final"]["ci95"], "leave_out": rows,
                    "n_categories_positive": sum(pc[c]["delta"] > 0 for c in pc)}
    return out


def acquisition(metrics):
    F = metrics["forgetting_acq_to_final"]
    new = ORDER[6:]
    acq = {c: {a: F[a]["acquisition_ap"][c] for a in ("ID", "FS_B", "CN")} for c in new}
    d_id = [acq[c]["FS_B"] - acq[c]["ID"] for c in new]
    d_cn = [acq[c]["FS_B"] - acq[c]["CN"] for c in new]
    return {"acquisition_ap_7_15": acq, "mean_acq_delta_FS_B-ID": float(np.mean(d_id)),
            "n_pos_FS_B-ID": int(sum(x > 0 for x in d_id)), "mean_acq_delta_FS_B-CN": float(np.mean(d_cn)),
            "n_pos_FS_B-CN": int(sum(x > 0 for x in d_cn)),
            "forgetting_per_category": {a: F[a]["per_category"] for a in ("ID", "FS_B", "CN")}}


def main():
    metrics = json.loads((OUT / "M4_METRICS.json").read_text())
    res = {"status": "POST-HOC DESCRIPTIVE ONLY: computed after FINAL_VERDICT; no gate, threshold or choice depends on it",
           "aux_mass": aux_mass(), "leave_category_out_final": leave_out(metrics), "acquisition": acquisition(metrics)}
    (OUT / "POSTHOC_DESCRIPTIVE.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["aux_mass"].items():
        print(k, {a: round(b, 4) for a, b in v.items()})
    print(json.dumps(res["leave_category_out_final"], indent=1, default=float)[:2500])
    a = res["acquisition"]
    print("acq", a["mean_acq_delta_FS_B-ID"], a["n_pos_FS_B-ID"], a["mean_acq_delta_FS_B-CN"], a["n_pos_FS_B-CN"])


if __name__ == "__main__":
    main()
