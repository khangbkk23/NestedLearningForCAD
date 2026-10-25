# exps/rh01b/posthoc.py
"""RH-01b post-decision descriptive analysis for the report (task §5). Reads only finished RH-01b artifacts (metrics,
replicates, decision, training records) and normal-only forensics; computes nothing that could change the
preregistered decision, which DECISION.json already holds.

  python -m exps.rh01b.posthoc                  # after DECISION.json; writes results/hope_cad/rh01b/POSTHOC.json
  python -m exps.rh01b.posthoc --dry 2711       # code check on the seeds scored so far; output to stdout only
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr, t as student_t

from exps.dinomaly import common as D
from exps.rh01b import analysis as A
from exps.rh01b import common as C
from exps.rh01b import run as RR

OUT = C.OUT
NL = D.ROOT / "results/hope_cad/nlcad"
RH01 = D.ROOT / "results/hope_cad/rh01"
BUDGET_GPU_H = 15.0


def seed_level(values):
    v = np.array(list(values), dtype=float)
    n = len(v)
    out = {"values": v.tolist(), "mean": float(v.mean()), "positive": int((v > 0).sum()), "n": n,
           "min": float(v.min()), "max": float(v.max())}
    if n > 1:
        sd = float(v.std(ddof=1))
        h = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        out |= {"sd": sd, "t_interval95": [out["mean"] - h, out["mean"] + h],
                "note": f"seed-level t interval, df = {n - 1}: descriptive, very wide by construction"}
    return out


def training_audit(seeds):
    rows = {}
    for s in seeds:
        for arm in ("R15", "J15"):
            t = D.read(OUT / f"runs/{arm}_seed{s}/TRAIN.json")
            st = t["stages"]
            rows[f"{arm}/{s}"] = {"gpu_hours_training": sum(x["seconds"] for x in st) / 3600,
                                  "skips": sum(len(x["nonfinite_skips"]) for x in st),
                                  "skip_events": [x["nonfinite_skips"] for x in st if x["nonfinite_skips"]],
                                  "numerics": t["numerics"], "peak_allocated_mib": t["peak_allocated_mib"],
                                  "stage_seconds": [x["seconds"] for x in st]}
    return rows


def chain_durations():
    """Wall-clock of every chain step from logs/chain.status (START/DONE pairs)."""
    start, out = {}, {}
    for line in (OUT / "logs/chain.status").read_text().splitlines():
        ts, kind, *name = line.split()
        if not name:
            continue
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if kind == "START":
            start[name[0]] = t
        elif kind == "DONE" and name[0] in start:
            out[name[0]] = (t - start[name[0]]).total_seconds() / 3600
    return out


def heldout(seeds):
    out = {}
    for s in seeds:
        r = D.read(OUT / f"runs/R15_seed{s}/TRAIN.json")["stages"]
        j = D.read(OUT / f"runs/J15_seed{s}/TRAIN.json")["stages"][0]["heldout_normal"]
        out[s] = {c: {"R15_acq": r[A.ACQ[c] - 1]["heldout_normal"][c]["global_cosine"],
                      "R15_final": r[-1]["heldout_normal"][c]["global_cosine"], "J15": j[c]["global_cosine"]}
                  for c in C.EVAL}
        for c in C.EVAL:
            v = out[s][c]
            v["R15_over_J15"], v["R15_final_over_acq"] = v["R15_final"] / v["J15"], v["R15_final"] / v["R15_acq"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", type=int, nargs="+")
    a = ap.parse_args()
    RR.verify_lock()
    seeds = tuple(a.dry) if a.dry else tuple(RR.SEEDS)
    decision = None if a.dry else D.read(OUT / "DECISION.json")
    per_seed = {s: A.summarize((s,)) for s in seeds}
    pooled = A.summarize(seeds)
    res = {"seeds": list(seeds), "decision": None if decision is None else decision["decision"],
           "G15_seed_level": seed_level(per_seed[s]["G15"] for s in seeds),
           "Fbar_seed_level": seed_level(per_seed[s]["Fbar"] for s in seeds),
           "categories": {}, "per_seed": {}, "normal_only": {}, "audit": {}}
    for c in C.EVAL:
        res["categories"][c] = {
            "J_minus_R_seed_level": seed_level(per_seed[s]["categories"][c]["J_minus_R"] for s in seeds),
            "F_seed_level": seed_level(per_seed[s]["categories"][c]["F"] for s in seeds),
            "pooled_seeds": {k: pooled["categories"][c][k] for k in ("J15", "R15_acq", "R15_final", "J_minus_R",
                                                                       "J_minus_R_ci95", "F", "F_ci95")}}
    for s in seeds:
        p = per_seed[s]
        res["per_seed"][s] = {"G15": p["G15"], "G15_ci95_images": p["G15_ci95"], "Fbar": p["Fbar"],
                              "macro": p["macro"], "pooled": p["pooled"],
                              "categories": {c: {k: p["categories"][c][k] for k in
                                                 ("J15", "R15_acq", "R15_final", "J_minus_R", "J_minus_R_ci95", "F",
                                                  "F_ci95", "image_auroc", "trajectory")} for c in C.EVAL}}
    ho = heldout(seeds)
    pts = [(ho[s][c]["R15_over_J15"], per_seed[s]["categories"][c]["J_minus_R"]) for s in seeds for c in C.EVAL]
    res["normal_only"] = {"heldout_global_cosine": ho,
                          "spearman_heldout_ratio_vs_AP_gap": list(map(float, spearmanr(*zip(*pts)))) if len(pts) > 2 else None,
                          "coverage_forensic": {s: D.read(f) for s in seeds
                                                if (f := NL / f"FORENSIC_COVERAGE_ADJ_R15_{s}_to15.json").exists()}}
    audit = training_audit(seeds)
    steps = chain_durations()
    total = sum(v["gpu_hours_training"] for v in audit.values())
    res["audit"] = {"training": audit, "chain_step_hours": steps, "training_gpu_hours_total": total,
                    "chain_gpu_hours_total": sum(steps.values()), "budget_gpu_hours": BUDGET_GPU_H,
                    "probe": D.read(OUT / "PROBE.json"),
                    "contention_note": "R15/J15 seed 2711 overlapped NLCAD-01 Stage C on the same GPU (user-authorised); "
                                       "their wall-clock is inflated"}
    rh01 = D.read(RH01 / "FUTILITY.json")
    res["rh01_bf16_reference"] = {"2711": {"J15": rh01["macro"]["J15"], "R15_final": rh01["macro"]["R15_final"],
                                           "G15": rh01["G15"]},
                                  "2712_R15": D.read(RH01 / "R15_FORGETTING.json")["2712"]}
    if a.dry:
        print({k: res[k] for k in ("G15_seed_level", "Fbar_seed_level")})
        print("normal_only spearman", res["normal_only"]["spearman_heldout_ratio_vs_AP_gap"], "audit total",
              round(total, 2), "chain", round(res["audit"]["chain_gpu_hours_total"], 2))
        return
    D.write_json(OUT / "POSTHOC.json", res)
    print("posthoc", res["decision"], res["G15_seed_level"], flush=True)


if __name__ == "__main__":
    main()
