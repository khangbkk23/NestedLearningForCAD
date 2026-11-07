# exps/nl_align_01/g1_forensics.py
"""NL-ALIGN-01 G1 root-cause forensics (READ-ONLY, CPU only, no anomaly label, no training).

Reads the sealed G1 artifacts (checks the seal first), reconstructs every decision condition with an independent AP
implementation (sklearn) and a fresh item bootstrap, and tabulates per-arm/per-seed/per-stage dynamics from the stored
diag scores, stage records and per-step logs. Writes results/hope_cad/nl_align_01/g1_forensics/FORENSICS.json.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python -m exps.nl_align_01.g1_forensics
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from exps.nl_align_01 import g1 as G
from exps.rh01b import common as C

OUT = G.OUT / "g1_forensics"
SEEDS, ARMS, START, LAST = G.SEEDS, G.ARMS, G.START, G.LAST
STAGES = list(range(START, LAST + 1))
FS = ("FS_A", "FS_B", "CN")
REPS, BSEED = 1000, 777


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stage_rec(arm, seed, s):
    if arm == "ID" or s == START:
        return json.load(open(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json"))
    if arm == "FP" and seed == 2711:
        return json.load(open(G.F1_RUN / f"STAGE{s}.json"))
    return json.load(open(G.run_dir(arm, seed) / f"STAGE{s}.json"))


def diag(arm, seed, s):
    a = "ID" if s == START else arm
    return torch.load(G.G1 / "diag" / f"{a}_seed{seed}_stage{s}.pt", weights_only=False)


def per_cat(res, gen, cats):
    """Independent per-category patch arrays: (scores, labels, item ids) with the registered inclusion rule."""
    s_all, c_all = res[gen]["scores"].numpy(), res[gen]["coverage"].numpy()
    keys = res["keys"]
    out = {}
    for c in cats:
        rows = [j for j, k in enumerate(keys) if k[0] == c]
        s, cv = s_all[rows].reshape(len(rows), -1), c_all[rows].reshape(len(rows), -1)
        out[c] = (s, cv)
    return out


def cat_stats(s, cv):
    keep = (cv >= 0.5) | (cv == 0)
    y = (cv >= 0.5)[keep]
    sc = s[keep]
    neg, pos = s[cv == 0], s[cv >= 0.5]
    return {"ap": float(average_precision_score(y, sc)),
            "neg_mean": float(neg.mean()), "neg_q99": float(np.quantile(neg, 0.99)),
            "neg_q999": float(np.quantile(neg, 0.999)), "pos_mean": float(pos.mean()),
            "pos_q10": float(np.quantile(pos, 0.10)),
            # within-image AP mean (ranking inside each item only; insensitive to cross-image tail inflation)
            "within_ap": float(np.nanmean([average_precision_score((v >= 0.5)[(v >= 0.5) | (v == 0)],
                                                                   r[(v >= 0.5) | (v == 0)])
                                           for r, v in zip(s, cv) if (v >= 0.5).any() and (v == 0).any()]))}


def boot_ap(s, cv, rng, reps):
    """Item bootstrap of pooled AP (independent of g1.weighted_ap): resample items with replacement."""
    n = len(s)
    out = np.empty(reps)
    keep = (cv >= 0.5) | (cv == 0)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        kk = keep[idx]
        out[r] = average_precision_score((cv[idx] >= 0.5)[kk], s[idx][kk])
    return out


def logs(arm, seed, s):
    p = G.run_dir(arm, seed) / f"LOG{s}.pt"
    if not p.exists():
        return None
    l = {k: np.asarray(v, dtype=np.float64) for k, v in torch.load(p).items()}
    out = {"clip_frac": float(np.mean(l["norm_pre_clip"] > 0.1)),
           "norm_pre_clip_median": float(np.median(l["norm_pre_clip"]))}
    if len(l["aux_loss"]):
        al, sc, cos = l["aux_loss"], l["scale"], l["cos"]
        n = len(al)
        q = [slice(0, n // 4), slice(n // 4, n // 2), slice(n // 2, 3 * n // 4), slice(3 * n // 4, n)]
        sat = al < 1e-3
        first_sat = int(np.argmax(np.convolve(sat, np.ones(20), "valid") >= 20)) if sat.sum() >= 20 else None
        out.update({
            "aux_loss_mean": float(np.nanmean(al)), "aux_loss_quarters": [float(np.nanmean(al[x])) for x in q],
            "sat_frac_1e-3": float(np.mean(sat)), "sat_frac_1e-5": float(np.mean(al < 1e-5)),
            "sat_frac_quarters": [float(np.mean(sat[x])) for x in q], "first_20step_saturated_run": first_sat,
            "scale_median": float(np.nanmedian(sc)), "scale_frac_gt1": float(np.mean(sc > 1)),
            "scale_frac_gt100": float(np.mean(sc > 100)),
            "nA_over_nN_median_unsat": float(np.nanmedian((l["nA"] / l["nN"])[~sat])) if (~sat).any() else None,
            "cos_mean": float(np.nanmean(cos)), "cos_q05_q95": np.nanquantile(cos, [0.05, 0.95]).tolist(),
            "cos_neg_frac": float(np.mean(cos < 0)),
            "cos_mean_unsat": float(np.nanmean(cos[~sat])) if (~sat).any() else None,
            "cos_mean_sat": float(np.nanmean(cos[sat])) if sat.any() else None,
            "nN_median": float(np.median(l["nN"]))})
    return out


def main():
    lk = json.load(open(G.SEAL))
    bad = [n for n, d in lk["files"].items() if sha(G.ROOT / n) != d]
    assert not bad, bad
    dec = json.load(open(G.DECISION))
    out = {"seal_ok": True, "seal_sha": sha(G.SEAL), "decision_sha": sha(G.DECISION), "completion": {}}
    # --- completion / provenance
    for arm in ARMS:
        for seed in SEEDS:
            recs = [stage_rec(arm, seed, s) for s in range(START + 1, LAST + 1)]
            out["completion"][f"{arm}_{seed}"] = {
                "stages": [r.get("index") for r in recs], "nonfinite_skips": [len(r["nonfinite_skips"]) for r in recs],
                "seconds": [round(r["seconds"], 1) for r in recs],
                "source": ("R15" if arm == "ID" else "F1_reused" if (arm == "FP" and seed == 2711) else "G1_runner")}
    # --- per arm x seed x stage tables
    tab, raw9 = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            for s in STAGES:
                if s == START and arm != "ID":
                    continue
                res = diag(arm, seed, s)
                cats = list(C.STREAM[:s])
                for g in G.GENS:
                    pc = per_cat(res, g, cats)
                    st = {c: cat_stats(*pc[c]) for c in cats}
                    tab[f"{arm}|{seed}|{s}|{g}"] = st
                    if s == LAST:
                        raw9[(arm, seed, g)] = pc
    out["tables"] = tab
    # --- A. independent reconstruction of the gate quantities
    rng = np.random.default_rng(BSEED)
    cats9 = list(C.STREAM[:LAST])
    boots = {}
    for (arm, seed, g), pc in [(k, v) for k, v in raw9.items() if k[0] in ("ID", "FS_A", "FS_B", "CN")]:
        # paired bootstrap: same item indices for every arm -> use a per-(seed, g, cat) rng stream
        boots[(arm, seed, g)] = np.mean([boot_ap(*pc[c], np.random.default_rng([BSEED, seed, G.GENS.index(g), j]),
                                                 REPS) for j, c in enumerate(cats9)], 0)

    def macro(arm, seed, g, s=LAST):
        return float(np.mean([tab[f"{arm if s != START else 'ID'}|{seed}|{s}|{g}"][c]["ap"] for c in C.STREAM[:s]]))

    def delta(arm, g, ref="ID"):
        per = [macro(arm, sd, g) - macro(ref, sd, g) for sd in SEEDS]
        b = np.mean([boots[(arm, sd, g)] - boots[(ref, sd, g)] for sd in SEEDS], 0)
        return {"delta": float(np.mean(per)), "per_seed": per, "ci95": np.percentile(b, [2.5, 97.5]).tolist(),
                "se": float(b.std()), "_b": b}

    recon = {}
    for name, (arm, T, Dg) in G.DIRECTIONS.items():
        dT, dD, dCN = delta(arm, T), delta(arm, Dg), delta("CN", Dg)
        vs = dD["_b"] - dCN["_b"]
        held = [np.mean([stage_rec(arm, sd, LAST)["heldout_normal"][c]["global_cosine"] for c in C.EVAL])
                - np.mean([stage_rec("ID", sd, LAST)["heldout_normal"][c]["global_cosine"] for c in C.EVAL])
                for sd in SEEDS]
        conds = {
            "manip_delta>=0.01": (dT["delta"], dT["delta"] >= G.GATE["manip_min"]),
            "manip_ciLow>0": (dT["ci95"][0], dT["ci95"][0] > 0),
            "transfer_delta>=0.005": (dD["delta"], dD["delta"] >= G.GATE["transfer_min"]),
            "transfer_ciLow>0": (dD["ci95"][0], dD["ci95"][0] > 0),
            "transfer_vsCN_ciLow>0": (float(np.percentile(vs, 2.5)), float(np.percentile(vs, 2.5)) > 0),
            "transfer_3of3_seeds": (sum(x > 0 for x in dD["per_seed"]), all(x > 0 for x in dD["per_seed"])),
            "goodhart_ciHigh<0": (dD["ci95"][1], dD["ci95"][1] < 0),
            "harm_seedmean>0.003": (float(np.mean(held)), float(np.mean(held)) > G.GATE["harm_max"])}
        recon[name] = {"arm": arm, "T": T, "D": Dg,
                       "dT": {k: v for k, v in dT.items() if k != "_b"}, "dD": {k: v for k, v in dD.items() if k != "_b"},
                       "dCN_D": {k: v for k, v in dCN.items() if k != "_b"},
                       "vsCN": {"delta": dD["delta"] - dCN["delta"], "ci95": np.percentile(vs, [2.5, 97.5]).tolist()},
                       "harm_per_seed": held, "conditions": {k: [float(v[0]), bool(v[1])] for k, v in conds.items()},
                       "registered": {k: dec["direction_verdicts"][name][k] for k in
                                      ("manipulation", "transfer", "goodhart", "harm", "reason")}}
        # minimum detectable effect (80% power, two-sided 5%) from the bootstrap SE
        recon[name]["mde_D_80pct"] = 2.8 * dD["se"]
    out["reconstruction"] = recon
    # --- B. dynamics: logs, records, heldout tails, acquisition
    dyn = {}
    for arm in ARMS:
        for seed in SEEDS:
            for s in range(START + 1, LAST + 1):
                r = stage_rec(arm, seed, s)
                cur = C.STREAM[s - 1]
                hn = r["heldout_normal"]
                d = {"heldout_gc": {c: hn[c]["global_cosine"] for c in C.EVAL},
                     "heldout_img_max": {c: float(np.max(hn[c]["image_score_RD"])) for c in C.EVAL},
                     "heldout_img_q90": {c: float(np.quantile(hn[c]["image_score_RD"], 0.9)) for c in C.EVAL},
                     "heldout_img_mean": {c: float(np.mean(hn[c]["image_score_RD"])) for c in C.EVAL},
                     "train_loss_last": r["loss_curve_every50"][-1],
                     "acq": {g: {k: tab[f"{arm}|{seed}|{s}|{g}"][cur][k] for k in ("ap", "neg_mean", "neg_q99")}
                             for g in G.GENS},
                     "displacement_total": (float(np.sqrt(sum(v ** 2 for v in r["displacement_from_stage_start"].values())))
                                            if r.get("displacement_from_stage_start") else None),
                     "dist_R15_total": (float(np.sqrt(sum(v ** 2 for v in r["distance_to_R15_same_stage"].values())))
                                        if r.get("distance_to_R15_same_stage") else None),
                     "dist_R15_blocks": r.get("distance_to_R15_same_stage")}
                if arm in FS:
                    d["log"] = logs(arm, seed, s)
                dyn[f"{arm}|{seed}|{s}"] = d
    idlog = logs("ID", 2711, START + 1)
    out["id_parity_log_s7"] = idlog
    out["dynamics"] = dyn
    # --- seed-mean summaries per arm and stage
    summ = {}
    for arm in ARMS:
        for s in STAGES:
            if s == START and arm != "ID":
                continue
            row = {}
            for g in G.GENS:
                cats = list(C.STREAM[:s])
                k = [tab[f"{arm}|{sd}|{s}|{g}"] for sd in SEEDS]
                row[g] = {"ap": float(np.mean([np.mean([t[c]["ap"] for c in cats]) for t in k])),
                          "within_ap": float(np.mean([np.nanmean([t[c]["within_ap"] for c in cats]) for t in k])),
                          "neg_q99": float(np.mean([np.mean([t[c]["neg_q99"] for c in cats]) for t in k])),
                          "neg_q999": float(np.mean([np.mean([t[c]["neg_q999"] for c in cats]) for t in k])),
                          "neg_mean": float(np.mean([np.mean([t[c]["neg_mean"] for c in cats]) for t in k])),
                          "pos_mean": float(np.mean([np.mean([t[c]["pos_mean"] for c in cats]) for t in k]))}
            summ[f"{arm}|{s}"] = row
    out["summary"] = summ
    # --- per-category stage-9 deltas vs ID (seed mean) for both generators
    pcd = {}
    for arm in ARMS[1:]:
        for g in G.GENS:
            pcd[f"{arm}|{g}"] = {c: {k: float(np.mean([tab[f"{arm}|{sd}|{LAST}|{g}"][c][k]
                                                       - tab[f"ID|{sd}|{LAST}|{g}"][c][k] for sd in SEEDS]))
                                     for k in ("ap", "within_ap", "neg_q99", "pos_mean")}
                                 for c in cats9}
    out["per_category_delta_s9"] = pcd
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FORENSICS.json").write_text(json.dumps(out, indent=1, default=float))
    print("written", OUT / "FORENSICS.json")


if __name__ == "__main__":
    main()
