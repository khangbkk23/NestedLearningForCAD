# exps/ptag/analysis.py
"""PTAG-01 one-shot alignment analysis (CPU) and independent audit. Real AP is reused, never re-measured."""
from __future__ import annotations

import csv
import itertools
import json

import numpy as np
import torch

from exps.ptag import common as C
from exps.ptag import metrics as M
from exps.task_p.p0_normal_support import require

SYNTHETIC, NORMAL_ONLY = ("Q1", "Q2", "Q3"), ("Q4", "Q5", "Q6")
TAU_MIN, LB_MIN, AGREE_MIN, N_MIN, DELTA_MIN, CEILING_MIN = 0.5, 0.3, 0.8, 10, 0.01, 0.5
REP_FAMILIES = ("P1", "P2")


def load():
    man = C.read(C.OUT / "MANIFEST.json")
    z = np.load(C.OUT / "PROXIES.npz")
    q = {p: k for k, p in enumerate(z["proxies"].tolist())}
    row = {cid: i for i, cid in enumerate(z["ids"].tolist())}
    cells = {x["id"]: x for x in man["cells"]}
    require(set(row) == set(cells), "proxy cells differ from manifest")
    files = {}
    real_rep = {}
    for cid, x in cells.items():
        if x["rep_source"]:
            f, key, lo, hi = x["rep_source"]
            if f not in files:
                files[f] = C.read(C.ROOT / f)
            real_rep[cid] = (np.asarray(files[f][key][lo]) + np.asarray(files[f][key][hi])) / 2
    return man, cells, z["point"], z["reps"], q, row, real_rep


class Data:
    def __init__(self):
        self.man, self.cells, self.P, self.R, self.q, self.row, self.real_rep = load()

    def proxy(self, cid, p, b=None):
        v = self.P[self.row[cid], self.q[p]] if b is None else self.R[self.row[cid], self.q[p], b]
        return float(v)

    def real(self, cid, b=None):
        if b is None or cid not in self.real_rep:
            return self.cells[cid]["real_ap"]
        return float(self.real_rep[cid][b])

    def strata(self, p, split, families):
        out = {}
        for cid, x in self.cells.items():
            if x["split"] == split and x["family"] in families and np.isfinite(self.proxy(cid, p)):
                out.setdefault((x["family"], x["category"]), []).append(cid)
        return {k: v for k, v in out.items() if len(v) >= 4 and np.ptp([self.real(c) for c in v]) > 0}

    def mean_tau(self, strata, p, b=None):
        taus = {k: M.tau_b([self.proxy(c, p, b) for c in v], [self.real(c, b) for c in v]) for k, v in strata.items()}
        return (float(np.mean(list(taus.values()))) if taus else float("nan")), taus

    def contrasts(self, p, split, families):
        out = []
        for fam, name, cat, pairs in self.man["contrasts"][split]:
            if fam not in families or not all(np.isfinite(self.proxy(c, p)) for pr in pairs for c in pr):
                continue
            out.append((fam, name, cat, [tuple(pr) for pr in pairs]))
        return out

    def delta(self, pairs, fn):
        return float(np.mean([fn(n) - fn(r) for r, n in pairs]))


def agreement(d, p, cons, b=None):
    agree = []
    for _, _, _, pairs in cons:
        dr = d.delta(pairs, lambda c: d.real(c, b))
        dp = d.delta(pairs, lambda c: d.proxy(c, p, b))
        agree.append(bool(dp != 0 and np.sign(dp) == np.sign(dr)))
    return agree


def gate(d, p, split, families):
    strata = d.strata(p, split, families)
    tau, per = d.mean_tau(strata, p)
    reps = np.array([d.mean_tau(strata, p, b)[0] for b in range(C.REPS)]) if strata else np.array([np.nan])
    cons = d.contrasts(p, split, families)
    qual = [cn for cn in cons if abs(d.delta(cn[3], d.real)) >= DELTA_MIN]
    agree = agreement(d, p, qual)
    agree_reps = np.array([np.mean(agreement(d, p, qual, b)) for b in range(C.REPS)]) if qual else np.array([np.nan])
    effective = len({frozenset(cn[3]) for cn in qual})
    rate = float(np.mean(agree)) if agree else float("nan")
    lb = float(np.percentile(reps, 2.5)) if strata else float("nan")
    valid = bool(strata and tau >= TAU_MIN and lb >= LB_MIN and len(qual) >= N_MIN and effective >= N_MIN
                 and rate >= AGREE_MIN)
    return {"mean_tau": tau, "tau_ci95": [lb, float(np.percentile(reps, 97.5))] if strata else None,
            "strata": {f"{k[0]}/{k[1]}": {"tau": v, "cells": len(strata[k])} for k, v in per.items()},
            "contrasts_total": len(cons), "qualifying": len(qual), "effective": effective,
            "agreement": rate,
            "agreement_ci95": [float(np.percentile(agree_reps, 2.5)), float(np.percentile(agree_reps, 97.5))]
            if qual else None,
            "contrast_detail": [{"family": f, "name": n, "category": c, "pairs": len(pr),
                                 "delta_real": d.delta(pr, d.real), "delta_proxy": d.delta(pr, lambda x: d.proxy(x, p)),
                                 "agree": a} for (f, n, c, pr), a in zip(qual, agree)],
            "valid": valid}


def ceiling(d, families=REP_FAMILIES):
    out = {}
    for (fam, cat), ids in d.strata("Q4", "selection", families).items():
        require(all(c in d.real_rep for c in ids), "ceiling needs replicates")
        half = C.REPS // 2
        out[f"{fam}/{cat}"] = float(np.mean([M.tau_b([d.real(c, b) for c in ids], [d.real(c, b + half) for c in ids])
                                             for b in range(half)]))
    return out


def analyze():
    target = C.OUT / "ALIGNMENT.json"
    require(not target.exists(), "analysis is one-shot")
    d = Data()
    all_f = ("P1", "C00", "CALM", "NTM", "P2")
    res = {"proxies": {}, "noise_ceiling": ceiling(d)}
    res["noise_ceiling_mean"] = float(np.mean(list(res["noise_ceiling"].values())))
    for p in C.PROXIES:
        fams = {"Q5": ("P2",), "Q6": ("CALM",)}.get(p, all_f)
        full = gate(d, p, "selection", fams)
        rep_f = tuple(f for f in fams if f in REP_FAMILIES)
        rep = gate(d, p, "selection", rep_f) if rep_f else {"valid": None, "note": "no family with real replicates"}
        per_family = {f: d.mean_tau(d.strata(p, "selection", (f,)), p)[0] for f in fams}
        cross = {}
        if p in SYNTHETIC:
            for cat in C.CATS:
                ids = [c for c, x in d.cells.items() if x["split"] == "selection" and x["category"] == cat]
                cross[cat] = M.tau_b([d.proxy(c, p) for c in ids], [d.real(c) for c in ids])
        valid = bool(full["valid"] and (rep["valid"] if rep_f and p != "Q5" else full["valid"]))
        entry = {"families": list(fams), "full": full, "replicate_subset": rep, "per_family_tau": per_family,
                 "cross_family_pooled_tau": cross, "valid": valid}
        if valid or full["valid"]:
            ho = gate(d, p, "heldout", ("P2",)) if "P2" in fams else None
            entry["heldout"] = ho
            entry["confirmed"] = bool(ho and ho["mean_tau"] >= TAU_MIN and ho["qualifying"] >= 1
                                      and ho["agreement"] >= AGREE_MIN)
        res["proxies"][p] = entry
        print(p, "tau", round(full["mean_tau"], 3), "lb", full["tau_ci95"] and round(full["tau_ci95"][0], 3),
              "agree", round(full["agreement"], 3) if full["qualifying"] else None, "n", full["qualifying"],
              "valid", valid, flush=True)
    syn_ok = [p for p in SYNTHETIC if res["proxies"][p]["valid"] and res["proxies"][p].get("confirmed")]
    nrm_ok = [p for p in NORMAL_ONLY if res["proxies"][p]["valid"] and res["proxies"][p].get("confirmed")]
    unconfirmed = [p for p in C.PROXIES if res["proxies"][p]["valid"] and not res["proxies"][p].get("confirmed")]
    disagree = [p for p in C.PROXIES if res["proxies"][p]["replicate_subset"].get("valid") is not None
                and p != "Q5" and res["proxies"][p]["full"]["valid"] != res["proxies"][p]["replicate_subset"]["valid"]]
    underpowered = all(res["proxies"][p]["full"]["qualifying"] < N_MIN for p in SYNTHETIC)
    if syn_ok:
        verdict = "SYNTHETIC_OUTER_OBJECTIVE_ADMISSIBLE"
    elif nrm_ok:
        verdict = "NORMAL_ONLY_OUTER_OBJECTIVE_ADMISSIBLE"
    elif unconfirmed or disagree or underpowered or res["noise_ceiling_mean"] < CEILING_MIN:
        verdict = "PTAG_INCONCLUSIVE"
    else:
        verdict = "NO_TRAINING_ONLY_OBJECTIVE_ALIGNED"
    res["decision_inputs"] = {"synthetic_valid_confirmed": syn_ok, "normal_valid_confirmed": nrm_ok,
                              "valid_unconfirmed": unconfirmed, "full_vs_replicate_disagree": disagree,
                              "synthetic_underpowered": underpowered}
    res["verdict"] = verdict
    with (C.OUT / "ALIGNMENT_TABLE.csv").open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "family", "category", "split", "real_ap", *C.PROXIES])
        for cid, x in sorted(d.cells.items()):
            w.writerow([cid, x["family"], x["category"], x["split"], f"{x['real_ap']:.6f}",
                        *[("" if not np.isfinite(d.proxy(cid, p)) else f"{d.proxy(cid, p):.6f}") for p in C.PROXIES]])
    C.write_json(target, res)
    print("VERDICT", verdict)


# ---------------------------------------------------------------- independent audit

def tau_b_naive(x, y):
    """O(n^2) Kendall tau-b from its definition (independent of scipy)."""
    nc = nd = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if a == 0 and b == 0:
            continue
        if a == 0:
            tx += 1
        elif b == 0:
            ty += 1
        elif a == b:
            nc += 1
        else:
            nd += 1
    den = np.sqrt((nc + nd + tx) * (nc + nd + ty))
    return 0.0 if den == 0 else (nc - nd) / den


def audit(device):
    target = C.OUT / "AUDIT.json"
    require(not target.exists(), "audit exists")
    from exps.task_p.p2.metrics import exact_ranking
    from exps.ptag import scoring as S
    out = {}
    d = Data()
    al = C.read(C.OUT / "ALIGNMENT.json")
    worst = 0.0
    for p in C.PROXIES:
        for key, v in al["proxies"][p]["full"]["strata"].items():
            fam, cat = key.split("/")
            ids = d.strata(p, "selection", (fam,))[(fam, cat)]
            worst = max(worst, abs(tau_b_naive([d.proxy(c, p) for c in ids], [d.real(c) for c in ids]) - v["tau"]))
    out["tau_naive_max_abs"] = worst
    rng = np.random.default_rng(7)
    ap_diff, q45 = [], []
    for c in C.CATS:
        syn = torch.load(C.OUT / "synthetic" / f"{c}.pt", weights_only=False)
        hw = tuple(syn["hw"]); n = hw[0] * hw[1]
        z = np.load(C.OUT / "scores" / f"{c}.npz")
        ids = z["ids"].tolist()
        for j in rng.choice(len(ids), 2, replace=False):
            for fi, f in enumerate(C.FAMILIES):
                sel = np.r_[0:10, C.HOSTS + fi * 100:C.HOSTS + fi * 100 + 30]
                masks = [np.zeros(hw, bool)] * 10 + [np.unpackbits(m)[:n].reshape(hw).astype(bool)
                                                     for m in syn["masks"][fi * 100:fi * 100 + 30]]
                maps = M.native_maps(z["scores"][j][sel], hw, device)
                lab = torch.from_numpy(np.stack(masks)).to(device)
                ap_diff.append(abs(M.exact_ap(maps, lab) - exact_ranking(maps, lab)["pixel_ap"]))
            clean = z["scores"][j][:C.HOSTS].ravel()
            q45.append(abs(-np.quantile(clean, 0.99, method="inverted_cdf") - d.proxy(ids[j], "Q4")))
    out["exact_ap_vs_exact_ranking_max_abs"] = float(max(ap_diff))
    out["q4_recompute_max_abs"] = float(max(q45))
    gidx = C.calm_global_index()
    man = C.read(C.OUT / "MANIFEST.json")
    rows, _, row_ids = S.calm_bank("CALM_L1", C.CALM_FINAL)
    src = (row_ids // 784).numpy()
    loo = {}
    for c in C.CATS:
        obj = torch.load(C.CALM / f"features/mvtec/{c}_train.pt", map_location="cpu", weights_only=False, mmap=True)
        idx = {r["relative_path"]: i for i, r in enumerate(obj["rows"])}
        own, other = [], []
        for p in man["hosts"][c]["hosts"]:
            x = obj["features"][idx[p]].double()
            r_h = rows[torch.from_numpy(src == gidx[p])].double()
            if len(r_h):
                own.append(float(torch.cdist(r_h, x).min(1).values.max()))
            other.append(float(torch.cdist(rows[torch.from_numpy(src != gidx[p])][:2000].double(), x).min(1).values.median()))
        loo[c] = {"hosts_with_rows": len(own), "max_own_row_distance": max(own) if own else None,
                  "median_other_row_distance": float(np.median(other))}
    out["loo_row_provenance"] = loo
    cell = next(x for x in man["cells"] if x["family"] == "P2" and x["category"] == "carpet")
    syn = torch.load(C.OUT / "synthetic" / "carpet.pt", weights_only=False)
    z = np.load(C.OUT / "scores" / "carpet.npz")
    j = z["ids"].tolist().index(cell["id"])
    cpu = S.p2_scores(S.p2_learner(cell["scorer"], "cpu"), syn["features"][:16], "cpu").numpy()
    out["p2_cpu_vs_gpu_max_abs"] = float(np.abs(cpu - z["scores"][j][:16]).max())
    out["pass"] = bool(out["tau_naive_max_abs"] < 1e-9 and out["exact_ap_vs_exact_ranking_max_abs"] < 1e-9
                       and out["q4_recompute_max_abs"] < 1e-6 and out["p2_cpu_vs_gpu_max_abs"] < 1e-3
                       and all(v["max_own_row_distance"] is None or v["max_own_row_distance"] < 0.25 * v["median_other_row_distance"]
                               for v in loo.values()))
    C.write_json(target, out)
    print(json.dumps(out, indent=1))
