"""NL-RET-00B: native anomaly-discrimination decomposition (acquisition -> stage 15), memory-bounded, CPU only.

Reads only sealed score maps and official masks. Forensic / BENCHMARK_DESCRIPTIVE (MVTec historically exposed).

  lock       AUDIT_LOCK.json + INPUT_MANIFEST.json (hashes, sources, frozen signature/verdict rules) - before any
             new ground-truth statistic
  parity     PARITY.json: exact AP(P_A,N_A), AP(P_F,N_F) for screw/cable/transistor/grid/bottle (seed 2711, ID and
             FS_B) vs sealed M1/M4 values, plus the canonical exact_ap on one category
  run        all 15 categories x 3 seeds x {ID, FS_B}; appends CELLS.jsonl (resumable)
  transitions  factorial at the largest sealed-AP stage-to-stage drop (screw/transistor/cable/capsule) where both
             stage maps exist on the full test set
  summarize  CATEGORY_MECHANISM_MATRIX.csv, SCORE_DISTRIBUTIONS.json, FACTORIAL_AP_DECOMPOSITION.json,
             WITHIN_CROSS_IMAGE_ANALYSIS.json, STATISTICAL_SUMMARY.json, MEMORY_PROFILE.json, VERDICT.json

  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python -m exps.nl_ret_00b.decomp <cmd>
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import subprocess
import time

import numpy as np
import psutil
import torch

from exps.cadfinal import common as K
from exps.cadfinal import metrics as M
from exps.nl_ret_00b import core as X
from exps.task_p.p0_normal_support import require

ROOT = K.ROOT
OUT = ROOT / "results/hope_cad/nl_ret_00b"
A2 = ROOT / "results/hope_cad/nl_align_02_mvtec"
M1DIR = K.OUT / "scores/mvtec"
LOCK, MANIFEST, PARITY = OUT / "AUDIT_LOCK.json", OUT / "INPUT_MANIFEST.json", OUT / "PARITY.json"
CELLS, TRANS = OUT / "CELLS.jsonl", OUT / "TRANSITIONS.jsonl"
SEEDS = (2711, 2712, 2713)
ARMS = ("ID", "FS_B")
ORDER = list(K.ORDERS["mvtec"])
FOCUS = ("screw", "transistor", "cable", "capsule")
PARITY_CATS = ("screw", "cable", "transistor", "grid", "bottle")
PARITY_TOL = 1e-6
RSS_TARGET_GIB = 3.0
SRC = ["exps/nl_ret_00b/core.py", "exps/nl_ret_00b/decomp.py", "exps/cadfinal/metrics.py", "exps/cadfinal/common.py",
       "exps/calm02/native.py", "exps/dinomaly/readout.py"]

# ---- frozen before any new ground-truth statistic is computed ---------------------------------------------
RULES = {
    "unit": "category x seed x arm; D = AP(P_A,N_A) - AP(P_F,N_F) (positive = deterioration); category reading uses "
            "seed means; seed consistency counts seeds with the same sign/dominance",
    "material": "a category is analysed for signatures only if seed-mean D >= 0.02 (about 4 x Delta_min); otherwise "
                "'STABLE_OR_IMPROVING'",
    "A_positive_suppression": "seed-mean D_P >= 0.6 * D",
    "B_background_tail": "seed-mean D_N >= 0.6 * D",
    "C_mixed": "seed-mean D_P >= 0.25 * D and D_N >= 0.25 * D",
    "D_cross_image_calibration": "mean within-image AP deterioration <= 0.5 * D, OR percentile-rank-normalised pooled "
                                 "AP deterioration <= 0.5 * D",
    "E_no_dominant": "none of A-D, or the dominant component (larger of D_P, D_N) agrees in < 2 of 3 seeds",
    "multi_label": "a category may carry several of A-D",
    "high_forgetting_set": "categories with seed-mean ID signed forgetting >= 0.05 in the sealed M4 table (known before "
                           "this task)",
    "verdict": {
        "NATIVE_FORGETTING_MECHANISM_IDENTIFIED_DESCRIPTIVE": ">= 75% of high-forgetting categories (ID arm) carry the "
            "same single primary signature (A or B) with seed consistency >= 2/3",
        "MIXED_MECHANISMS_IDENTIFIED_DESCRIPTIVE": "not the above, and >= 75% of high-forgetting categories carry at "
            "least one of A-D with seed consistency >= 2/3",
        "MECHANISM_STILL_UNRESOLVED": "otherwise",
        "INTEGRITY_OR_RESOURCE_BLOCKED": "audit or parity failure, or missing cells",
    },
    "limits": "factorial mixes are artificial (not realisable decoder states); D_P/D_N are algebraic attributions, "
              "not causal; mask-conditioned quantities are never training credit",
}
WITHIN_WEIGHTING = "primary: unweighted mean over anomalous images with >=1 positive and >=1 negative pixel; " \
                   "secondary: positive-pixel-weighted mean"


def rss_gib():
    return psutil.Process().memory_info().rss / 2 ** 30


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ---- sources --------------------------------------------------------------------------------------------

def stage_of(cat):
    return ORDER.index(cat) + 1


def src(arm, seed, cat, which):
    """which in {'acq','fin'} -> (path, stage, provenance)."""
    j = stage_of(cat)
    if which == "fin":
        if arm == "ID":
            return M1DIR / f"DINOMALY_R_seed{seed}" / f"fin_{j - 1:02d}.pt", 15, "M1 fin (R15 stage 15)"
        return A2 / "predictions_m4" / f"seed{seed}" / f"FS_B_s15_{cat}.pt", 15, "NL-ALIGN-02 M4 FS_B stage 15"
    if arm == "ID" or j <= 6:
        return M1DIR / f"DINOMALY_R_seed{seed}" / f"acq_{j - 1:02d}.pt", j, "M1 acq (R15 lineage; shared by FS_B for j<=6)"
    d = "predictions" if j <= 9 else "predictions_m4"
    return A2 / d / f"seed{seed}" / f"FS_B_s{j}_{cat}.pt", j, f"NL-ALIGN-02 FS_B own stage {j}"


def stage_map(arm, seed, cat, stage):
    """Full-test map file of `cat` at `stage` (7..15) from NL-ALIGN-02 predictions; None if not produced."""
    if stage < max(7, stage_of(cat)):
        return None
    d = "predictions" if stage <= 9 else "predictions_m4"
    p = A2 / d / f"seed{seed}" / f"{arm}_s{stage}_{cat}.pt"
    return p if p.exists() else None


def sealed_ap(arm, seed, cat, which):
    j = stage_of(cat)
    if which == "fin":
        return K.read(A2 / "M4_METRICS.json")["per_cell"][f"{arm}|{seed}|15|{cat}"]["ap"]
    if j <= 6:
        return K.read(A2 / "M1_METRICS.json")["acquisition_ap_m1_r15"][f"{seed}|{cat}"]
    f = "M1_METRICS.json" if j <= 9 else "M4_METRICS.json"
    return K.read(A2 / f)["per_cell"][f"{arm}|{seed}|{j}|{cat}"]["ap"]


def rows_of(cat):
    ids = K.benchmark_identities("mvtec")
    require([t["task"] for t in ids["tasks"]] == ORDER, "stream order")
    return ids["tasks"][ORDER.index(cat)]["test"]


def category_masks(cat):
    root = ROOT / "data/mvtec"
    rows = rows_of(cat)
    hws = [M.native_hw(root / r["relative_path"]) for r in rows]
    masks = [M.native_mask(root / r["mask_path"] if r["mask_path"] else None, hw) for r, hw in zip(rows, hws)]
    for r, m in zip(rows, masks):
        require(bool(m.any()) == (r["label"] == 1), f"mask polarity/label mismatch {r['relative_path']}")
    return rows, hws, masks


# ---- one model state, streamed image by image ----------------------------------------------------------

def state(path, rows, hws, masks, normalise=False, per_image=True):
    p = torch.load(path, weights_only=False)
    require(p["paths"] == [r["relative_path"] for r in rows], f"identity order {path}")
    s256 = p["s256"]
    del p
    npos = sum(int(m.sum()) for m in masks)
    ngood = sum(m.size for r, m in zip(rows, masks) if r["label"] == 0)
    nbg = sum(int((~m).sum()) for r, m in zip(rows, masks) if r["label"] == 1)
    P = np.empty(npos, np.float32); NG = np.empty(ngood, np.float32); NB = np.empty(nbg, np.float32)
    ip = ig = ib = 0
    img = []
    for i, (r, hw, m) in enumerate(zip(rows, hws, masks)):
        x = M.to_native("dinomaly", s256[i].numpy(), hw)
        require(x.shape == m.shape, "map/mask shape")
        if normalise:
            x = X.pct_rank(x)
        if r["label"] == 0:
            v = x.ravel(); NG[ig:ig + v.size] = v; ig += v.size
            if per_image:
                q = np.quantile(v, [0.5, 0.99])
                img.append({"path": r["relative_path"], "label": 0, "neg_q50": float(q[0]), "neg_q99": float(q[1])})
        else:
            pos, neg = x[m], x[~m]
            P[ip:ip + pos.size] = pos; ip += pos.size
            NB[ib:ib + neg.size] = neg; ib += neg.size
            if per_image:
                ns = np.sort(neg)
                img.append({"path": r["relative_path"], "label": 1, "n_pos": int(pos.size),
                            "within_ap": X.ap_pn(pos, [ns]), "pos_mean": float(pos.mean()),
                            "neg_q50": float(ns[(ns.size - 1) // 2]), "neg_q99": float(ns[int(0.99 * (ns.size - 1))])})
                del ns
        del x
    require(ip == npos and ig == ngood and ib == nbg, "pixel partition")
    NG.sort(); NB.sort()
    del s256
    return P, NG, NB, img


def dist(P, NG, NB):
    qs = (0.1, 0.5, 0.9, 0.99)
    ps = np.sort(P)
    out = {"pos": {"n": int(P.size), "mean": float(P.mean(dtype=np.float64)),
                   **{f"q{int(q * 100)}": float(ps[int(np.floor(q * (ps.size - 1)))]) for q in qs}}}
    del ps
    nq = (0.5, 0.9, 0.95, 0.99, 0.999)
    for name, parts in (("neg_all", [NG, NB]), ("neg_good", [NG]), ("neg_bg_anom", [NB])):
        q = X.quantiles(parts, nq)
        out[name] = {"n": X.total(parts), "mean": X.mean_parts(parts), "median": q[0.5],
                     **{f"q{str(k).replace('0.', '')}": v for k, v in q.items() if k != 0.5}}
    return out


def pair(arm, seed, cat, data=None):
    t0 = time.time()
    rows, hws, masks = data or category_masks(cat)
    pa, sa, _ = src(arm, seed, cat, "acq")
    pf, sf, _ = src(arm, seed, cat, "fin")
    res = {"arm": arm, "seed": seed, "category": cat, "acq_stage": sa, "acq_src": str(pa.relative_to(ROOT)),
           "fin_src": str(pf.relative_to(ROOT)), "acq_sha": K.sha(pa), "fin_sha": K.sha(pf)}
    PA, NGA, NBA, IA = state(pa, rows, hws, masks)
    PF, NGF, NBF, IF = state(pf, rows, hws, masks)
    aa, ff = X.ap_pn(PA, [NGA, NBA]), X.ap_pn(PF, [NGF, NBF])
    fa, af = X.ap_pn(PF, [NGA, NBA]), X.ap_pn(PA, [NGF, NBF])
    d, dp, dn = X.shapley(aa, fa, af, ff)
    require(abs(dp + dn - d) < 1e-12, "Shapley identity")
    res["ap"] = {"AA": aa, "FA": fa, "AF": af, "FF": ff, "D": d, "D_P": dp, "D_N": dn,
                 "identity_err": abs(dp + dn - d)}
    res["dist_acq"], res["dist_fin"] = dist(PA, NGA, NBA), dist(PF, NGF, NBF)
    t = X.quantiles([NGA, NBA], (0.99, 0.999))
    res["thresholds_acq"] = {"tau99": t[0.99], "tau999": t[0.999]}
    nall = X.total([NGA, NBA])
    res["threshold_mass"] = {
        "pos_below_tau99_acq": float(np.mean(PA < t[0.99])), "pos_below_tau99_fin": float(np.mean(PF < t[0.99])),
        "pos_below_tau999_acq": float(np.mean(PA < t[0.999])), "pos_below_tau999_fin": float(np.mean(PF < t[0.999])),
        "neg_above_tau99_acq": X.count_ge([NGA, NBA], t[0.99]) / nall,
        "neg_above_tau99_fin": X.count_ge([NGF, NBF], t[0.99]) / nall,
        "neg_above_tau999_acq": X.count_ge([NGA, NBA], t[0.999]) / nall,
        "neg_above_tau999_fin": X.count_ge([NGF, NBF], t[0.999]) / nall,
        "good_above_tau99_fin": X.count_ge([NGF], t[0.99]) / max(NGF.size, 1),
        "good_above_tau99_acq": X.count_ge([NGA], t[0.99]) / max(NGA.size, 1),
        "bg_above_tau99_fin": X.count_ge([NBF], t[0.99]) / max(NBF.size, 1),
        "bg_above_tau99_acq": X.count_ge([NBA], t[0.99]) / max(NBA.size, 1)}
    del PA, NGA, NBA, PF, NGF, NBF
    gc.collect()
    # within-image and cross-image calibration
    require([a["path"] for a in IA] == [b["path"] for b in IF], "image identity A/F")
    an = [(a, b) for a, b in zip(IA, IF) if a["label"] == 1]
    wa = np.array([a["within_ap"] for a, _ in an]); wf = np.array([b["within_ap"] for _, b in an])
    w = np.array([a["n_pos"] for a, _ in an], float)
    dw = wf - wa
    order = np.argsort(dw)
    res["within"] = {"n_images": len(an), "mean_acq": float(wa.mean()), "mean_fin": float(wf.mean()),
                     "median_acq": float(np.median(wa)), "median_fin": float(np.median(wf)),
                     "wmean_acq": float(np.sum(w * wa) / w.sum()), "wmean_fin": float(np.sum(w * wf) / w.sum()),
                     "mean_delta": float(dw.mean()), "frac_worse": float(np.mean(dw < -1e-9)),
                     "worst": [(an[i][0]["path"], float(dw[i])) for i in order[:3]],
                     "best": [(an[i][0]["path"], float(dw[i])) for i in order[::-1][:3]]}
    def cal(I):
        q50 = np.array([x["neg_q50"] for x in I]); q99 = np.array([x["neg_q99"] for x in I])
        g = np.array([x["label"] == 0 for x in I])
        return {"offset_mean": float(q50.mean()), "scale_mean": float((q99 - q50).mean()),
                "q99_sd_all": float(q99.std()), "q99_cv_all": float(q99.std() / q99.mean()),
                "q99_sd_good": float(q99[g].std()) if g.any() else None,
                "q99_sd_anom_bg": float(q99[~g].std()), "q99_mean_good": float(q99[g].mean()) if g.any() else None,
                "q99_mean_anom_bg": float(q99[~g].mean())}
    ca, cf = cal(IA), cal(IF)
    off_d = np.array([b["neg_q50"] - a["neg_q50"] for a, b in zip(IA, IF)])
    sc_d = np.array([(b["neg_q99"] - b["neg_q50"]) - (a["neg_q99"] - a["neg_q50"]) for a, b in zip(IA, IF)])
    res["calibration"] = {"acq": ca, "fin": cf, "offset_delta_mean": float(off_d.mean()),
                          "offset_delta_sd": float(off_d.std()), "scale_delta_mean": float(sc_d.mean()),
                          "scale_delta_sd": float(sc_d.std())}
    del IA, IF
    # percentile-rank normalised pooled AP (retrospective diagnostic; preserves within-image order)
    PA, NGA, NBA, _ = state(pa, rows, hws, masks, normalise=True, per_image=False)
    PF, NGF, NBF, _ = state(pf, rows, hws, masks, normalise=True, per_image=False)
    naa, nff = X.ap_pn(PA, [NGA, NBA]), X.ap_pn(PF, [NGF, NBF])
    nfa, naf = X.ap_pn(PF, [NGA, NBA]), X.ap_pn(PA, [NGF, NBF])
    nd, ndp, ndn = X.shapley(naa, nfa, naf, nff)
    res["normalised"] = {"AA": naa, "FF": nff, "FA": nfa, "AF": naf, "D": nd, "D_P": ndp, "D_N": ndn}
    del PA, NGA, NBA, PF, NGF, NBF
    gc.collect()
    res["seconds"] = time.time() - t0
    res["rss_gib_end"] = rss_gib(); res["peak_rss_gib_so_far"] = peak_gib()
    return res


# ---- commands -------------------------------------------------------------------------------------------

def lock():
    require(not LOCK.exists(), "lock exists (frozen)")
    OUT.mkdir(parents=True, exist_ok=True)
    agents_sha = K.sha(ROOT / "AGENTS.md")
    require(agents_sha == "b88393815586295209c2704a6e16262730f2aba4000a6904b7c5c6d2b65ed622", "AGENTS.md pin")
    m1l = K.read(K.OUT / "M1_LOCK.json")["files"]
    s1 = K.read(A2 / "PRED_SEAL.json")["files"]; s4 = K.read(A2 / "M4_PRED_SEAL.json")["files"]
    cust = K.read(A2 / "CUSTODY.json")["tasks"]
    man, bad = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            for cat in ORDER:
                for which in ("acq", "fin"):
                    p, st, prov = src(arm, seed, cat, which)
                    h = K.sha(p)
                    rel_m1 = str(p.relative_to(K.OUT)) if str(p).startswith(str(K.OUT)) else None
                    rel_a2 = str(p.relative_to(A2)) if str(p).startswith(str(A2)) else None
                    ref = m1l.get(rel_m1) if rel_m1 else (s1.get(rel_a2) or s4.get(rel_a2))
                    if ref != h:
                        bad.append(str(p))
                    man[f"{arm}|{seed}|{cat}|{which}"] = {"path": str(p.relative_to(ROOT)), "stage": st,
                                                          "provenance": prov, "sha256": h, "sealed_by":
                                                          "M1_LOCK" if rel_m1 else "PRED_SEAL/M4_PRED_SEAL",
                                                          "sealed_ap": sealed_ap(arm, seed, cat, which)}
    # FS_B start = R15 stage 6 (shared lineage for categories 1-6)
    start = {}
    for seed in SEEDS:
        rec = K.read(ROOT / f"results/hope_cad/rh01b/runs/R15_seed{seed}/STAGE6.json")
        sc = K.read(M1DIR / f"DINOMALY_R_seed{seed}/SCORES.json")["files"]["acq_05"]
        start[seed] = {"r15_stage6_ckpt": rec["checkpoint_sha256"], "m1_acq05_ckpt": sc["checkpoint_sha256"],
                       "match": rec["checkpoint_sha256"] == sc["checkpoint_sha256"],
                       "g1_start": "exps.nlcad.run.start_state(seed, 6) loads R15 STAGE6 with SHA check"}
    # masks/images vs custody hashes
    root = ROOT / "data/mvtec"
    mask_bad, n_img, n_mask = [], 0, 0
    for cat in ORDER:
        rows = rows_of(cat)
        crow = {r["path"]: r for r in cust[cat]["test"]}
        require([r["relative_path"] for r in rows] == [r["path"] for r in cust[cat]["test"]], f"custody order {cat}")
        for r in rows:
            n_img += 1
            if r["mask_path"]:
                n_mask += 1
                if K.sha(root / r["mask_path"]) != crow[r["relative_path"]]["mask_sha"]:
                    mask_bad.append(r["mask_path"])
    integrity = {"inputs_ok": not bad, "bad_inputs": bad, "start_shared": all(v["match"] for v in start.values()),
                 "masks_ok": not mask_bad, "bad_masks": mask_bad, "n_test_images": n_img, "n_masks": n_mask}
    K.write_json(MANIFEST, {"cells": man, "fsb_start": start, "utc": K.now()})
    K.write_json(LOCK, {"task": "NL-RET-00B", "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                        "head": git("rev-parse", "HEAD"), "agents_md_sha256": agents_sha,
                        "sources": {s: K.sha(ROOT / s) for s in SRC}, "manifest_sha256": K.sha(MANIFEST),
                        "rules": RULES, "within_weighting": WITHIN_WEIGHTING, "parity_tol": PARITY_TOL,
                        "rss_target_gib": RSS_TARGET_GIB, "integrity": integrity,
                        "note": "frozen before any new ground-truth statistic of this task; sealed AP values were "
                                "already known from M1/M4", "utc": K.now()})
    print("lock", integrity)
    require(integrity["inputs_ok"] and integrity["start_shared"] and integrity["masks_ok"], "integrity")


def verify_lock():
    lk = K.read(LOCK)
    require(lk["integrity"]["inputs_ok"] and lk["integrity"]["masks_ok"], "integrity failed at lock")
    for s, h in lk["sources"].items():
        require(K.sha(ROOT / s) == h, f"source changed after lock: {s}")
    require(K.sha(MANIFEST) == lk["manifest_sha256"], "manifest changed")
    return lk


def parity():
    verify_lock()
    man = K.read(MANIFEST)["cells"]
    out, worst = [], 0.0
    for cat in PARITY_CATS:
        data = category_masks(cat)
        for arm in ARMS:
            rows, hws, masks = data
            for which in ("acq", "fin"):
                p, _, _ = src(arm, 2711, cat, which)
                P, NG, NB, _ = state(p, rows, hws, masks, per_image=False)
                v = X.ap_pn(P, [NG, NB]); ref = man[f"{arm}|2711|{cat}|{which}"]["sealed_ap"]
                worst = max(worst, abs(v - ref))
                out.append({"arm": arm, "category": cat, "which": which, "ap_streaming": v, "ap_sealed": ref,
                            "abs_diff": abs(v - ref), "rss_gib": rss_gib()})
                print(out[-1], flush=True)
                del P, NG, NB
                gc.collect()
    # canonical exact_ap (all maps in memory) on the smallest category
    rows, hws, masks = category_masks("toothbrush")
    p, _, _ = src("ID", 2711, "toothbrush", "fin")
    s = torch.load(p, weights_only=False)["s256"]
    maps = [M.to_native("dinomaly", s[i].numpy(), hw) for i, hw in enumerate(hws)]
    canon = M.exact_ap(maps, masks)["ap"]
    P, NG, NB, _ = state(p, rows, hws, masks, per_image=False)
    stream = X.ap_pn(P, [NG, NB])
    res = {"cells": out, "worst_abs_vs_sealed": worst, "canonical_exact_ap_toothbrush_ID_fin": canon,
           "streaming_toothbrush_ID_fin": stream, "canonical_abs_diff": abs(canon - stream),
           "tolerance": PARITY_TOL, "pass": worst <= PARITY_TOL and abs(canon - stream) <= PARITY_TOL,
           "peak_rss_gib": peak_gib(), "utc": K.now()}
    K.write_json(PARITY, res)
    print("parity", res["worst_abs_vs_sealed"], res["canonical_abs_diff"], res["pass"])
    require(res["pass"], "parity failed: stop and diagnose")


def run():
    verify_lock()
    require(K.read(PARITY)["pass"], "parity first")
    done = set()
    if CELLS.exists():
        for line in CELLS.read_text().splitlines():
            r = json.loads(line); done.add((r["arm"], r["seed"], r["category"]))
    for cat in ORDER:
        todo = [(a, s) for s in SEEDS for a in ARMS if (a, s, cat) not in done]
        if not todo:
            continue
        data = category_masks(cat)
        for arm, seed in todo:
            r = pair(arm, seed, cat, data)
            with open(CELLS, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{cat} {arm} {seed} D={r['ap']['D']:+.4f} DP={r['ap']['D_P']:+.4f} DN={r['ap']['D_N']:+.4f} "
                  f"within={r['within']['mean_delta']:+.4f} normD={r['normalised']['D']:+.4f} "
                  f"{r['seconds']:.0f}s rss={r['rss_gib_end']:.2f} peak={r['peak_rss_gib_so_far']:.2f}", flush=True)
        del data
        gc.collect()
    print("run done")


def _sealed_traj(arm, seed, cat):
    """Sealed full-test AP at stages max(7, j)..15 (M1 for 7-9, M4 for 10-15)."""
    m1 = K.read(A2 / "M1_METRICS.json")["per_cell"]; m4 = K.read(A2 / "M4_METRICS.json")["per_cell"]
    out = {}
    for s in range(max(7, stage_of(cat)), 16):
        k = f"{arm}|{seed}|{s}|{cat}"
        v = (m1 if s <= 9 else m4).get(k)
        if v:
            out[s] = v["ap"]
    return out


def transitions():
    verify_lock()
    require(CELLS.exists(), "run first")
    for cat in FOCUS:
        data = category_masks(cat)
        for seed in SEEDS:
            for arm in ARMS:
                tr = _sealed_traj(arm, seed, cat)
                st = sorted(tr)
                if len(st) < 2:
                    rec = {"arm": arm, "seed": seed, "category": cat, "status": "UNAVAILABLE",
                           "reason": f"full-test maps exist only for stages >= 7; {cat} acquired at stage "
                                     f"{stage_of(cat)}; earlier transitions have no full-test predictions"}
                else:
                    drops = {(a, b): tr[a] - tr[b] for a, b in zip(st[:-1], st[1:])}
                    (a, b), dmax = max(drops.items(), key=lambda kv: kv[1])
                    pa, pb = stage_map(arm, seed, cat, a), stage_map(arm, seed, cat, b)
                    rows, hws, masks = data
                    PA, NGA, NBA, _ = state(pa, rows, hws, masks, per_image=False)
                    PF, NGF, NBF, _ = state(pb, rows, hws, masks, per_image=False)
                    aa, ff = X.ap_pn(PA, [NGA, NBA]), X.ap_pn(PF, [NGF, NBF])
                    fa, af = X.ap_pn(PF, [NGA, NBA]), X.ap_pn(PA, [NGF, NBF])
                    d, dp, dn = X.shapley(aa, fa, af, ff)
                    rec = {"arm": arm, "seed": seed, "category": cat, "status": "OK", "transition": [a, b],
                           "selected_by": "largest sealed AP drop among available stage pairs (retrospective)",
                           "sealed_trajectory": tr, "AA": aa, "FF": ff, "D": d, "D_P": dp, "D_N": dn,
                           "dist_from": dist(PA, NGA, NBA), "dist_to": dist(PF, NGF, NBF),
                           "observed_window": f"stages {st[0]}-15 only"}
                    del PA, NGA, NBA, PF, NGF, NBF
                    gc.collect()
                with open(TRANS, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(cat, seed, arm, rec.get("transition"), rec.get("D"), rec.get("D_P"), rec.get("D_N"), flush=True)
        del data
    print("transitions done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lock", "parity", "run", "transitions", "summarize"])
    a = ap.parse_args()
    torch.set_num_threads(8)
    if a.cmd == "summarize":
        from exps.nl_ret_00b import summarize
        summarize.main()
    else:
        globals()[a.cmd]()
