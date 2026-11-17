"""NL-DRIFT-00 CPU analysis: provenance checks, arms, tracking errors, bootstrap, decision (prereg sections 1-7).

    python -m exps.nl_drift_00.analyze

Reads only STATS.pt (hash-checked against STATS_LOCK.json), the stage records and the R15 protocol. Writes RESULT.json
(create-exclusive). Label-free.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.nl_drift_00 import common as C
from exps.nl_drift_00 import estimators as E

ARMS = ("E0", "E2", "E2A", "E3", "E4", "E4_0", "E5")
FIXED = ("E2", "E2A", "E3")
LEARNED = {"E4": 3, "E4_0": 1, "E5": 3}
VARIANTS = ("raw", "pq")
NB = 15


# ------------------------------------------------------------------ statistics helpers

class Stats:
    def __init__(self, st):
        self.st = st
        self.ridx = {c: {p: i for i, p in enumerate(st["paths"][c])} for c in C.ORDER}
        self.qidx = {c: {p: i for i, p in enumerate(st["pq_paths"][c])} for c in C.ORDER}

    def arr(self, c, variant):
        return self.st["raw"][c] if variant == "raw" else self.st["pq"][c]

    def rows(self, c, paths, variant):
        m = self.ridx[c] if variant == "raw" else self.qidx[c]
        return np.array([m[p] for p in paths])

    def pooled(self, c, paths, t, variant="raw", unclipped=False):
        S = self.arr(c, variant)[self.rows(c, paths, variant), t]
        n = len(paths) * C.TOKENS
        if unclipped:
            return float(S[:, 2].sum() / n)
        mu = S[:, 0].sum() / n
        var = S[:, 1].sum() / n - mu * mu
        C.require(np.isfinite(mu) and var > 0, f"degenerate pooled statistics {c} t={t}")
        return float(mu), float(np.sqrt(var))


def pooled_from_rows(S):
    """S [k, 3] per-image sums -> (mu, sd)."""
    n = len(S) * C.TOKENS
    mu = S[:, 0].sum() / n
    return mu, np.sqrt(max(S[:, 1].sum() / n - mu * mu, 1e-300))


# ------------------------------------------------------------------ provenance

def pv3(S, held, rep):
    out = {"heldout": [], "replay": []}
    for s in range(1, 16):
        rec = C.stage_record(s)
        t = s - 1
        for h, v in (rec.get("heldout_normal") or {}).items():
            got = S.pooled(h, held[h], t, unclipped=True)
            want = v["mean_patch_1mcos"]
            out["heldout"].append({"stage": s, "category": h, "abs_diff": abs(got - want)})
        if rec["replay"]:
            tot, n = 0.0, 0
            for p in rep[s]:
                c = p.split("/")[0]
                tot += S.arr(c, "raw")[S.ridx[c][p], t, 2]
                n += C.TOKENS
            out["replay"].append({"stage": s, "abs_diff": abs(tot / n - rec["replay_normal"]["mean_patch_1mcos"])})
    mx = max(x["abs_diff"] for x in out["heldout"] + out["replay"])
    out["max_abs_diff"] = mx
    out["n_checks"] = len(out["heldout"]) + len(out["replay"])
    out["pass"] = bool(mx <= C.THRESH["pv3_abs"] and len(out["heldout"]) == 15 + 12 + 10 and len(out["replay"]) == 14)
    return out


# ------------------------------------------------------------------ covariates (prereg section 2)

def covariates(S, T, rep):
    """u [15, 15, 3]: (1, u_buf, u_aff), defined for t > a(c); NaN elsewhere. Also buffer sizes for cost."""
    u = np.full((15, NB, 3), np.nan)
    bsize = np.zeros((15, NB), dtype=int)
    cen = {c: np.asarray(S.st["centroid"][c], dtype=np.float64) for c in C.ORDER}
    for i, c in enumerate(C.ORDER):
        a = C.A[c]
        _, sd_full = S.pooled(c, T[c], a)
        for t in range(a + 1, NB):
            B = [p for p in rep[t + 1] if p.startswith(c + "/")]
            bsize[i, t] = len(B)
            if B:
                mu_now = S.pooled(c, B, t)[0]
                mu_prev = S.pooled(c, B, t - 1)[0]
                ub = (mu_now - mu_prev) / sd_full
            else:
                ub = 0.0
            x, y = cen[C.ORDER[t]], cen[c]
            u[i, t] = (1.0, ub, 1.0 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    return u, bsize


# ------------------------------------------------------------------ one deployment draw

def series(S, cohort, cats, pools, variant, n, d):
    """Normalized measurements and the estimand for one draw. Returns dict of arrays over cats."""
    nc = len(cats)
    y = np.full((nc, NB, 2), np.nan)
    star = np.full((nc, NB, 2), np.nan)                       # (mu*, sd*)
    sa = np.zeros((nc, 2))
    rs = np.zeros((nc, 2))
    a = np.array([C.A[c] for c in cats])
    for i, c in enumerate(cats):
        w4, ref = C.witness_sets(cohort, c, pools[c])[d]
        w = w4[:n]
        ai = a[i]
        refrows = S.arr(c, "raw")[S.rows(c, ref, "raw")]                     # [k, 15, 3]
        wrows = S.arr(c, variant)[S.rows(c, w, variant)]
        for t in range(ai, NB):
            star[i, t] = pooled_from_rows(refrows[:, t])
        mu_a, sd_a = star[i, ai]
        sa[i] = (mu_a, sd_a)
        for t in range(ai, NB):
            mw, sw = pooled_from_rows(wrows[:, t])
            y[i, t] = ((mw - mu_a) / sd_a, np.log(sw) - np.log(sd_a))
        rng = np.random.default_rng(C.subset_seed(cohort, c, d, n))
        sub = np.empty((C.N_SUB, 2))
        A0 = refrows[:, ai]
        for j in range(C.N_SUB):
            pick = rng.choice(len(ref), size=n, replace=False)
            mm, ss = pooled_from_rows(A0[pick])
            sub[j] = ((mm - mu_a) / sd_a, np.log(ss) - np.log(sd_a))
        rs[i] = np.maximum(sub.var(axis=0, ddof=1), 1e-12)
    ya = y - y[np.arange(nc), a][:, None, :]                                 # anchored: y_t - y_a
    return {"y": y, "ya": ya, "star": star, "sa": sa, "rs": rs, "a": a}


def errors(est, sr):
    """W, e_mu, e_l [nc, 15] (NaN where no estimate)."""
    mu_a, sd_a = sr["sa"][:, 0][:, None], sr["sa"][:, 1][:, None]
    mu_h = mu_a + sd_a * est[..., 0]
    sd_h = sd_a * np.exp(est[..., 1])
    mu_s, sd_s = sr["star"][..., 0], sr["star"][..., 1]
    W = np.sqrt((mu_h - mu_s) ** 2 + (sd_h - sd_s) ** 2) / sd_s
    return W, (mu_h - mu_s) / sd_s, est[..., 1] - np.log(sd_s / sd_a)


def run_arms(sr, u, u_shuf, phis_in=None):
    a, y, ya, rs = sr["a"], sr["y"], sr["ya"], sr["rs"]
    est = {"E0": E.e0(a), "E2": E.e2(y, a), "E2A": E.e2a(ya, a), "E3": E.e3(ya, a)}
    phis, diags = {}, {}
    for arm, k in LEARNED.items():
        uu = u_shuf if arm == "E5" else u
        est[arm], phis[arm], diags[arm] = E.learned(ya, uu, rs, a, k, None if phis_in is None else phis_in[arm])
    return est, phis, diags


# ------------------------------------------------------------------ bootstrap

def pooled_mean(W):
    return float(np.nansum(W) / np.isfinite(W).sum())


def boot_setup(W_any):
    keep = np.isfinite(W_any).any(1)
    rng = np.random.default_rng(C.BOOT_SEED)
    idx = rng.integers(0, int(keep.sum()), size=(C.BOOT_B, int(keep.sum())))
    return keep, idx


def boot_stats(Wd, keep, idx):
    """Wd: {arm: [nc,15]} -> per arm bootstrap distribution of the pooled mean."""
    out = {}
    for arm, W in Wd.items():
        Wk = W[keep]
        sums = np.nansum(Wk, 1)
        cnt = np.isfinite(Wk).sum(1)
        out[arm] = sums[idx].sum(1) / cnt[idx].sum(1)
    return out


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ------------------------------------------------------------------ main

def main():
    C.install_custody()
    t_start = time.time()
    C.verify_lock()
    sl = C.read_json(C.OUT / "STATS_LOCK.json")
    C.require(C.sha_file(C.OUT / "STATS.pt") == sl["stats_sha256"], "STATS.pt does not match STATS_LOCK")
    C.require(sl["lock_sha256"] == C.sha_file(C.LOCK), "STATS computed under a different LOCK")
    C.require(not (C.OUT / "RESULT.json").exists(), "RESULT.json exists (create-exclusive)")
    torch.set_num_threads(6)
    st = torch.load(C.OUT / "STATS.pt", weights_only=False)
    res = compute(st, t_start, pv3_required=True)
    res.update({"lock_sha256": C.sha_file(C.LOCK), "stats_lock_sha256": C.sha_file(C.OUT / "STATS_LOCK.json"),
                "access_log_n": len(C.access_log()), "access_log": C.access_log()})
    h = C.write_json_x(C.OUT / "RESULT.json", res)
    print("VERDICT", res["verdict"], "RESULT", h[:16])
    print("primary pooled W", {k: round(v, 4) for k, v in res["summary"]["T/raw/n1"]["pooled_W"].items()})


def compute(st, t_start, pv3_required=True):
    S = Stats(st)
    # rebuild the pools from the stored paths (GPU phase order: T, then U, then replayed; each in cache row order)
    train, held = C.r15_and_heldout()
    rep = C.replay_lists()
    ever = {q for v in rep.values() for q in v}
    T = {c: [p for p in st["paths"][c] if p in set(train[c]) and p not in ever] for c in C.ORDER}
    U = {c: [p for p in st["paths"][c] if p not in set(train[c])] for c in C.U_CATS}
    for X, P in (("T", T), ("U", U)):
        for c, v in P.items():
            C.require(len(v) == C.POOL_SIZES[X][c], f"PV4 pool size {X}/{c}")
    pv = {"PV1_PV2": {"checkpoints": len(st["meta"]["checkpoints"]), "calm02": len(st["meta"]["calm02_train_sha256"])},
          "PV3": pv3(S, held, rep), "PV4": True}
    if pv3_required:
        C.require(pv["PV3"]["pass"], f"PV3 failed: max abs diff {pv['PV3']['max_abs_diff']}")
    u, bsize = covariates(S, T, rep)
    u_shuf = E.shuffle_covariates(u, np.arange(15))
    uidx = [C.A[c] for c in C.U_CATS]

    # PV6: prefix-only, on the primary configuration
    sr0 = series(S, "T", C.ORDER, T, "raw", 1, 0)
    e_ref, ph_ref, _ = E.learned(sr0["ya"], u, sr0["rs"], sr0["a"], 3)
    t0 = 7
    ya_p, u_p = sr0["ya"].copy(), u.copy()
    ya_p[:, t0 + 1:] += 1.0
    u_p[:, t0 + 1:, 1:] += 1.0
    e_pert, ph_pert, _ = E.learned(ya_p, u_p, sr0["rs"], sr0["a"], 3)
    pv["PV6_prefix_only"] = bool(np.array_equal(e_ref[:, :t0 + 1], e_pert[:, :t0 + 1], equal_nan=True)
                                 and all(np.array_equal(ph_ref[t], ph_pert[t]) for t in range(t0 + 1)))
    C.require(pv["PV6_prefix_only"], "PV6: learned filter is not prefix-only")

    # all configurations
    cells = {}                                                # (cohort, variant, n) -> arm -> {W, emu, el} [nc,15]
    stab = {"finite": True, "min_P": np.inf, "min_K": np.inf, "max_K": -np.inf, "max_grad_norm": 0.0,
            "n_fits": 0, "fit_s": 0.0}
    for variant in VARIANTS:
        for n in C.NW:
            accT = {arm: [] for arm in ARMS}
            accU = {arm: [] for arm in ARMS}
            for d in range(C.DRAWS):
                srT = series(S, "T", C.ORDER, T, variant, n, d)
                tf = time.time()
                estT, phis, diags = run_arms(srT, u, u_shuf)
                stab["fit_s"] += time.time() - tf
                for dg in diags.values():
                    stab["finite"] &= dg["finite"]
                    stab["min_P"] = min(stab["min_P"], dg["min_P"])
                    stab["min_K"] = min(stab["min_K"], dg["min_K"])
                    stab["max_K"] = max(stab["max_K"], dg["max_K"])
                    stab["n_fits"] += len(dg["fits"])
                    if dg["fits"]:
                        stab["max_grad_norm"] = max(stab["max_grad_norm"], max(f["grad_norm"] for f in dg["fits"].values()))
                for arm in ARMS:
                    accT[arm].append(errors(estT[arm], srT))
                srU = series(S, "U", C.U_CATS, U, variant, n, d)
                estU, _, _ = run_arms(srU, u[uidx], u_shuf[uidx], phis_in=phis)
                for arm in ARMS:
                    accU[arm].append(errors(estU[arm], srU))
            for X, acc in (("T", accT), ("U", accU)):
                cells[(X, variant, n)] = {arm: {"W": np.mean([x[0] for x in v], 0),
                                                "abs_emu": np.mean([np.abs(x[1]) for x in v], 0),
                                                "emu": np.mean([x[1] for x in v], 0),
                                                "abs_el": np.mean([np.abs(x[2]) for x in v], 0)}
                                          for arm, v in acc.items()}
            wall = time.time() - t_start
            print(f"[{time.strftime('%H:%M:%S')}] {variant} n={n} done, wall {wall:.0f}s", flush=True)
            C.require(wall <= C.CAPS["analysis_wall_h"] * 3600, "CAP: analysis wall")
            C.require(C.peak_rss_gb() <= C.CAPS["rss_gb"], "CAP: RSS")

    # ---------------- statistics on cohort T
    def summary(X, variant, n):
        cw = cells[(X, variant, n)]
        Wd = {arm: cw[arm]["W"] for arm in ARMS}
        keep, idx = boot_setup(Wd["E0"])
        bs = boot_stats(Wd, keep, idx)
        pm = {arm: pooled_mean(Wd[arm]) for arm in ARMS}
        macro = {arm: float(np.nanmean(np.nanmean(Wd[arm][keep], 1))) for arm in ARMS}
        pairs = {}
        for A_ in ARMS:
            for B_ in ARMS:
                if A_ < B_:
                    pairs[f"{A_}-{B_}"] = {"point": pm[A_] - pm[B_], "ci95": ci(bs[A_] - bs[B_])}
        return {"pooled_W": pm, "macro_W": macro, "ci95": {arm: ci(bs[arm]) for arm in ARMS}, "pairs": pairs,
                "abs_emu": {arm: pooled_mean(cw[arm]["abs_emu"]) for arm in ARMS},
                "abs_el": {arm: pooled_mean(cw[arm]["abs_el"]) for arm in ARMS},
                "n_transitions": int(np.isfinite(Wd["E0"]).sum()), "n_clusters": int(keep.sum())}, bs

    summ, boots = {}, {}
    for variant in VARIANTS:
        for n in C.NW:
            summ[f"T/{variant}/n{n}"], boots[("T", variant, n)] = summary("T", variant, n)
            summ[f"U/{variant}/n{n}"], _ = summary("U", variant, n)

    # ---------------- decision (prereg section 4)
    prim = summ[f"T/raw/n{C.NW_PRIMARY}"]
    bp = boots[("T", "raw", C.NW_PRIMARY)]
    dec = {"steps": []}
    W0 = prim["pooled_W"]["E0"]
    verdict = None
    if W0 < C.THRESH["drift_min_W"]:
        verdict = "FIXED_RULE_SUFFICIENT"
        dec["steps"].append(f"drift below threshold: W_E0 {W0:.4f} < {C.THRESH['drift_min_W']}")
    else:
        track = []
        for n in (1, 2):
            b = boots[("T", "raw", n)]
            sm = summ[f"T/raw/n{n}"]
            for f in ("E2", "E2A", "E3", "E4"):
                ok = (sm["pooled_W"][f] <= C.THRESH["track_ratio"] * sm["pooled_W"]["E0"]
                      and ci(b[f] - b["E0"])[1] < 0)
                track.append({"arm": f, "n": n, "ratio": sm["pooled_W"][f] / sm["pooled_W"]["E0"], "ok": bool(ok)})
        dec["trackability"] = track
        if not any(x["ok"] for x in track):
            verdict = "DRIFT_SIGNAL_NOT_IDENTIFIABLE"
        else:
            cond = {}
            for f in FIXED:
                cond[f"E4 <= 0.9*{f}"] = bool(prim["pooled_W"]["E4"] <= C.THRESH["learned_rel"] * prim["pooled_W"][f])
                cond[f"UB(E4-{f}) < 0"] = bool(ci(bp["E4"] - bp[f])[1] < 0)
            cond["UB(E4-E5) < 0"] = bool(ci(bp["E4"] - bp["E5"])[1] < 0)
            s2 = summ["T/raw/n2"]["pooled_W"]
            cond["n2: E4 <= best fixed"] = bool(s2["E4"] <= min(s2[f] for f in FIXED))
            cond["stability"] = bool(stab["finite"] and stab["min_P"] >= -1e-12 and stab["min_K"] >= 0
                                     and stab["max_K"] <= 1)
            dec["learned_conditions"] = cond
            verdict = "DRIFT_TRACKING_FEASIBLE" if all(cond.values()) else "FIXED_RULE_SUFFICIENT"
            if verdict == "FIXED_RULE_SUFFICIENT":
                beats_fixed = all(cond[k] for k in cond if k.startswith("E4 <=") or k.startswith("UB(E4-E2") or k.startswith("UB(E4-E3"))
                if beats_fixed and not cond["UB(E4-E5) < 0"]:
                    dec["steps"].append("E4 beats the fixed rules but not E5: learned constants only (tuned fixed rule)")
    dec["verdict"] = verdict

    # ---------------- PQ fidelity
    pq_mu, pq_l = [], []
    for c in C.ORDER:
        a = C.A[c]
        for p in st["pq_paths"][c]:
            r = st["raw"][c][S.ridx[c][p]]
            q = st["pq"][c][S.qidx[c][p]]
            for t in range(a, NB):
                _, sd_pool = S.pooled(c, T[c], t)
                mr, sr_ = pooled_from_rows(r[None, t])
                mq, sq = pooled_from_rows(q[None, t])
                pq_mu.append(abs(mq - mr) / sd_pool)
                pq_l.append(abs(np.log(sq) - np.log(sr_)))
    best_fixed = min(FIXED, key=lambda f: prim["pooled_W"][f])
    rel = summ[f"T/pq/n{C.NW_PRIMARY}"]["pooled_W"][best_fixed] / prim["pooled_W"][best_fixed] - 1
    pq = {"p95_abs_dmu_over_sd": float(np.percentile(pq_mu, 95)), "p95_abs_dlogsd": float(np.percentile(pq_l, 95)),
          "median_abs_dmu_over_sd": float(np.median(pq_mu)), "best_fixed_rule": best_fixed,
          "best_fixed_rel_change_raw_to_pq": float(rel)}
    pq["PQ_SUFFICIENT"] = bool(rel <= C.THRESH["pq_rel"] and pq["p95_abs_dmu_over_sd"] <= C.THRESH["pq_p95"]
                               and pq["p95_abs_dlogsd"] <= C.THRESH["pq_p95"])

    # ---------------- drift tables and the T/U estimand gap
    drift = {}
    for X, P in (("T", T), ("U", U)):
        for c, pool in P.items():
            a = C.A[c]
            mu_a, sd_a = S.pooled(c, pool, a)
            drift[f"{X}/{c}"] = {t: {"dmu_over_sd_a": (S.pooled(c, pool, t)[0] - mu_a) / sd_a,
                                     "dlog_sd": float(np.log(S.pooled(c, pool, t)[1] / sd_a))} for t in range(a, NB)}
    gap = {}
    for c in C.U_CATS:
        a = C.A[c]
        gap[c] = {}
        for t in range(a, NB):
            (mt, stt), (mu_, su) = S.pooled(c, T[c], t), S.pooled(c, U[c], t)
            gap[c][t] = {"W2_T_vs_U_over_sdU": float(np.hypot(mt - mu_, stt - su) / su),
                         "drift_T": drift[f"T/{c}"][t]["dmu_over_sd_a"], "drift_U": drift[f"U/{c}"][t]["dmu_over_sd_a"]}
    dT = [gap[c][t]["drift_T"] for c in gap for t in gap[c] if t > C.A[c]]
    dU = [gap[c][t]["drift_U"] for c in gap for t in gap[c] if t > C.A[c]]
    gap_summary = {"corr_drift_T_U": float(np.corrcoef(dT, dU)[0, 1]),
                   "mean_drift_T": float(np.mean(dT)), "mean_drift_U": float(np.mean(dU)),
                   "mean_W2_gap_post_acq": float(np.mean([gap[c][t]["W2_T_vs_U_over_sdU"] for c in gap for t in gap[c]
                                                          if t > C.A[c]]))}

    # ---------------- per category / per boundary tables (primary)
    cw = cells[("T", "raw", C.NW_PRIMARY)]
    per_cat = {c: {arm: (float(np.nanmean(cw[arm]["W"][i])) if np.isfinite(cw[arm]["W"][i]).any() else None)
                   for arm in ARMS} for i, c in enumerate(C.ORDER)}
    per_bnd = {t: {arm: (float(np.nanmean(cw[arm]["W"][:, t])) if np.isfinite(cw[arm]["W"][:, t]).any() else None)
                   for arm in ARMS} for t in range(1, NB)}

    # ---------------- costs and bytes (prereg section 7)
    spg = st["meta"]["s_per_grid"]
    n_fit_calls = stab["n_fits"]
    cost = {"gpu_s_per_grid": spg, "phase_gpu_forward_s": st["meta"]["gpu_forward_s"],
            "phase_wall_s": st["meta"]["wall_s"], "n_grids": st["meta"]["n_grids"],
            "learned_fit_s_mean": stab["fit_s"] / max(n_fit_calls, 1), "analysis_fit_s_total": stab["fit_s"]}
    per_boundary = {}
    for n in C.NW:
        g = {"E0": 0, "E2": n, "E2A": n, "E3": n, "E4_0": n,
             "E4": n + 2 * float(bsize[bsize > 0].mean()), "E5": n + 2 * float(bsize[bsize > 0].mean())}
        per_boundary[n] = {arm: {"grids_per_old_category": g[arm], "gpu_ms_per_old_category": 1000 * g[arm] * spg}
                           for arm in ARMS}
    cost["per_boundary"] = per_boundary
    hist_trans = 105
    ledger = {}
    for n in C.NW:
        for rep_name, gb in (("fp32", C.BYTES["fp32_grid"]), ("fp16", C.BYTES["fp16_grid"]), ("pq", C.BYTES["pq_grid"])):
            wit = n * gb
            per = {"E0": 16, "E2": wit, "E2A": wit + 32, "E3": wit + 32 + 16,
                   "E4_0": wit + 32 + 32 + 16, "E4": wit + 32 + 32 + 16 + C.BYTES["centroid_fp32"] + 8,
                   "E5": wit + 32 + 32 + 16 + C.BYTES["centroid_fp32"] + 8}
            glob = {"E0": 0, "E2": 0, "E2A": 0, "E3": 0, "E4_0": 48 + 16 * hist_trans, "E4": 80 + 40 * hist_trans,
                    "E5": 80 + 40 * hist_trans}
            ledger[f"n{n}/{rep_name}"] = {arm: {"per_category_B": per[arm], "total_15_categories_B": 15 * per[arm] + glob[arm]}
                                          for arm in ARMS}

    res = {"schema": "nl_drift_00_result_v1", "verdict": verdict, "decision": dec, "provenance": pv,
           "primary": {"cohort": "T", "variant": "raw", "n": C.NW_PRIMARY}, "summary": summ,
           "per_category_primary": per_cat, "per_boundary_primary": per_bnd, "pq_fidelity": pq,
           "drift": drift, "estimand_gap_T_vs_U": gap, "estimand_gap_summary": gap_summary,
           "stability": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in stab.items()},
           "cost": cost, "bytes": ledger,
           "covariates": {"u_buf_mean_abs": float(np.nanmean(np.abs(u[..., 1]))),
                          "u_aff_range": [float(np.nanmin(u[..., 2])), float(np.nanmax(u[..., 2]))],
                          "buffer_size_range": [int(bsize[bsize > 0].min()), int(bsize.max())]},
           "analysis_wall_s": time.time() - t_start, "rss_peak_gb": C.peak_rss_gb(), "utc": C.now()}
    return res


if __name__ == "__main__":
    main()
