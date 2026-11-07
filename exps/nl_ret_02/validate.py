"""NL-RET-02 G3-G5 (CPU). Implements SIGNAL_DICTIONARY.json exactly with the NL-RET-01A estimator code
(fold-internal imputation/standardisation, inner leave-one-category-out alpha). Y4 is read only in `align`, after
`validate` has frozen the predictions (G4_FREEZE.json).

  validate  -> INFORMATION_ABLATIONS, CROSS_CLASS_VALIDATION, CROSS_SEED_VALIDATION, TEMPORAL_VALIDATION,
               FSB_REPLICATION, PER_CATEGORY_TABLE.csv, DESCRIPTIVE_SIGNALS.json, FROZEN_RISK_PREDICTIONS.csv, G4_FREEZE
  align     -> REAL_AP_ALIGNMENT_DESCRIPTIVE.json (POSTHOC_EXPOSED_REAL_AP_DIAGNOSTIC)
  decide    -> FINAL_VERDICT.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from exps.cadfinal import common as K
from exps.dinomaly.common import require, sha
from exps.nl_ret_01a import validate as V
from exps.nl_ret_02 import common as C
from exps.nl_ret_02 import prereg as R

TARGETS = ["Y1_primary", "Y2", "Y3"]
SCHEMES = ("LOCO", "LOSO", "temporal")
REPS, LO, HI = 2000, 1.25, 98.75
GROUPS = V.GROUPS
I1 = list(dict.fromkeys(R.I0 + V.LAWFUL + ["THR"]))
GN_ROWCOLS = list(R.GN_ROW)


def uniq(*parts):
    return list(dict.fromkeys(c for p in parts for c in p))


def sets():
    S = {"I0": R.I0, "I1": I1, "I2": uniq(I1, R.GE), "I3": uniq(I1, R.GN), "I4": uniq(I1, R.GE, R.GN),
         "I5": uniq(I1, R.GE, R.GN, R.FN), "I0+GE": uniq(R.I0, R.GE), "I0+GN": uniq(R.I0, R.GN),
         "I0+GE+GN": uniq(R.I0, R.GE, R.GN), "I0+GE+GN+FN": uniq(R.I0, R.GE, R.GN, R.FN)}
    return S


def load():
    F = pd.read_csv(C.OUT / "FEATURE_TABLE.csv")
    O = pd.read_csv(C.OUT / "OUTCOME_TABLE.csv")
    require("Y4_exposed" not in O, "Y4 must not be in the outcome table")
    D = F.merge(O, on=["arm", "seed", "category", "s"], validate="1:1")
    D["THR"] = (D["A_grow_prev"].fillna(-1) > 0).astype(float)
    D["INV_N"] = 1.0 / D["D_n_members"]
    D["AGE_BUCKET"] = D["K_age"].clip(upper=3)
    for k in range(4):
        D[f"AB{k}"] = (D["AGE_BUCKET"] == k).astype(float)
    return D


def controls(D, rng):
    """Label-free whole-table permutations (fixed RNG)."""
    gc = D.copy()
    for _, idx in gc.groupby(["arm", "seed", "s"]).groups.items():
        idx = list(idx)
        if len(idx) > 1:
            gc.loc[idx, GN_ROWCOLS] = gc.loc[rng.permutation(idx), GN_ROWCOLS].to_numpy()
    hist = D.copy()
    G = [c for c in V.LAWFUL if c.startswith("G_")]
    for _, idx in hist.groupby(["arm", "seed", "category"]).groups.items():
        idx = list(idx)
        hist.loc[idx, G] = hist.loc[rng.permutation(idx), G].to_numpy()
    return gc, hist


def fit_predict(D, tr, te, spec, y, rng):
    if spec[0] == "shuf_inc":                     # permute the incoming block inside the training fold only
        cols, inc = spec[1], spec[2]
        D2 = D.copy()
        for _, g in D.loc[tr].groupby("AGE_BUCKET").groups.items():
            g = list(g)
            D2.loc[g, inc] = D.loc[rng.permutation(g), inc].to_numpy()
        return V.fit_predict(D2, tr, te, ("ridge", cols), y)
    return V.fit_predict(D, tr, te, spec, y)


def oof(D, spec, y, scheme, rng):
    pred = pd.Series(np.nan, index=D.index)
    thr = pd.Series(np.nan, index=D.index)
    for tr, te, _ in V.folds(D, scheme):
        tr = tr[~D.loc[tr, y].isna()]
        pred.loc[te] = fit_predict(D, tr, te, spec, y, rng)
        ptr = fit_predict(D, tr, tr, spec, y, rng) if spec[0] != "oracle" else D.loc[tr, y].to_numpy()
        thr.loc[te] = np.quantile(ptr, 0.8)
    return pred, thr


def boot(D, pa, pb, y, rng, reps=REPS):
    """Category-clustered paired bootstrap of MAE(a)-MAE(b) and Spearman(a)-Spearman(b)."""
    ok = ~D[y].isna() & ~pa.isna() & ~pb.isna()
    t, a, b = D.loc[ok, y].to_numpy(), pa[ok].to_numpy(), pb[ok].to_numpy()
    cat = D.loc[ok, "category"].to_numpy()
    cs = sorted(set(cat))
    rows = {c: np.where(cat == c)[0] for c in cs}
    dm, dr = [], []
    for _ in range(reps):
        i = np.concatenate([rows[c] for c in rng.choice(cs, len(cs))])
        dm.append(np.abs(a[i] - t[i]).mean() - np.abs(b[i] - t[i]).mean())
        dr.append(spearmanr(a[i], t[i]).statistic - spearmanr(b[i], t[i]).statistic)
    return {"mae_diff": float(np.abs(a - t).mean() - np.abs(b - t).mean()),
            "mae_ci98.75": np.percentile(dm, [LO, HI]).tolist(), "mae_ci95": np.percentile(dm, [2.5, 97.5]).tolist(),
            "spearman_diff": float(spearmanr(a, t).statistic - spearmanr(b, t).statistic),
            "spearman_ci98.75": np.nanpercentile(dr, [LO, HI]).tolist(),
            "spearman_ci95": np.nanpercentile(dr, [2.5, 97.5]).tolist()}


def specs(D):
    S01 = V.specs()
    S = {k: S01[k] for k in V.LAWFUL_BASE}
    S["O1_oracle_persistence"] = S01["O1_oracle_persistence"]
    for k, cols in sets().items():
        S[k] = ("ridge", cols)
    S["CTRL_shuffle_incoming"] = ("shuf_inc", sets()["I4"], uniq(R.GE, R.GN))
    return S


BASE = V.LAWFUL_BASE + ["I0"]


def evaluate(D, Dgc, Dh, y, scheme, rng):
    S = specs(D)
    P, M = {}, {}
    for k, sp in S.items():
        if sp[0] == "oracle" and y != "Y1_primary":
            continue
        P[k], thr = oof(D, sp, y, scheme, rng)
        M[k] = V.metrics(D, P[k], thr, y)
    for k, Dx in (("CTRL_shuffle_grad_to_category", Dgc), ("CTRL_shuffle_history_order", Dh)):
        P[k], thr = oof(Dx, ("ridge", sets()["I4"]), y, scheme, rng)
        M[k] = V.metrics(D, P[k], thr, y)
    best = min(BASE, key=lambda k: M[k]["MAE"])
    return {"best_fixed": best, "models": M}, P


def fsb_replication(D, Dfs, y, rng, rows_mask):
    S = specs(D)
    Dall = pd.concat([D.assign(_src="ID"), Dfs.assign(_src="FS")], ignore_index=True)
    out, P = {}, {}
    for k, sp in S.items():
        if sp[0] in ("oracle", "shuf_inc"):
            continue
        pred = pd.Series(np.nan, index=Dall.index)
        for c in C.STREAM:
            tr = Dall.index[(Dall._src == "ID") & (Dall.category != c) & ~Dall[y].isna()]
            te = Dall.index[(Dall._src == "FS") & (Dall.category == c)]
            if len(te):
                pred.loc[te] = fit_predict(Dall, tr, te, sp, y, rng)
        P[k] = pred[Dall._src == "FS"].reset_index(drop=True)
    Fs = Dall[Dall._src == "FS"].reset_index(drop=True)
    for name, m in rows_mask.items():
        sel = m(Fs)
        r = {}
        for k in P:
            ok = sel & ~Fs[y].isna()
            e = (P[k][ok] - Fs.loc[ok, y]).abs()
            r[k] = {"n": int(ok.sum()), "MAE": float(e.mean()),
                    "spearman": float(spearmanr(P[k][ok], Fs.loc[ok, y]).statistic) if P[k][ok].std() > 0 else 0.0,
                    "per_seed_MAE": {int(sd): float(e[Fs.loc[ok, "seed"] == sd].mean()) for sd in C.SEEDS}}
        best = min(BASE, key=lambda k: r[k]["MAE"])
        out[name] = {"best_fixed": best, "models": r}
        for p in R.PRIMARY + ["I1"]:
            ok = sel & ~Fs[y].isna()
            out[name]["models"][p]["vs_best_fixed"] = boot(Fs[ok].reset_index(drop=True), P[p][ok].reset_index(drop=True),
                                                           P[best][ok].reset_index(drop=True), y, rng)
    return out, P, Fs


def descriptive(D, Dfs, P_b5):
    """Per new feature: Spearman with Y1/Y2/Y3, partial Spearman given age + occupancy, Spearman with the B5 OOF
    residual, seed signs, FS_B (s >= 7) replication rho, missingness."""
    out = {}
    res = D["Y1_primary"] - P_b5
    x = D[["D_n_members", "K_age"]].to_numpy(float)
    Xd = np.c_[np.ones(len(x)), x]
    def resid(v):
        ok = ~np.isnan(v)
        b = np.linalg.lstsq(Xd[ok], v[ok], rcond=None)[0]
        r = np.full(len(v), np.nan); r[ok] = v[ok] - Xd[ok] @ b; return r
    ry = resid(D["Y1_primary"].to_numpy(float))
    fs7 = Dfs[Dfs.s >= 7]
    for f in R.GN + R.GE + R.FN:
        v = D[f].to_numpy(float)
        ok = ~np.isnan(v)
        r = {f"rho_{y}": float(spearmanr(v[ok & ~D[y].isna().to_numpy()], D.loc[ok & ~D[y].isna().to_numpy(), y]).statistic)
             for y in TARGETS}
        r["partial_rho_Y1_given_age_occupancy"] = float(spearmanr(resid(v)[ok], ry[ok]).statistic)
        r["rho_Y1_residual_of_B5"] = float(spearmanr(v[ok], res[ok]).statistic)
        r["seed_signs_Y1"] = [float(np.sign(spearmanr(D.loc[(D.seed == sd) & ok, f],
                                                      D.loc[(D.seed == sd) & ok, "Y1_primary"]).statistic)) for sd in C.SEEDS]
        r["seed_signs_partial"] = [float(np.sign(spearmanr(resid(v)[(D.seed == sd).to_numpy() & ok],
                                                           ry[(D.seed == sd).to_numpy() & ok]).statistic)) for sd in C.SEEDS]
        okf = fs7[f].notna()
        r["rho_Y1_FSB_s>=7"] = float(spearmanr(fs7.loc[okf, f], fs7.loc[okf, "Y1_primary"]).statistic) \
            if fs7.loc[okf, f].std() > 0 else np.nan
        r["missing_frac"] = float(np.isnan(v).mean())
        out[f] = r
    fsq = [c for c in Dfs if c.startswith("FSQ_")]
    fq = {}
    for f in fsq:
        d = Dfs[Dfs[f].notna()]
        fq[f] = {y: float(spearmanr(d[f], d[y]).statistic) if d[f].std() > 0 and d[y].notna().all() else None
                 for y in ("Y1_primary", "Y3")}
    return {"ID_features": out, "FSB_Q2_descriptive": fq,
            "label": "DESCRIPTIVE (not a gate input); FS_B Q2 features are FS_B-only"}


def run_validate():
    require(not (C.OUT / "CROSS_CLASS_VALIDATION.json").exists(), "one-shot")
    rng = np.random.default_rng(20261107)
    V.RNG = np.random.default_rng(20261108)
    Dfull = load()
    D = Dfull[Dfull.arm == "ID"].reset_index(drop=True)
    Dfs = Dfull[Dfull.arm == "FS_B"].reset_index(drop=True)
    Dgc, Dh = controls(D, np.random.default_rng(20261109))
    res, keep = {}, {}
    for y in TARGETS:
        for sc in SCHEMES:
            r, P = evaluate(D, Dgc, Dh, y, sc, rng)
            res[(y, sc)] = r
            keep[(y, sc)] = P
            m = r["models"]
            print(y, sc, "best", r["best_fixed"], round(m[r["best_fixed"]]["MAE"], 5),
                  {k: round(m[k]["MAE"], 5) for k in sets()}, flush=True)
    # paired clustered comparisons (LOCO), every target
    for y in TARGETS:
        r, P = res[(y, "LOCO")], keep[(y, "LOCO")]
        for k in r["models"]:
            if k in BASE or k.startswith("O1"):
                continue
            r["models"][k]["vs_best_fixed"] = boot(D, P[k], P[r["best_fixed"]], y, rng)
            if k != "I1":
                r["models"][k]["vs_I1"] = boot(D, P[k], P["I1"], y, rng)
    # FS_B replication (ID-trained, LOCO folds)
    rep = {}
    masks = {"FSB_state_distinct_s>=7": lambda F: F.s >= 7, "FSB_shared_theta6_s=6": lambda F: F.s == 6}
    for y in TARGETS:
        rep[y], Pfs, Fs = fsb_replication(D, Dfs, y, rng, masks)
    # gate (Y1, preregistered)
    loco, loso, tmp = (res[("Y1_primary", s)] for s in SCHEMES)
    bf = loco["best_fixed"]
    gate = {}
    for p in R.PRIMARY:
        m = loco["models"][p]
        i = m["MAE"] <= 0.95 * loco["models"][bf]["MAE"]
        ii = m["vs_best_fixed"]["mae_ci98.75"][1] < 0
        iii = m["vs_I1"]["mae_ci98.75"][1] < 0
        lb = loso["best_fixed"]
        iv = sum(loso["models"][p]["per_seed_MAE"][sd] < loso["models"][lb]["per_seed_MAE"][sd] for sd in C.SEEDS)
        v = tmp["models"][p]["MAE"] <= 1.02 * tmp["models"][tmp["best_fixed"]]["MAE"]
        r7 = rep["Y1_primary"]["FSB_state_distinct_s>=7"]
        vi = r7["models"][p]["MAE"] < r7["models"][r7["best_fixed"]]["MAE"]
        rank = (m["vs_best_fixed"]["spearman_ci98.75"][0] > 0 and m["vs_I1"]["spearman_ci98.75"][0] > 0)
        under = (i and iv >= 2 and v and vi and m["vs_best_fixed"]["mae_diff"] < 0 and m["vs_I1"]["mae_diff"] < 0
                 and not (ii and iii))
        gate[p] = {"i_mae_le_0.95_best_fixed": bool(i), "ii_ci_vs_best_fixed_lt0": bool(ii), "iii_ci_vs_I1_lt0": bool(iii),
                   "iv_loso_seeds_beating": int(iv), "v_temporal_ok": bool(v), "vi_fsb_s>=7_ok": bool(vi),
                   "PASS": bool(i and ii and iii and iv >= 2 and v and vi), "rank_only_condition": bool(rank),
                   "variance_underpowered_condition": bool(under),
                   "LOCO_MAE": m["MAE"], "best_fixed_MAE": loco["models"][bf]["MAE"], "I1_MAE": loco["models"]["I1"]["MAE"],
                   "categories_beating_best_fixed": int(sum(m["per_category_MAE"][c] < loco["models"][bf]["per_category_MAE"][c]
                                                            for c in m["per_category_MAE"]))}
    n_events = loco["models"]["B0_constant"].get("n_events", 0)
    # information ablation table
    abl = {}
    for y in TARGETS:
        abl[y] = {}
        for sc in SCHEMES:
            mm = res[(y, sc)]["models"]
            abl[y][sc] = {"best_fixed": res[(y, sc)]["best_fixed"], "best_fixed_MAE": mm[res[(y, sc)]["best_fixed"]]["MAE"],
                          **{k: {"MAE": mm[k]["MAE"], "spearman": mm[k]["spearman"]} for k in
                             list(sets()) + ["CTRL_shuffle_incoming", "CTRL_shuffle_grad_to_category",
                                             "CTRL_shuffle_history_order", "B1_clock", "B2_occupancy", "B5_fixed_schedule"]},
                          "Delta_incoming_MAE_I1_minus_I4": mm["I1"]["MAE"] - mm["I4"]["MAE"]}
        abl[y]["LOCO_Delta_incoming_bootstrap_I4_vs_I1"] = res[(y, "LOCO")]["models"]["I4"]["vs_I1"]
    # descriptive + per-category
    desc = descriptive(D, Dfs, keep[("Y1_primary", "LOCO")]["B5_fixed_schedule"])
    sig = {c: g for g, cs in GROUPS.items() for c in cs}
    pc = []
    PL = keep[("Y1_primary", "LOCO")]
    for c in C.STREAM:
        d = D[D.category == c]
        r = {"category": c, "a": C.a_of(c), "nl_ret_00b_group": sig[c], "transitions_per_seed": int(len(d) / 3)}
        if not len(d):
            r["note"] = "acquired at stage 15: no eligible transition"
            pc.append(r)
            continue
        e = {k: (PL[k][d.index] - d.Y1_primary).abs() for k in (bf, "I1") + tuple(R.PRIMARY)}
        r.update({"occupancy_mean": float(d.D_n_members.mean()), "occupancy_min": int(d.D_n_members.min()),
                  "Y1_mean": float(d.Y1_primary.mean()), "GN_cos_mean": float(d.GN_cos.mean()),
                  "GN_fo_sign_mean": float(d.GN_fo_sign.mean()), "GN_blk_negfrac_mean": float(d.GN_blk_negfrac.mean()),
                  "GE_nn_i2c_mean": float(d.GE_nn_i2c.mean()), "GE_patch_cov_mean": float(d.GE_patch_cov.mean()),
                  "GE_rank_mean": float(d.GE_rank.mean()),
                  f"MAE_{bf}": float(e[bf].mean()), "MAE_I1": float(e["I1"].mean()),
                  **{f"MAE_{p}": float(e[p].mean()) for p in R.PRIMARY},
                  **{f"improvement_{p}_vs_best_fixed": float(e[bf].mean() - e[p].mean()) for p in R.PRIMARY},
                  **{f"seeds_improved_{p}": int(sum(e[bf][d.seed == sd].mean() > e[p][d.seed == sd].mean()
                                                     for sd in C.SEEDS)) for p in R.PRIMARY},
                  "rho_GN_cos_Y1": float(spearmanr(d.GN_cos, d.Y1_primary).statistic) if len(d) >= 6 else np.nan,
                  "rho_GE_nn_i2c_Y1": float(spearmanr(d.GE_nn_i2c, d.Y1_primary).statistic) if len(d) >= 6 else np.nan,
                  "missing_GE_cov_overlap": float(d.GE_cov_overlap.isna().mean()),
                  "n_rows": int(len(d)), "uncertainty": "LOW (<= 6 rows/seed)" if len(d) <= 18 else "moderate"})
        pc.append(r)
    pd.DataFrame(pc).to_csv(C.OUT / "PER_CATEGORY_TABLE.csv", index=False)
    # single-feature LOCO ranking (descriptive)
    singles = {}
    for f in R.GN + R.GE + R.FN:
        p, thr = oof(D, ("ridge", [f]), "Y1_primary", "LOCO", rng)
        mm = V.metrics(D, p, thr, "Y1_primary")
        singles[f] = {"MAE": mm["MAE"], "spearman": mm["spearman"]}
    V.W(C.OUT / "CROSS_CLASS_VALIDATION.json", {y: res[(y, "LOCO")] for y in TARGETS} | {
        "gate_Y1": gate, "n_events_Y1": n_events, "single_feature_LOCO_Y1": singles})
    V.W(C.OUT / "CROSS_SEED_VALIDATION.json", {y: res[(y, "LOSO")] for y in TARGETS})
    V.W(C.OUT / "TEMPORAL_VALIDATION.json", {y: res[(y, "temporal")] for y in TARGETS})
    V.W(C.OUT / "FSB_REPLICATION.json", rep)
    V.W(C.OUT / "INFORMATION_ABLATIONS.json", abl)
    V.W(C.OUT / "DESCRIPTIVE_SIGNALS.json", desc)
    # freeze: predictions for alignment = the primary set with the lower LOCO MAE (normal-only choice)
    chosen = min(R.PRIMARY, key=lambda p: loco["models"][p]["MAE"])
    D.assign(pred=PL[chosen], pred_other_primary=PL[[p for p in R.PRIMARY if p != chosen][0]],
             pred_best_fixed=PL[bf], pred_I1=PL["I1"])[["arm", "seed", "category", "s", "pred", "pred_other_primary",
                                                       "pred_best_fixed", "pred_I1"]].to_csv(
        C.OUT / "FROZEN_RISK_PREDICTIONS.csv", index=False)
    V.W(C.OUT / "G4_FREEZE.json", {"chosen_primary_for_alignment": chosen, "best_fixed_LOCO": bf, "gate_Y1": gate,
                                   "n_events": n_events, "missing_primary": int(D.Y1_primary.isna().sum()),
                                   "predictions_sha256": sha(C.OUT / "FROZEN_RISK_PREDICTIONS.csv"), "utc": K.now()})
    print("validate done; gate", {p: (g["PASS"], g["rank_only_condition"], g["variance_underpowered_condition"])
                                  for p, g in gate.items()})


def run_align():
    """POSTHOC_EXPOSED_REAL_AP_DIAGNOSTIC: Y4 read only now; cannot change G4."""
    fr = K.read(C.OUT / "G4_FREEZE.json")
    require(sha(C.OUT / "FROZEN_RISK_PREDICTIONS.csv") == fr["predictions_sha256"], "frozen predictions changed")
    require(not (C.OUT / "REAL_AP_ALIGNMENT_DESCRIPTIVE.json").exists(), "one-shot")
    P = pd.read_csv(C.OUT / "FROZEN_RISK_PREDICTIONS.csv")
    O1 = pd.read_csv(C.OUT01 / "OUTCOME_TABLE.csv")
    D = P.merge(O1[O1.arm == "ID"][["seed", "category", "s", "Y4_exposed", "Y1_primary"]], on=["seed", "category", "s"],
                validate="1:1")

    def al(d, col):
        d = d.dropna(subset=["Y4_exposed", col])
        if len(d) < 6 or d[col].std() == 0:
            return {"n": len(d), "spearman": None}
        return {"n": len(d), "spearman": float(spearmanr(d[col], d["Y4_exposed"]).statistic)}

    cols = {"chosen_primary": "pred", "other_primary": "pred_other_primary", "best_fixed": "pred_best_fixed",
            "I1": "pred_I1", "actual_Y1": "Y1_primary"}
    out = {"label": "POSTHOC_EXPOSED_REAL_AP_DIAGNOSTIC", "chosen_primary": fr["chosen_primary_for_alignment"],
           "overall": {k: al(D, c) for k, c in cols.items()},
           "groups": {g: {k: al(D[D.category.isin(cs)], c) for k, c in cols.items()} for g, cs in GROUPS.items()},
           "per_category": {c: al(D[D.category == c], "pred") for c in C.STREAM},
           "per_seed": {int(sd): al(D[D.seed == sd], "pred") for sd in C.SEEDS}}
    sev = D.Y4_exposed >= 0.05
    stable = D.category.isin(GROUPS["stable_or_improving"])
    for k, c in cols.items():
        hi = D[c] > D[c].quantile(0.8)
        out.setdefault("severe_capture_top20pct", {})[k] = float(hi[sev].mean()) if sev.any() else None
        out.setdefault("false_alarm_stable_top20pct", {})[k] = float(hi[stable].mean())
    out["severe_transitions_Y4_ge_0.05"] = int(sev.sum())
    V.W(C.OUT / "REAL_AP_ALIGNMENT_DESCRIPTIVE.json", out)
    print("align", {k: v["spearman"] for k, v in out["overall"].items()})


def run_decide():
    fr = K.read(C.OUT / "G4_FREEZE.json")
    al = K.read(C.OUT / "REAL_AP_ALIGNMENT_DESCRIPTIVE.json")
    cov = K.read(C.OUT / "BOUNDARY_COVERAGE.json")
    rep = K.read(C.OUT / "BASELINE_REPRODUCTION.json")
    seal = K.read(C.OUT / "TABLES_SEAL.json")
    g = fr["gate_Y1"]
    if cov["bad_checkpoints"] or cov["replay_record_mismatches"] or not rep["pass"]:
        label = "INTEGRITY_BLOCKED"
    elif fr["missing_primary"] or fr["n_events"] < 15 or seal["rows"]["ID"] != 315:
        label = "UNDERPOWERED_OR_INCOMPLETE"
    else:
        passing = [p for p, v in g.items() if v["PASS"]]
        if passing:
            rho = al["overall"]["chosen_primary"]["spearman"]
            label = "INCOMING_TASK_SIGNAL_SUPPORTED" if rho is not None and rho > 0.10 else "NORMAL_PREDICTIVE_BUT_AP_UNALIGNED"
        elif any(v["variance_underpowered_condition"] for v in g.values()):
            label = "UNDERPOWERED_OR_INCOMPLETE"
        elif any(v["rank_only_condition"] for v in g.values()):
            label = "INCOMING_SIGNAL_RANK_ONLY"
        else:
            label = "NO_INCREMENTAL_INCOMING_SIGNAL"
    V.W(C.OUT / "FINAL_VERDICT.json", {"label": label, "gate_Y1": g, "rule": R.GATE,
                                       "alignment_overall": al["overall"], "utc": K.now()})
    print("verdict", label)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "align", "decide"])
    {"validate": run_validate, "align": run_align, "decide": run_decide}[ap.parse_args().cmd]()
