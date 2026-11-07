"""NL-RET-01A Phases 3-6: descriptive discovery, out-of-fold validation, incremental information, posthoc real-AP
alignment, decision. Implements SIGNAL_DICTIONARY.json exactly; Y4 is read only in `align` (after `validate`)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from exps.cadfinal import common as K
from exps.dinomaly.common import require, sha
from exps.nl_ret_01a import common as C
from exps.nl_ret_01a.prereg import BASELINES, GATE, SIGNALS

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
TARGETS = ["Y1_primary", "Y2", "Y3"]
EVENT = 0.05
FAM = {k: list(v) for k, v in SIGNALS.items() if k != "CLOCK"}
FAM["F_optimization"] = [f for f in FAM["F_optimization"] if f not in ("F_aux_eff", "F_clip_sat")]  # arm-only
CLOCK = list(SIGNALS["CLOCK"])
CONTROLS = CLOCK + ["D_n_members", "A_pix_q99", "A_img_mean", "THR"]
LAWFUL = sorted({f for v in FAM.values() for f in v} | set(CLOCK))
GROUPS = {"within_image": ["screw", "cable", "capsule", "grid"],
          "calibration": ["transistor", "metal_nut", "pill", "hazelnut"],
          "stable_or_improving": ["bottle", "carpet", "leather", "tile", "toothbrush", "wood", "zipper"]}
RNG = np.random.default_rng(20261106)


def _san(o):
    if isinstance(o, dict):
        return {str(k): _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def W(path, obj):
    K.write_json(path, _san(obj))


def load(arm="ID"):
    F = pd.read_csv(C.OUT / "FEATURE_TABLE.csv")
    O = pd.read_csv(C.OUT / "OUTCOME_TABLE.csv")
    D = F.merge(O.drop(columns=["a", "shared_with_ID"]), on=["arm", "seed", "category", "s"], validate="1:1")
    D["THR"] = (D["A_grow_prev"].fillna(-1) > 0).astype(float)
    D["INV_N"] = 1.0 / D["D_n_members"]
    D["AGE_BUCKET"] = D["K_age"].clip(upper=3)
    if arm == "ID":
        return D[D.arm == "ID"].reset_index(drop=True)
    return D[(D.arm == "FS_B") & (~D.shared_with_ID)].reset_index(drop=True)


# ---- ridge with fold-internal preprocessing ------------------------------------------------------------------

def prep(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    miss = np.isnan(Xtr).any(0) | np.isnan(Xte).any(0)
    def tf(X):
        Z = np.where(np.isnan(X), med, X)
        return np.hstack([Z, np.isnan(X[:, miss]).astype(float)])
    A, B = tf(Xtr), tf(Xte)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def ridge(Xtr, ytr, Xte, alpha):
    Ztr, Zte = prep(Xtr, Xte)
    ym = ytr.mean()
    p = Ztr.shape[1]
    w = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(p), Ztr.T @ (ytr - ym))
    return Zte @ w + ym


def fit_predict(D, tr, te, spec, y):
    """spec: ('ridge', cols) | ('const',) | ('thr',) | ('sched',) | ('shuffle', cols) | ('oracle',)."""
    ytr = D.loc[tr, y].to_numpy()
    kind = spec[0]
    if kind == "const":
        return np.full(len(te), ytr.mean())
    if kind == "thr":
        g = D.loc[tr, "THR"].to_numpy(); m = {k: ytr[g == k].mean() if (g == k).any() else ytr.mean() for k in (0, 1)}
        return np.array([m[k] for k in D.loc[te, "THR"]])
    if kind == "sched":
        g = D.loc[tr, "AGE_BUCKET"].to_numpy()
        m = {k: ytr[g == k].mean() if (g == k).any() else ytr.mean() for k in range(4)}
        return np.array([m[int(k)] for k in D.loc[te, "AGE_BUCKET"]])
    if kind == "oracle":
        v = D.loc[te, "ORACLE_prev_Y1" if y == "Y1_primary" else f"ORACLE_prev_{y}"].to_numpy()
        return np.where(np.isnan(v), ytr.mean(), v)
    cols = spec[1]
    Xtr, Xte = D.loc[tr, cols].to_numpy(float), D.loc[te, cols].to_numpy(float)
    if kind == "shuffle":
        Xtr = Xtr[RNG.permutation(len(Xtr))]
    # inner leave-one-category-out alpha selection inside the training fold
    cats = D.loc[tr, "category"].to_numpy()
    best, best_a = np.inf, ALPHAS[0]
    if len(set(cats)) >= 3:
        for a in ALPHAS:
            err = []
            for c in sorted(set(cats)):
                itr, ite = cats != c, cats == c
                err.append(np.abs(ridge(Xtr[itr], ytr[itr], Xtr[ite], a) - ytr[ite]))
            e = np.mean(np.concatenate(err))
            if e < best:
                best, best_a = e, a
    return ridge(Xtr, ytr, Xte, best_a)


def folds(D, scheme):
    if scheme == "LOCO":
        return [(D.index[D.category != c], D.index[D.category == c], c) for c in C.STREAM if (D.category == c).any()]
    if scheme == "LOSO":
        return [(D.index[D.seed != r], D.index[D.seed == r], r) for r in C.SEEDS]
    if scheme == "temporal":
        return [(D.index[D.s + 1 <= t], D.index[D.s == t], t) for t in range(4, 15)
                if (D.s + 1 <= t).sum() >= 20 and (D.s == t).any()]
    raise ValueError(scheme)


def oof(D, spec, y, scheme):
    pred = pd.Series(np.nan, index=D.index)
    thr = pd.Series(np.nan, index=D.index)
    for tr, te, _ in folds(D, scheme):
        tr = tr[~D.loc[tr, y].isna()]
        pred.loc[te] = fit_predict(D, tr, te, spec, y)
        ptr = fit_predict(D, tr, tr, spec, y) if spec[0] not in ("oracle",) else D.loc[tr, y].to_numpy()
        thr.loc[te] = np.quantile(ptr, 0.8)
    return pred, thr


def metrics(D, pred, thr, y):
    ok = ~D[y].isna() & ~pred.isna()
    t, p = D.loc[ok, y].to_numpy(), pred[ok].to_numpy()
    e = np.abs(p - t)
    out = {"n": int(ok.sum()), "MAE": float(e.mean()), "RMSE": float(np.sqrt(((p - t) ** 2).mean())),
           "spearman": float(spearmanr(p, t).statistic) if np.std(p) > 0 else 0.0, "bias": float((p - t).mean()),
           "calib_slope": float(np.polyfit(p, t, 1)[0]) if np.std(p) > 0 else np.nan,
           "per_category_MAE": D.loc[ok].assign(e=e).groupby("category")["e"].mean().to_dict(),
           "per_seed_MAE": {int(k): float(v) for k, v in D.loc[ok].assign(e=e).groupby("seed")["e"].mean().items()}}
    if y == "Y1_primary":
        ev = t > EVENT
        out["n_events"] = int(ev.sum())
        if 0 < ev.sum() < len(ev):
            al = p > thr[ok].to_numpy()
            out.update({"PR_AUC": float(average_precision_score(ev, p)), "prevalence": float(ev.mean()),
                        "precision@p80": float(ev[al].mean()) if al.any() else np.nan,
                        "recall@p80": float(al[ev].mean()), "false_alarm@p80": float(al[~ev].mean())})
    return out


def cluster_diff(D, pa, pb, y, reps=2000):
    """Category-clustered bootstrap of MAE(a) - MAE(b)."""
    ok = ~D[y].isna() & ~pa.isna() & ~pb.isna()
    t = D.loc[ok, y].to_numpy()
    ea, eb = np.abs(pa[ok].to_numpy() - t), np.abs(pb[ok].to_numpy() - t)
    cat = D.loc[ok, "category"].to_numpy()
    cs = sorted(set(cat))
    sa = {c: ea[cat == c].sum() for c in cs}; sb = {c: eb[cat == c].sum() for c in cs}; nn = {c: (cat == c).sum() for c in cs}
    d = []
    for _ in range(reps):
        pick = RNG.choice(cs, len(cs))
        N = sum(nn[c] for c in pick)
        d.append((sum(sa[c] for c in pick) - sum(sb[c] for c in pick)) / N)
    return {"diff": float(ea.mean() - eb.mean()), "ci95": np.percentile(d, [2.5, 97.5]).tolist()}


def specs():
    S = {"B0_constant": ("const",), "B1_clock": ("ridge", CLOCK), "B2_occupancy": ("ridge", ["D_n_members", "INV_N"]),
         "B3_last_tail": ("ridge", ["A_pix_q99", "A_img_mean"]), "B4_fixed_threshold": ("thr",),
         "B5_fixed_schedule": ("sched",), "B6_lawful_persistence": ("ridge", ["A_grow_prev", "G_Agrow_prev_lag1"]),
         "B7_shuffled_history": ("shuffle", LAWFUL), "O1_oracle_persistence": ("oracle",),
         "controls": ("ridge", CONTROLS), "ALL": ("ridge", LAWFUL + ["THR"])}
    for f, cols in FAM.items():
        S[f"family_{f}"] = ("ridge", cols)
        S[f"controls+{f}"] = ("ridge", list(dict.fromkeys(CONTROLS + cols)))
    return S


LAWFUL_BASE = ["B0_constant", "B1_clock", "B2_occupancy", "B3_last_tail", "B4_fixed_threshold", "B5_fixed_schedule",
               "B6_lawful_persistence"]


def evaluate(D, y, scheme, S):
    P, R = {}, {}
    for name, sp in S.items():
        if sp[0] == "oracle" and y != "Y1_primary":
            continue
        P[name], thr = oof(D, sp, y, scheme)
        R[name] = metrics(D, P[name], thr, y)
    best = min(LAWFUL_BASE, key=lambda k: R[k]["MAE"])
    for name in R:
        if name not in LAWFUL_BASE and not name.startswith("O1"):
            R[name]["vs_best_lawful"] = cluster_diff(D, P[name], P[best], y)
            if name.startswith("controls+"):
                R[name]["vs_controls"] = cluster_diff(D, P[name], P["controls"], y)
    return {"best_lawful_baseline": best, "models": R}, P


def descriptive(D, Dfs):
    """G3: per-feature lead/lag association, sign consistency, occupancy confound, redundancy, R15 vs FS_B."""
    out = {"features": {}}
    lag = "ORACLE_prev_Y1"
    for f in LAWFUL:
        r = {}
        for y in TARGETS:
            ok = ~D[f].isna() & ~D[y].isna()
            r[f"rho_{y}"] = float(spearmanr(D.loc[ok, f], D.loc[ok, y]).statistic) if ok.sum() > 10 else np.nan
        ok = ~D[f].isna() & ~D[lag].isna()
        r["rho_current_damage"] = float(spearmanr(D.loc[ok, f], D.loc[ok, lag]).statistic) if ok.sum() > 10 else np.nan
        signs = []
        for sd in C.SEEDS:
            d = D[(D.seed == sd) & ~D[f].isna()]
            signs.append(float(np.sign(spearmanr(d[f], d["Y1_primary"]).statistic)) if len(d) > 10 else 0.0)
        r["seed_signs_Y1"] = signs
        res = []
        for c in C.STREAM:
            d = D[(D.category == c) & ~D[f].isna()]
            if len(d) >= 6 and d[f].std() > 0 and d["Y1_primary"].std() > 0:
                res.append(float(np.sign(spearmanr(d[f], d["Y1_primary"]).statistic)))
        r["category_sign_pos"], r["category_sign_neg"] = res.count(1.0), res.count(-1.0)
        # occupancy confound: residualise both on n_members (rank-linear)
        ok = ~D[f].isna()
        if ok.sum() > 10:
            x = D.loc[ok, ["D_n_members", "K_age"]].to_numpy(float)
            Xd = np.c_[np.ones(len(x)), x]
            def res_(v):
                b = np.linalg.lstsq(Xd, v, rcond=None)[0]; return v - Xd @ b
            r["partial_rho_Y1_given_occupancy_age"] = float(spearmanr(res_(D.loc[ok, f].to_numpy(float)),
                                                                     res_(D.loc[ok, "Y1_primary"].to_numpy(float))).statistic)
        okf = ~Dfs[f].isna()
        r["rho_Y1_FS_B_distinct"] = float(spearmanr(Dfs.loc[okf, f], Dfs.loc[okf, "Y1_primary"]).statistic) \
            if okf.sum() > 10 and Dfs.loc[okf, f].std() > 0 else np.nan
        r["missing_frac"] = float(D[f].isna().mean())
        r["std_over_rows"] = float(D[f].std())
        out["features"][f] = r
    cm = D[LAWFUL].corr(method="spearman")
    out["redundant_pairs_|rho|>0.9"] = [(a, b, float(cm.loc[a, b])) for i, a in enumerate(LAWFUL)
                                        for b in LAWFUL[i + 1:] if abs(cm.loc[a, b]) > 0.9]
    return out


def per_class(D, occ, desc_singles):
    sig = {c: g for g, cs in GROUPS.items() for c in cs}
    out = {}
    for c in C.STREAM:
        d = D[D.category == c]
        rec = {"acquisition_stage": C.a_of(c), "transitions_per_seed": int(d.groupby("seed").size().max()) if len(d) else 0,
               "nl_ret_00b_signature_group": sig[c],
               "replay_occupancy_by_end_of_stage": occ[(occ.category == c) & (occ.seed == 2711)].set_index(
                   "end_of_stage")["n_members"].to_dict()}
        if not len(d):
            rec["note"] = "acquired at stage 15: no transition after acquisition (no eligible rows)"
            out[c] = rec
            continue
        rec["Y1_by_stage_seedmean"] = d.groupby("s")["Y1_primary"].mean().round(5).to_dict()
        rec["largest_normal_tail_shock"] = d.loc[d.Y1_primary.idxmax(), ["seed", "s", "Y1_primary"]].to_dict()
        rec["largest_synthetic_shock_Y3"] = d.loc[d.Y3.idxmax(), ["seed", "s", "Y3"]].to_dict() if d.Y3.notna().any() else None
        rec["max_member_function_drift"] = float(d.E_member_img_drift.max()) if d.E_member_img_drift.notna().any() else None
        lead = {}
        for f in LAWFUL:
            x = d[[f, "Y1_primary", "ORACLE_prev_Y1"]].dropna()
            if len(x) >= 6 and x[f].std() > 0:
                lead[f] = (float(spearmanr(x[f], x["Y1_primary"]).statistic),
                           float(spearmanr(x[f], x["ORACLE_prev_Y1"]).statistic) if x["ORACLE_prev_Y1"].std() > 0 else np.nan)
        top = sorted(lead.items(), key=lambda kv: -abs(kv[1][0]))[:3]
        rec["top_descriptive_signals"] = [{"feature": f, "rho_future_Y1": a, "rho_current_damage": b,
                                           "leads": abs(a) > abs(b) if not np.isnan(b) else None,
                                           "seed_signs": [float(np.sign(spearmanr(x[f], x["Y1_primary"]).statistic))
                                                          if len(x := d[d.seed == sd][[f, "Y1_primary"]].dropna()) >= 4
                                                          and x[f].std() > 0 else 0.0 for sd in C.SEEDS]}
                                          for f, (a, b) in top]
        rec["missing"] = {f: float(d[f].isna().mean()) for f in LAWFUL if d[f].isna().any()}
        if len(d) < 12:
            rec["limitation"] = f"only {len(d)} rows over 3 seeds; per-class association is descriptive only"
        out[c] = rec
    return out


def run_validate():
    require(not (C.OUT / "CROSS_CLASS_VALIDATION.json").exists(), "one-shot")
    D, Dfs = load("ID"), load("FS_B")
    S = specs()
    res = {}
    keep = {}
    for y in TARGETS:
        for scheme in ("LOCO", "LOSO", "temporal"):
            r, P = evaluate(D, y, scheme, S)
            res[(y, scheme)] = r
            if y == "Y1_primary" and scheme == "LOCO":
                keep = P
            print(y, scheme, "best lawful", r["best_lawful_baseline"], round(r["models"][r["best_lawful_baseline"]]["MAE"], 5),
                  {k: round(v["MAE"], 5) for k, v in r["models"].items() if k.startswith("controls")}, flush=True)
    # FS_B replication: LOCO folds trained on ID rows of other categories, predicting FS_B distinct rows
    rep = {}
    for y in TARGETS:
        Dall = pd.concat([D.assign(_src="ID"), Dfs.assign(_src="FS")], ignore_index=True)
        P, R = {}, {}
        for name, sp in S.items():
            if sp[0] == "oracle":
                continue
            pred = pd.Series(np.nan, index=Dall.index)
            for c in C.STREAM:
                tr = Dall.index[(Dall._src == "ID") & (Dall.category != c) & ~Dall[y].isna()]
                te = Dall.index[(Dall._src == "FS") & (Dall.category == c)]
                if len(te):
                    pred.loc[te] = fit_predict(Dall, tr, te, sp, y)
            fs = Dall._src == "FS"
            P[name] = pred[fs]
            R[name] = metrics(Dall[fs], pred[fs], pd.Series(np.inf, index=Dall.index[fs]), y)
        best = min(LAWFUL_BASE, key=lambda k: R[k]["MAE"])
        rep[y] = {"best_lawful_baseline": best,
                  "models": {k: {"MAE": v["MAE"], "spearman": v["spearman"], "n": v["n"]} for k, v in R.items()}}
    # incremental information table and gate (Y1 primary)
    gate, inc = {}, {}
    loco, loso, tmp = res[("Y1_primary", "LOCO")], res[("Y1_primary", "LOSO")], res[("Y1_primary", "temporal")]
    bl = loco["best_lawful_baseline"]
    for f in FAM:
        m = loco["models"][f"controls+{f}"]
        base = loco["models"][bl]
        i = m["MAE"] <= 0.95 * base["MAE"]
        ii = m["vs_best_lawful"]["ci95"][1] < 0
        iii = m["vs_controls"]["ci95"][1] < 0
        lb = loso["best_lawful_baseline"]
        iv = sum(loso["models"][f"controls+{f}"]["per_seed_MAE"][sd] < loso["models"][lb]["per_seed_MAE"][sd]
                 for sd in C.SEEDS)
        v = sum(m["per_category_MAE"][c] < base["per_category_MAE"][c] for c in m["per_category_MAE"])
        t_ok = tmp["models"][f"controls+{f}"]["MAE"] <= tmp["models"][tmp["best_lawful_baseline"]]["MAE"]
        r_ok = rep["Y1_primary"]["models"][f"controls+{f}"]["MAE"] <= rep["Y1_primary"]["models"][
            rep["Y1_primary"]["best_lawful_baseline"]]["MAE"]
        gate[f] = {"i_mae_le_0.95_best": bool(i), "ii_ci_vs_best_lt0": bool(ii), "iii_ci_vs_controls_lt0": bool(iii),
                   "iv_loso_seeds_beating": int(iv), "v_categories_beating": int(v), "temporal_ok": bool(t_ok),
                   "fsb_replication_ok": bool(r_ok),
                   "PASS": bool(i and ii and iii and iv >= 2 and v >= 9)}
        inc[f] = {y: {sc: {"controls_MAE": res[(y, sc)]["models"]["controls"]["MAE"],
                           "controls+family_MAE": res[(y, sc)]["models"][f"controls+{f}"]["MAE"],
                           "vs_controls": res[(y, sc)]["models"][f"controls+{f}"].get("vs_controls")}
                      for sc in ("LOCO", "LOSO", "temporal")} for y in TARGETS}
    n_events = loco["models"]["B0_constant"].get("n_events", 0)
    missing_primary = int(D["Y1_primary"].isna().sum())
    # single-feature OOF ranking (descriptive, LOCO, Y1)
    singles = {}
    for f in LAWFUL:
        p, thr = oof(D, ("ridge", [f]), "Y1_primary", "LOCO")
        mm = metrics(D, p, thr, "Y1_primary")
        singles[f] = {"MAE": mm["MAE"], "spearman": mm["spearman"]}
    fam_rank = sorted(FAM, key=lambda f: loco["models"][f"controls+{f}"]["MAE"])
    W(C.OUT / "CROSS_CLASS_VALIDATION.json", {y: res[(y, "LOCO")] for y in TARGETS} | {
        "single_feature_LOCO_Y1": singles, "family_ranking_LOCO_Y1_by_OOF_MAE": fam_rank,
        "fsb_replication": rep, "gate_Y1": gate, "n_events_Y1": n_events, "missing_primary": missing_primary})
    W(C.OUT / "CROSS_SEED_VALIDATION.json", {y: res[(y, "LOSO")] for y in TARGETS})
    W(C.OUT / "TEMPORAL_VALIDATION.json", {y: res[(y, "temporal")] for y in TARGETS})
    W(C.OUT / "INCREMENTAL_INFORMATION.json", inc)
    # descriptive (G3) + per-class
    occ = pd.read_csv(C.OUT / "REPLAY_OCCUPANCY.csv")
    desc = descriptive(D, Dfs)
    pcs = per_class(D, occ, singles)
    W(C.OUT / "PER_CLASS_TRAJECTORIES.json", {"per_class": pcs, "descriptive_associations": desc})
    rows = []
    for c in C.STREAM:
        r = pcs[c]
        top = r.get("top_descriptive_signals", [])
        m = loco["models"]
        rows.append({"category": c, "a": C.a_of(c), "group": r["nl_ret_00b_signature_group"],
                     "rows_ID": int((D.category == c).sum()),
                     "Y1_mean": float(D.loc[D.category == c, "Y1_primary"].mean()) if (D.category == c).any() else np.nan,
                     "top1_signal": top[0]["feature"] if top else "", "top1_rho": top[0]["rho_future_Y1"] if top else np.nan,
                     "top1_leads": top[0]["leads"] if top else None,
                     "top1_seed_signs": "/".join(str(int(x)) for x in top[0]["seed_signs"]) if top else "",
                     "best_lawful_MAE": m[bl]["per_category_MAE"].get(c, np.nan),
                     **{f"MAE_controls+{f}": m[f"controls+{f}"]["per_category_MAE"].get(c, np.nan) for f in FAM},
                     "min_members_end": int(occ[(occ.category == c)]["n_members"].min())})
    pd.DataFrame(rows).to_csv(C.OUT / "PER_CLASS_SIGNAL_MATRIX.csv", index=False)
    # frozen risk predictions for posthoc alignment: best non-baseline LOCO model on Y1 (normal-only choice)
    cand = [k for k in loco["models"] if k.startswith("controls+") or k.startswith("family_") or k == "ALL"]
    chosen = min(cand, key=lambda k: loco["models"][k]["MAE"])
    D.assign(pred=keep[chosen], pred_best_lawful=keep[bl])[["arm", "seed", "category", "s", "pred",
                                                            "pred_best_lawful"]].to_csv(C.OUT / "FROZEN_RISK_PREDICTIONS.csv",
                                                                                        index=False)
    W(C.OUT / "G4_FREEZE.json", {"chosen_model_for_alignment": chosen, "gate_Y1": gate,
                                            "best_lawful_baseline_LOCO": bl, "n_events": n_events,
                                            "missing_primary": missing_primary,
                                            "predictions_sha256": sha(C.OUT / "FROZEN_RISK_PREDICTIONS.csv"),
                                            "utc": K.now()})
    print("validate done; chosen", chosen, "gate", {f: g["PASS"] for f, g in gate.items()})


def run_align():
    """POSTHOC_EXPOSED_REAL_AP_DIAGNOSTIC: reads Y4 only now; cannot change G4."""
    fr = K.read(C.OUT / "G4_FREEZE.json")
    require(sha(C.OUT / "FROZEN_RISK_PREDICTIONS.csv") == fr["predictions_sha256"], "frozen predictions changed")
    P = pd.read_csv(C.OUT / "FROZEN_RISK_PREDICTIONS.csv")
    O = pd.read_csv(C.OUT / "OUTCOME_TABLE.csv")
    D = P.merge(O[O.arm == "ID"][["seed", "category", "s", "Y4_exposed", "Y1_primary"]], on=["seed", "category", "s"])
    def al(d, col):
        d = d.dropna(subset=["Y4_exposed", col])
        if len(d) < 6 or d[col].std() == 0:
            return {"n": len(d), "spearman": None}
        return {"n": len(d), "spearman": float(spearmanr(d[col], d["Y4_exposed"]).statistic)}
    out = {"label": "POSTHOC_EXPOSED_REAL_AP_DIAGNOSTIC", "chosen_model": fr["chosen_model_for_alignment"],
           "overall": al(D, "pred"), "overall_best_lawful_baseline": al(D, "pred_best_lawful"),
           "Y1_vs_Y4_actual": al(D.assign(pred=D.Y1_primary), "pred"),
           "groups": {g: al(D[D.category.isin(cs)], "pred") for g, cs in GROUPS.items()},
           "per_category": {c: al(D[D.category == c], "pred") for c in C.STREAM},
           "per_seed": {int(sd): al(D[D.seed == sd], "pred") for sd in C.SEEDS}}
    hi = D.pred > D.pred.quantile(0.8)
    sev = D.Y4_exposed >= 0.05
    out["severe_transitions"] = int(sev.sum())
    out["false_negatives_severe_below_median_risk"] = D[sev & (D.pred <= D.pred.median())][
        ["seed", "category", "s", "Y4_exposed", "pred"]].to_dict("records")
    stable = D.category.isin(GROUPS["stable_or_improving"])
    out["false_positive_rate_stable_top20pct"] = float(hi[stable].mean())
    out["top20pct_precision_for_Y4_ge_0.02"] = float((D.Y4_exposed[hi] >= 0.02).mean())
    W(C.OUT / "REAL_AP_ALIGNMENT_DESCRIPTIVE.json", out)
    print("align", out["overall"], out["groups"])


def run_decide():
    fr = K.read(C.OUT / "G4_FREEZE.json")
    cv = K.read(C.OUT / "CROSS_CLASS_VALIDATION.json")
    al = K.read(C.OUT / "REAL_AP_ALIGNMENT_DESCRIPTIVE.json")
    cov = K.read(C.OUT / "FULL_STREAM_COVERAGE.json")
    g = fr["gate_Y1"]
    if cov["bad_checkpoints"] or cov["replay_record_mismatches"]:
        label = "INTEGRITY_BLOCKED"
    elif fr["missing_primary"] or fr["n_events"] < 15:
        label = "UNDERPOWERED_OR_INCOMPLETE"
    else:
        full = [f for f, v in g.items() if v["PASS"] and v["temporal_ok"] and v["fsb_replication_ok"]]
        cond = [f for f, v in g.items() if v["ii_ci_vs_best_lt0"] and v["iii_ci_vs_controls_lt0"] and not v["PASS"]
                and v["v_categories_beating"] >= 4]
        if full:
            rho = al["overall"]["spearman"]
            label = ("NORMAL_ONLY_PREDICTIVE_BUT_REAL_AP_UNALIGNED" if rho is None or rho <= 0.10
                     else "MULTICLASS_SLOW_SIGNAL_SUPPORTED")
        elif cond:
            label = "CLASS_CONDITIONAL_SIGNAL_ONLY"
        else:
            label = "NO_INCREMENTAL_PREDICTIVE_SIGNAL"
    W(C.OUT / "FINAL_VERDICT.json", {"label": label, "gate_Y1": g, "rule": GATE,
                                                "alignment_overall": al["overall"], "utc": K.now()})
    print("verdict", label)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "align", "decide"])
    a = ap.parse_args()
    {"validate": run_validate, "align": run_align, "decide": run_decide}[a.cmd]()
