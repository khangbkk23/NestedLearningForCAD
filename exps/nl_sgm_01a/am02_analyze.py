"""NL-SGM-01A-AM02/v2 analysis: frozen metrics (§4, §13), clustered/nested CIs (§5), preregistered verdict (§6).

    python -m exps.nl_sgm_01a.am02_analyze   -> results/hope_cad/nl_sgm_01a_am02/{METRICS.json, FINAL_VERDICT.json,
                                                PER_CATEGORY_TABLE.csv, MATCHED_CONTROL_RESULTS.csv, RESOURCE_LEDGER.json}
"""
from __future__ import annotations

import csv
import io

import numpy as np
import torch

from exps.nl_sgm_01a import am02_eval as E
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import lock as L
from exps.nl_sgm_01a.am02_run import CAPS, LOCKP, OUT
from exps.task_p.p0_normal_support import require

ARMS = {"C1": "C1_FPS_FP32", "C2": "C2_FPS_PQ", "C3": "C3_HASH_PQ", "C4": "c4"}
T_END = 14
OLD = list(C.OLD)


def load():
    st = OUT / "stages"
    for d in ("a0", *ARMS.values()):
        require((st / d / "STAGE_DONE.json").exists(), f"stage {d} incomplete")
    W = {k: torch.load(st / "a0" / f"W_{k:02d}_{c}.pt") for k, c in enumerate(C.ORDER)}
    ev = {a: {t: torch.load(st / d / f"eval_t{t:02d}.pt") for t in range(15)} for a, d in ARMS.items()}
    ck = {a: {t: torch.load(st / d / f"ckpt_t{t:02d}.pt") for t in range(15)} for a, d in ARMS.items()}
    return st, W, ev, ck


def gidsets(ck_t):
    g = ck_t["gids"]; o = ck_t["owner"][:len(g)].long()
    return {k: set(g[o == k].tolist()) for k in torch.unique(o).tolist()}


def main():
    L.verify_lock(LOCKP)
    st, W, ev, ck = load()
    integrity = []
    # ---- integrity: identical C0 across C2, C3, C4; C4 identities == C1 identities
    c0 = torch.load(st / "c4" / "C0.pt")["codebook"]
    for a in ("C2", "C3"):
        cbs = ck[a][T_END]["codebooks"]
        require(cbs.shape[0] == 1 and torch.equal(cbs[0], c0), f"INTEGRITY: {a} codebook != C4 C0")
    for t in range(15):
        require(torch.equal(ck["C4"][t]["gids"], ck["C1"][t]["gids"]), f"INTEGRITY: C4 ids != C1 ids at t={t}")
    integrity.append("C0 identical across C2/C3/C4; C4 identities == C1 identities at all 15 boundaries")
    sets = {a: {t: gidsets(ck[a][t]) for t in range(15)} for a in ("C1", "C2", "C3")}
    for a in sets:
        for t in range(1, 15):
            for k in range(t):
                require(sets[a][t][k] <= sets[a][t - 1][k], f"INTEGRITY: {a} subset violated k={k} t={t}")
    integrity.append("historical support of every category is a subset of its previous support (all arms)")

    # ---- scales (SEL only)
    q50, q90 = {}, {}
    for k in range(15):
        q50[k], q90[k] = E.scales(W[k]["sel_d"].numpy())
    degenerate = [C.ORDER[k] for k in range(15) if q50[k] <= E.DEGENERATE]
    blocked = any(C.ORDER.index(c) in OLD for c in degenerate)

    rows, per_img = [], {m: {} for m in ("R", "A", "e14s", "Q14")}
    cats_img = {}
    cat_tab = []
    for k, c in enumerate(C.ORDER):
        sl = slice(6 * k, 6 * k + 6)
        dW = W[k]["eval_d_all"][sl].numpy()                                # [6,784] frozen reference W_c
        idW = W[k]["eval_id_all"][sl].numpy()
        orc = np.stack([W[j]["eval_d_all"][sl].numpy() for j in range(15)])  # [15 cats, 6, 784]
        tail = dW >= q90[k]
        row = {"a(c)": k, "category": c, "group": "old" if k in OLD else "late", "q50": q50[k], "q90": q90[k],
               "degenerate": c in degenerate}
        for a in ARMS:
            d = {t: ev[a][t][k]["cat_d"].numpy() for t in range(k, 15)}
            e = {t: E.excess(d[t], dW, q50[k]) for t in d}
            R = e[T_END] - e[k]; Aa = np.abs(e[T_END])
            per_img["R"].setdefault(a, {})[k] = R
            per_img["A"].setdefault(a, {})[k] = Aa
            per_img["e14s"].setdefault(a, {})[k] = e[T_END]
            mo = orc[:T_END + 1].min(0)                                     # moving global oracle W_14
            mo_acq = orc[:k + 1].min(0)
            gd14 = ev[a][T_END][k]["glob_d"].numpy(); gdA = ev[a][k][k]["glob_d"].numpy()
            g_e14 = np.abs(E.excess(gd14, mo, q50[k])); g_eA = np.abs(E.excess(gdA, mo_acq, q50[k]))
            stored14 = sets["C1" if a == "C4" else a][T_END].get(k, set())
            storedA = sets["C1" if a == "C4" else a][k].get(k, set())
            row.update({
                f"{a}_R": R.mean(), f"{a}_A": Aa.mean(), f"{a}_e_acq_signed": e[k].mean(), f"{a}_e14_signed": e[T_END].mean(),
                f"{a}_cov14": E.coverage(d[T_END], dW, q50[k]).mean(),
                f"{a}_tailcov_acq": np.nanmean(E.coverage(d[k], dW, q50[k], tail)),
                f"{a}_tailcov14": np.nanmean(E.coverage(d[T_END], dW, q50[k], tail)),
                f"{a}_under14": E.under_fraction(d[T_END], dW, q50[k]).mean(),
                f"{a}_falsesupport14": E.false_support(d[T_END], dW, q50[k]).mean(),
                f"{a}_rows_acq": len(storedA), f"{a}_rows14": len(stored14),
                f"{a}_survival14": E.survival(stored14, storedA),
                f"{a}_Wnearest_stored_acq": float(np.isin(idW, list(storedA)).mean()),
                f"{a}_Wnearest_stored14": float(np.isin(idW, list(stored14)).mean()),
                f"{a}_glob_fid_acq": g_eA.mean(), f"{a}_glob_fid14": g_e14.mean(), f"{a}_glob_drift": (g_e14 - g_eA).mean(),
                f"{a}_capture_acq": E.capture(ev[a][k][k]["glob_owner"].numpy(), k).mean(),
                f"{a}_capture14": E.capture(ev[a][T_END][k]["glob_owner"].numpy(), k).mean(),
            })
        o14 = orc[:T_END + 1].argmin(0); oA = orc[:k + 1].argmin(0)
        row["oracle_capture_acq"] = float((oA != k).mean()); row["oracle_capture14"] = float((o14 != k).mean())
        Q14 = (E.score(ev["C4"][T_END][k]["cat_d"].numpy()) - E.score(ev["C1"][T_END][k]["cat_d"].numpy())) / q50[k]
        QA = (E.score(ev["C4"][k][k]["cat_d"].numpy()) - E.score(ev["C1"][k][k]["cat_d"].numpy())) / q50[k]
        per_img["Q14"][k] = Q14
        pq = (ev["C4"][T_END][k]["cat_d"].numpy() - ev["C1"][T_END][k]["cat_d"].numpy()) / q50[k]
        row.update({"Q_acq_signed": QA.mean(), "Q14_signed": Q14.mean(), "Q14_abs": np.abs(Q14).mean(),
                    "Q14_patch_abs": np.abs(pq).mean(),
                    "jaccard_C1_C2_acq": E.jaccard(sets["C1"][k][k], sets["C2"][k][k]),
                    "jaccard_C1_C2_14": E.jaccard(sets["C1"][T_END][k], sets["C2"][T_END][k]),
                    "jaccard_C2_C3_acq": E.jaccard(sets["C2"][k][k], sets["C3"][k][k]),
                    "jaccard_C2_C3_14": E.jaccard(sets["C2"][T_END][k], sets["C3"][T_END][k])})
        cats_img[k] = np.full(6, k)
        cat_tab.append(row)

    diag = torch.load(st / "c4" / "codec_diagnostics.pt")
    rho = {k: diag[k]["recon_norm_median"] / q50[k] for k in range(15)}
    for row in cat_tab:
        k = row["a(c)"]
        row.update({"rho": rho[k], "rho_ratio_bottle": rho[k] / rho[0],
                    "codeword_entropy_mean": float(diag[k]["codeword_entropy"].mean()),
                    "frac_resid_gt_bottle_p99": diag[k]["frac_resid_gt_bottle_p99"]})

    # ---- inference over old categories (EVAL images, paired)
    def vec(metric, a, ks=OLD):
        return np.concatenate([per_img[metric][a][k] for k in ks])
    cats = np.concatenate([cats_img[k] for k in OLD])
    S = {"R_C1": E.ci_both(vec("R", "C1"), cats), "A_C1": E.ci_both(vec("A", "C1"), cats),
         "R_C2": E.ci_both(vec("R", "C2"), cats), "A_C2": E.ci_both(vec("A", "C2"), cats),
         "dR": E.ci_both(vec("R", "C1") - vec("R", "C2"), cats), "dA": E.ci_both(vec("A", "C1") - vec("A", "C2"), cats)}
    extra = {"R_C3": E.ci_both(vec("R", "C3"), cats), "A_C3": E.ci_both(vec("A", "C3"), cats),
             "R_C4": E.ci_both(vec("R", "C4"), cats), "A_C4": E.ci_both(vec("A", "C4"), cats),
             "Q14_abs": E.ci_both(np.abs(np.concatenate([per_img["Q14"][k] for k in OLD])), cats),
             "capacity_plus_selection_path_A_C4_minus_C2": E.ci_both(vec("A", "C4") - vec("A", "C2"), cats),
             "selector_A_C3_minus_C2": E.ci_both(vec("A", "C3") - vec("A", "C2"), cats),
             "selector_R_C3_minus_C2": E.ci_both(vec("R", "C3") - vec("R", "C2"), cats)}
    signs = {"dR>0": int(sum(per_img["R"]["C1"][k].mean() - per_img["R"]["C2"][k].mean() > 0 for k in OLD)),
             "dA>0": int(sum(per_img["A"]["C1"][k].mean() - per_img["A"]["C2"][k].mean() > 0 for k in OLD))}
    late = list(C.LATE)
    cl = np.concatenate([cats_img[k] for k in late])
    late_stats = {f"{m}_{a}": E.ci_both(vec(m, a, late), cl) for m in ("R", "A") for a in ARMS}

    v = E.verdict(S, blocked=blocked)
    flags = []
    if v == "RESIDUAL_GAP_CONSOLIDATION_CANDIDATE" and E.codebook_limited(
            extra["Q14_abs"]["primary"]["point"], S["A_C2"]["primary"]["point"], [rho[k] / rho[0] for k in OLD]):
        flags.append("CODEBOOK_LIMITED")

    metrics = {"old_category_stats": S, "old_category_extra": extra, "old_sign_counts_of_7": signs,
               "late_category_stats": late_stats, "q50": q50, "q90": q90, "degenerate": degenerate,
               "integrity": integrity, "note": "CIs over 7 old-category clusters are approximate"}
    C.write_json(OUT / "METRICS.json", metrics)

    def w_csv(path, rows_):
        buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=list(rows_[0].keys())); wr.writeheader()
        for r in rows_:
            wr.writerow({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()})
        with path.open("x") as f:
            f.write(buf.getvalue())
    w_csv(OUT / "PER_CATEGORY_TABLE.csv", cat_tab)
    arm_rows = []
    for a, d in ARMS.items():
        done = C.read_json(st / d / "STAGE_DONE.json")
        def pt(key):
            s = (S | extra).get(key)
            return "" if s is None else s["primary"]["point"]
        arm_rows.append({"arm": a, "stage": d, "bytes": done.get("bytes"),
                         "old_R": pt(f"R_{a}"), "old_R_lo": (S | extra)[f"R_{a}"]["primary"]["lo"], "old_R_hi": (S | extra)[f"R_{a}"]["primary"]["hi"],
                         "old_A": pt(f"A_{a}"), "old_A_lo": (S | extra)[f"A_{a}"]["primary"]["lo"], "old_A_hi": (S | extra)[f"A_{a}"]["primary"]["hi"],
                         "old_survival14": np.mean([r[f"{a}_survival14"] for r in cat_tab if r["group"] == "old"]),
                         "old_tailcov14": np.mean([r[f"{a}_tailcov14"] for r in cat_tab if r["group"] == "old"]),
                         "old_falsesupport14": np.mean([r[f"{a}_falsesupport14"] for r in cat_tab if r["group"] == "old"]),
                         "old_under14": np.mean([r[f"{a}_under14"] for r in cat_tab if r["group"] == "old"]),
                         "old_capture14": np.mean([r[f"{a}_capture14"] for r in cat_tab if r["group"] == "old"]),
                         "old_glob_fid14": np.mean([r[f"{a}_glob_fid14"] for r in cat_tab if r["group"] == "old"]),
                         "wall_h": done["wall_s"] / 3600, "peak_rss_gb": done["peak_rss_gb"]})
    w_csv(OUT / "MATCHED_CONTROL_RESULTS.csv", arm_rows)

    stages = {d: C.read_json(st / d / "STAGE_DONE.json") for d in ("a0", *ARMS.values())}
    total_h = sum(s["wall_s"] for s in stages.values()) / 3600
    ledger = {"caps": CAPS, "stages": {d: {k: s[k] for k in ("wall_s", "peak_rss_gb", "device", "threads")} |
                                       {"bytes": s.get("bytes"), "transient_peak_bytes": s.get("transient_peak_bytes")}
                                       for d, s in stages.items()},
              "total_wall_h": total_h, "disk_gb": sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9,
              "persistent_bytes_formula": "C1 2,499x(3072+1); C2/C3 75,121x(96+1)+393,216 FP16 codebook; C4 2,499x97+393,216"}
    C.write_json(OUT / "RESOURCE_LEDGER.json", ledger)
    fv = {"task": "NL-SGM-01A-AM02", "version": "v2", "utc": C.now(), "verdict": v, "flags": flags,
          "primary": {k: {s: {kk: S[k][s][kk] for kk in ("point", "lo", "hi")} for s in ("primary", "nested")} for k in S},
          "thresholds": {"delta": E.DELTA, "eps": E.EPS, "note": "operational normal-support margins, not anomaly-AP thresholds"},
          "scope": "bottle-trained fixed-C0 PQ only; normal-support proxy; no anomaly labels used",
          "lock_sha256": C.sha(LOCKP), "integrity": integrity}
    C.write_json(OUT / "FINAL_VERDICT.json", fv)
    print(v, flags)
    for k in S:
        print(k, {s: (round(S[k][s]["point"], 4), round(S[k][s]["lo"], 4), round(S[k][s]["hi"], 4)) for s in ("primary", "nested")})


if __name__ == "__main__":
    main()
