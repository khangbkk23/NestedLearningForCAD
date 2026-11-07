"""NL-RET-01A: build FEATURE_TABLE / OUTCOME_TABLE from the extraction records (CPU).

Every feature z(c, seed, arm, s) uses only: stats of c's Replay-24 members held at the end of s, current-task (task s)
train normals, Q2 items hosted on those members, decoder states <= s, and logs of stages <= s. Eval-only nonmembers
enter ONLY the outcome table (and oracle diagnostics clearly prefixed `ORACLE_`).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch

from exps.cadfinal import common as K
from exps.dinomaly.common import require, sha
from exps.nl_ret_00b import core as X
from exps.nl_ret_01a import common as C
from exps.nl_ret_01a.extract import EXT, BS

COL = {k: i for i, k in enumerate(["img", "q50", "q90", "q95", "q99", "q999", "grid_mean", "gcos"])}
TOKVEC = C.OUT / "token_mean_vectors.pt"


def map_rows(ever, n_c):
    """Row order of maps64 in an extraction record (batch-major, `ever` order within a batch)."""
    order = [j for i in range(0, n_c, BS) for j in ever if i <= j < i + BS]
    return {j: r for r, j in enumerate(order)}


def token_vectors(proto, n):
    """Seed-independent token-mean encoder vectors [n_c, 1536] for every train identity (from the frozen cache)."""
    if TOKVEC.exists():
        return torch.load(TOKVEC, weights_only=True)
    from exps.rh01 import common as RC
    out = {}
    for c in C.STREAM:
        f = RC.load_features(c, mmap=True)
        out[c] = torch.stack([f[i].mean(dim=1).flatten() for i in range(len(f))]).float()
    torch.save(out, TOKVEC)
    return out


def syn_stats(grid, cov):
    """grid, cov: [m, 28, 28] for the selected items -> pooled AP, margin, rank loss, saturation, bg q99."""
    if grid is None or len(grid) == 0:
        return dict(ap=np.nan, margin=np.nan, rankloss=np.nan, sat=np.nan, bg_q99=np.nan)
    g, cv = grid.float(), cov.float()
    pos, neg = g[cv >= 0.5].numpy(), g[cv == 0].numpy()
    if pos.size == 0 or neg.size == 0:
        return dict(ap=np.nan, margin=np.nan, rankloss=np.nan, sat=np.nan, bg_q99=np.nan)
    rl = []
    for gi, ci in zip(g, cv):
        p, q = gi[ci >= 0.5], gi[ci == 0]
        if p.numel() and q.numel():
            rl.append(float(torch.nn.functional.softplus((q[None, :] - p[:, None]) / 0.05).mean()))
    return dict(ap=X.ap_pn(pos, [np.sort(neg)]), margin=float(pos.mean() - np.quantile(neg, 0.99)),
                rankloss=float(np.mean(rl)) if rl else np.nan, sat=float(np.mean(np.array(rl) < 1e-5)) if rl else np.nan,
                bg_q99=float(np.quantile(neg, 0.99)))


def aux_eff(seed, s):
    base = (C.ROOT / "results/hope_cad/nl_align_01/g1/runs" if s <= 9 else
            C.ROOT / "results/hope_cad/nl_align_02_mvtec/m3/runs") / f"FS_B_seed{seed}/LOG{s}.pt"
    d = torch.load(base, weights_only=False)
    nA, nN = np.asarray(d["nA"]), np.asarray(d["nN"])
    return float(np.mean(np.asarray(d["scale"]) * nA / nN))


def build():
    proto, n = C.proto_and_n()
    tok = token_vectors(proto, n)
    traj = {(r["arm"], r["seed"], r["category"]): {st["stage"]: st["ap"] for st in r["stages"]}
            for r in K.read(C.ROOT / "results/hope_cad/nl_ret_00b/STAGE_TRAJECTORY.json")["records"]}
    feats, outs = [], []
    for seed in C.SEEDS:
        recs = {c: torch.load(EXT / f"seed{seed}" / f"{c}.pt", weights_only=False) for c in C.STREAM}
        grads = json.loads((EXT / f"seed{seed}" / "grads.json").read_text())
        for arm in C.ARMS:
            key = lambda s: f"{C.owner(arm, s)}|{s}"
            logs = {}
            for s in range(1, 16):
                _, _, _, rec = C.checkpoint(arm, seed, s)
                cur = rec["loss_curve_every50"]
                logs[s] = {"F_loss_last": float(cur[-1]), "F_loss_shock": float(cur[0] / cur[-1]),
                           "F_clip_sat": rec.get("clip_saturation", np.nan) if C.owner(arm, s) == "FS_B" else np.nan,
                           "F_aux_eff": aux_eff(seed, s) if C.owner(arm, s) == "FS_B" else 0.0}
            for c in C.STREAM:
                r = recs[c]
                a = C.a_of(c)
                ever = r["ever"]
                mrow = map_rows(ever, n[c])
                NM = C.nonmembers(seed, n, c)
                require(not set(NM) & set(ever), "nonmember/member overlap")
                acq = key(a)
                m_acq = C.members(seed, n, a)[c]
                tau = float(np.quantile(torch.stack([r["maps64"][acq][mrow[j]] for j in m_acq]).float().numpy(), 0.99))
                keys = r["syn_keys"]
                cov = r["syn_cov"]
                ev_hosts = set(r["hosts"]["current"]) - set(ever)
                item_eval = [k for k, kk in enumerate(keys) if kk[1] in ev_hosts]
                hist_grow, hist_drift, prev_cap = [], [], None
                for s in range(a, 16):
                    S = r["stats"][key(s)].numpy()
                    m = C.members(seed, n, s)[c]
                    item_m = [k for k, kk in enumerate(keys) if kk[1] in set(m)]
                    q99m = S[m, COL["q99"]].mean()
                    mp = torch.stack([r["maps64"][key(s)][mrow[j]] for j in m]).float()
                    f = {"arm": arm, "seed": seed, "category": c, "a": a, "s": s,
                         "shared_with_ID": arm == "FS_B" and s <= 5,
                         "A_img_mean": S[m, 0].mean(), "A_pix_q99": q99m, "A_pix_q999": S[m, COL["q999"]].mean(),
                         "A_img_max": S[m, 0].max(), "A_tail_mass": float((mp > tau).float().mean()),
                         "A_grow_acq": math.log(q99m / r["stats"][acq].numpy()[m, COL["q99"]].mean()),
                         "A_cv": float(S[m, 0].std() / S[m, 0].mean()) if len(m) > 1 else np.nan,
                         "D_n_members": len(m), "K_s": s, "K_age": s - a, "K_inv_age": 1.0 / (1 + s - a)}
                    # current task (task s) normals at state s, all completed members (global)
                    cs = C.STREAM[s - 1]
                    Scur = recs[cs]["stats"][key(s)].numpy()
                    f["A_cur_vs_rep"] = math.log(S[m, 0].mean() / Scur[:, 0].mean())
                    allm = C.members(seed, n, s)
                    if s > a:
                        P = r["stats"][key(s - 1)].numpy()
                        f["A_grow_prev"] = math.log(q99m / P[m, COL["q99"]].mean())
                        mprev = C.members(seed, n, s - 1)[c]
                        f["A_membership_effect"] = q99m - S[mprev, COL["q99"]].mean()
                        f["E_member_img_drift"] = float(np.abs(S[m, 0] - P[m, 0]).mean())
                        mpp = torch.stack([r["maps64"][key(s - 1)][mrow[j]] for j in m]).float()
                        f["E_member_map_corr"] = float(np.mean([np.corrcoef(x.flatten(), y.flatten())[0, 1]
                                                                for x, y in zip(mp.numpy(), mpp.numpy())]))
                        num = sum(recs[d]["stats"][key(s)].numpy()[ix, 0].sum() for d, ix in allm.items()
                                  if C.a_of(d) <= s - 1)
                        den = sum(recs[d]["stats"][key(s - 1)].numpy()[ix, 0].sum() for d, ix in allm.items()
                                  if C.a_of(d) <= s - 1)
                        f["A_global_grow_prev"] = math.log(num / den) if den > 0 else np.nan
                        f["E_global_img_drift"] = float(np.mean(np.concatenate(
                            [np.abs(recs[d]["stats"][key(s)].numpy()[ix, 0] - recs[d]["stats"][key(s - 1)].numpy()[ix, 0])
                             for d, ix in allm.items() if C.a_of(d) <= s - 1])))
                    else:
                        for k_ in ("A_grow_prev", "A_membership_effect", "E_member_img_drift", "E_member_map_corr",
                                   "A_global_grow_prev", "E_global_img_drift"):
                            f[k_] = np.nan
                    # synthetic on member-hosted items (lawful)
                    g = r["syn_grid"][key(s)]
                    st = syn_stats(g[item_m] if item_m else None, cov[item_m] if item_m else None)
                    f.update({f"C_{k}": v for k, v in st.items()})
                    f["C_n_items"] = len(item_m)
                    st_acq = syn_stats(r["syn_grid"][acq][item_m] if item_m else None, cov[item_m] if item_m else None)
                    f["C_ap_grow_acq"] = f["C_ap"] - st_acq["ap"]
                    f["G_Cap_delta_prev"] = f["C_ap"] - prev_cap if prev_cap is not None else np.nan
                    prev_cap = f["C_ap"]
                    # coverage (frozen encoder vectors of lawful members and current task)
                    vm, vc = tok[c][m], tok[cs]
                    nm_ = torch.nn.functional.normalize
                    f["D_dispersion"] = float((1 - nm_(vm, dim=1) @ nm_(vm, dim=1).T)[
                        torch.triu_indices(len(m), len(m), 1).unbind()].mean()) if len(m) > 1 else np.nan
                    f["D_cur_dist"] = float(1 - torch.nn.functional.cosine_similarity(vm.mean(0), vc.mean(0), dim=0))
                    f["D_cur_nn"] = float((1 - nm_(vm, dim=1) @ nm_(vc, dim=1).T).min(dim=1).values.mean())
                    # drift and optimisation
                    gr = grads[key(s)]
                    f["E_param_rel"] = gr.get("param_rel_change", np.nan)
                    f["E_param_block_max"] = gr.get("param_block_max_rel", np.nan)
                    f["F_g_member_norm"] = gr["per_cat"][c]["norm"]
                    f["F_cos_cur_member"] = gr["per_cat"][c]["cos_cur"]
                    f["F_cos_rep_member"] = gr["per_cat"][c]["cos_rep"]
                    f.update(logs[s])
                    # history (s' <= s only)
                    if not math.isnan(f["A_grow_prev"]):
                        hist_grow.append(f["A_grow_prev"])
                    if not math.isnan(f["E_member_img_drift"]):
                        hist_drift.append(f["E_member_img_drift"])
                    f["G_Agrow_prev_lag1"] = hist_grow[-2] if len(hist_grow) >= 2 else np.nan
                    f["G_Agrow_ewma"] = _ewma(hist_grow)
                    f["G_Agrow_maxexc"] = max(hist_grow) if hist_grow else np.nan
                    f["G_Edrift_ewma"] = _ewma(hist_drift)
                    # oracle diagnostics (EVAL-ONLY; never in lawful models)
                    vn = tok[c][NM]
                    f["ORACLE_nonmember_cov_nn"] = float((1 - nm_(vn, dim=1) @ nm_(vm, dim=1).T).min(dim=1).values.mean())
                    if s <= 14:
                        feats.append(f)
                    # outcomes over s -> s+1
                    if s <= 14:
                        S1 = r["stats"][key(s + 1)].numpy()
                        q0, q1 = np.quantile(S[NM, 0], 0.9), np.quantile(S1[NM, 0], 0.9)
                        gap = lambda T: math.log(T[NM, 0].mean()) - math.log(T[m, 0].mean())
                        ev0 = syn_stats(g[item_eval], cov[item_eval])["ap"] if item_eval else np.nan
                        ev1 = syn_stats(r["syn_grid"][key(s + 1)][item_eval], cov[item_eval])["ap"] if item_eval else np.nan
                        ap = traj.get((arm, seed, c), {})
                        outs.append({"arm": arm, "seed": seed, "category": c, "a": a, "s": s,
                                     "shared_with_ID": arm == "FS_B" and s <= 5,
                                     "Y1_primary": math.log(q1) - math.log(q0), "Y1_abs": q1 - q0,
                                     "Y2": gap(S1) - gap(S), "Y3": ev0 - ev1,
                                     "Y4_exposed": (ap[s] - ap[s + 1]) if (s in ap and s + 1 in ap) else np.nan,
                                     "n_nonmembers": len(NM), "n_eval_items": len(item_eval),
                                     "nonmember_q90_s": q0, "nonmember_mean_s": float(S[NM, 0].mean())})
    F = pd.DataFrame(feats)
    O = pd.DataFrame(outs)
    # lagged oracle persistence (previous transition's Y1; eval-only)
    O = O.sort_values(["arm", "seed", "category", "s"]).reset_index(drop=True)
    O["ORACLE_prev_Y1"] = O.groupby(["arm", "seed", "category"])["Y1_primary"].shift(1)
    F.to_csv(C.OUT / "FEATURE_TABLE.csv", index=False)
    O.to_csv(C.OUT / "OUTCOME_TABLE.csv", index=False)
    print("features", F.shape, "outcomes", O.shape)


def _ewma(xs, alpha=0.5):
    if not xs:
        return np.nan
    v = xs[0]
    for x in xs[1:]:
        v = alpha * x + (1 - alpha) * v
    return float(v)


if __name__ == "__main__":
    build()
