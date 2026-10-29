"""NLCAD-06B-WASHOUT: why does REP-05's stage-7 old-normal protection vanish at stage 8? CPU-only, read-only forensics.

Inputs (read only, hash-checked where 06B recorded a hash): DECISION/ANALYSIS_06B, the CL06B run records (BASE,
STAGE7-9 with the 128-step probes and block displacements), STEPSTATS7-9 (per-step attn.proj update statistics), the
CL06B stage checkpoints and the R15 theta_6..theta_9 checkpoints.

Recomputed, because 06B did not save them: the stage-boundary REP bases (each arm's own theta_{s-1}, that stage's
Replay-24 members, tau .97) with the locked closed_loop_06b code (rep_cov, bases), FP32 forward on CPU instead of the
GPU. Validated against the logged ranks and replay-covariance captures, and by the REP-10 identity (alpha = 1 leaves no
stage displacement inside its own basis). Optimizer moments, per-step update vectors and gradients were never saved and
are not reconstructed. No training, no CUDA, no anomaly labels. Writes WASHOUT_06B.json (create-exclusive).

Uncertainty: category-stratified image bootstrap (closed_loop_06b.strat_boot, B = 1e4, seed 6062). It covers image
sampling given one trajectory per arm; it says nothing about trajectory or seed variability.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly.common import require, sha
from exps.nlcad import closed_loop_06b as M
from exps.nlcad import forensic_06a as F
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

ARMS = M.ORDER                                      # ID, REP-05, NORM-05, RAND-05, REP-10, ACQ-05, ORACLE-05
STEPS = [0] + list(range(M.EVERY, M.ITERS, M.EVERY)) + [M.ITERS]
OUT = M.OUT / "WASHOUT_06B.json"
LOCKED = {"DECISION_06B.json": "51989b0a206df02b0481a4cc1ea32bdad05abe7d63a879a82b904fc7baa0e1f3",
          "ANALYSIS_06B.json": "cd82a6aa60b2d1c53355cc300f5db6e4b7b898d06881e571e80cb43a446a4f03",
          "PREREG_06B.json": "b99195e904f65486aec4b091aa5b5e053c9de60091328b370ba597cb7ed87d47"}
BOOT_B, BOOT_SEED = M.BOOT_B, M.BOOT_SEED


# --- inputs ------------------------------------------------------------------------------------------------------

def records():
    for name, h in LOCKED.items():
        require(sha(M.OUT / name) == h, f"{name} changed")
    recs = {a: {s: D.read(M.run_dir(a) / f"STAGE{s}.json") for s in M.STAGES} for a in ARMS}
    base = D.read(M.run_dir("ID") / "BASE.json")
    for a in ARMS:
        require(D.read(M.run_dir(a) / "BASE.json")["theta6_old"] == base["theta6_old"], f"{a}: theta_6 baseline")
        for s in M.STAGES:
            require(recs[a][s]["integrity_pass"], f"{a} {s}: integrity")
            require(sha(D.ROOT / recs[a][s]["stepstats"]) == recs[a][s]["stepstats_sha256"], f"{a} {s}: stepstats")
    return recs, base


def state(recs, a, s):
    """theta_s of arm a (theta_6 is the shared R15 start); hash-checked."""
    if s == M.START or a == "ID":
        return M.r15_state(s)[0]
    rec = recs[a][s]
    path = D.ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"{a} {s}: checkpoint changed")
    return torch.load(path, weights_only=True)["state"]


# --- statistics ----------------------------------------------------------------------------------------------------

def boot(diff, groups):
    b = M.strat_boot(diff, groups)
    return {"point": b["point"], "ci95": b["ci95"]}


def cat_boot(diff, groups):
    """Per category: image mean and within-category percentile 95% CI."""
    rng = np.random.default_rng(BOOT_SEED)
    diff, groups = np.asarray(diff, dtype=np.float64), np.asarray(groups)
    out = {}
    for c in dict.fromkeys(groups.tolist()):
        v = diff[groups == c]
        d = v[rng.integers(0, len(v), (BOOT_B, len(v)))].mean(1)
        out[c] = {"n": int(len(v)), "point": float(v.mean()),
                  "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
    return out


# --- 1. functional decomposition (saved per-image probes) -----------------------------------------------------------

def functional(recs, base):
    groups = np.array(base["groups"])
    n6 = len(groups)
    L6, P6 = np.array(base["theta6_old"]["loss"]), np.array(base["theta6_old"]["p99"])
    g = lambda a, s, i, key="loss", part="old": np.array(recs[a][s]["probes"][str(i)][part][key])
    end = lambda a, s, key="loss": g(a, s, M.ITERS, key)[:n6]
    base6 = {"loss": L6, "p99": P6}
    out = {"n_images": {c: int((groups == c).sum()) for c in dict.fromkeys(groups.tolist())}}

    # consistency: each stage's step-0 probe re-measures the previous stage end on the same parameters
    out["start_equals_previous_end_max_abs"] = {
        a: max(float(np.abs(g(a, s + 1, 0)[:n6] - end(a, s)).max()) for s in (7, 8)) for a in ARMS}

    # stage-end inflation since theta_6 (D_s) and tail (T_s)
    out["D"] = {a: {s: boot(end(a, s) - L6, groups) for s in M.STAGES} for a in ARMS}
    out["T"] = {a: {s: boot(end(a, s, "p99") - P6, groups) for s in M.STAGES} for a in ARMS}
    out["D_per_category"] = {a: {s: M.per_category(end(a, s) - L6, groups) for s in M.STAGES} for a in ARMS}

    # own change over each transition, and paired difference-in-differences against ID and against NORM-05
    out["change"], out["did"], out["did_cat"], out["did_grid_vs_rest"], out["did_tail"] = {}, {}, {}, {}, {}
    out["gap_end"], out["gap_end_cat"] = {}, {}
    rest = groups != "grid"
    for a in ARMS:
        out["change"][a] = {"6->7": boot(end(a, 7) - L6, groups), "7->8": boot(end(a, 8) - end(a, 7), groups),
                            "8->9": boot(end(a, 9) - end(a, 8), groups)}
    for a, ref in [(x, "ID") for x in ARMS[1:]] + [("REP-05", "NORM-05"), ("REP-10", "REP-05")]:
        key = f"{a}~{ref}"
        out["gap_end"][key] = {s: boot(end(a, s) - end(ref, s), groups) for s in M.STAGES}
        out["gap_end_cat"][key] = {s: cat_boot(end(a, s) - end(ref, s), groups) for s in M.STAGES}
        out["did"][key], out["did_cat"][key], out["did_grid_vs_rest"][key], out["did_tail"][key] = {}, {}, {}, {}
        for s in (6, 7, 8):
            prev_a = base6["loss"] if s == 6 else end(a, s)
            prev_r = base6["loss"] if s == 6 else end(ref, s)
            dd = (end(a, s + 1) - prev_a) - (end(ref, s + 1) - prev_r)
            pa = base6["p99"] if s == 6 else end(a, s, "p99")
            pr = base6["p99"] if s == 6 else end(ref, s, "p99")
            dt = (end(a, s + 1, "p99") - pa) - (end(ref, s + 1, "p99") - pr)
            t = f"{s}->{s + 1}"
            out["did"][key][t] = boot(dd, groups)
            out["did_cat"][key][t] = cat_boot(dd, groups)
            out["did_grid_vs_rest"][key][t] = {"grid": cat_boot(dd[~rest], groups[~rest])["grid"],
                                               "rest_macro5": boot(dd[rest], groups[rest]),
                                               "grid_share_of_macro": float(dd[~rest].mean() / 6 /
                                                                            np.mean([dd[groups == c].mean() for c in
                                                                                     dict.fromkeys(groups.tolist())]))}
            out["did_tail"][key][t] = boot(dt, groups)

    # per image: REP-05 vs ID, 7->8 (contribution of each image to the category-macro DiD)
    dd = (end("REP-05", 8) - end("REP-05", 7)) - (end("ID", 8) - end("ID", 7))
    w = np.array([1.0 / (6 * (groups == c).sum()) for c in groups])
    contrib = dd * w
    order = np.argsort(-contrib)
    out["per_image_REP05_ID_7to8"] = {
        "macro": float(contrib.sum()), "n_positive": int((dd > 0).sum()), "n": int(n6),
        "top5_share": float(contrib[order[:5]].sum() / contrib.sum()),
        "top10_share": float(contrib[order[:10]].sum() / contrib.sum()),
        "positive_by_category": {c: int((dd[groups == c] > 0).sum()) for c in dict.fromkeys(groups.tolist())},
        "top10": [{"i": int(i), "category": str(groups[i]), "did": float(dd[i]), "contribution": float(contrib[i]),
                   "L7_ID": float(end("ID", 7)[i]), "L8_ID": float(end("ID", 8)[i]),
                   "L7_REP": float(end("REP-05", 7)[i]), "L8_REP": float(end("REP-05", 8)[i])} for i in order[:10]],
        "spearman_did_vs_stage7_protection": float(np.corrcoef(
            np.argsort(np.argsort(dd)), np.argsort(np.argsort(end("ID", 7) - end("REP-05", 7))))[0, 1])}

    # within-stage trajectories at the 128-step probes: gap to ID, own change from the stage start, output drift
    out["traj"] = {}
    for a in ARMS:
        out["traj"][a] = {}
        for s in M.STAGES:
            rows = []
            for i in STEPS:
                o = g(a, s, i)[:n6]
                row = {"step": i, "D": boot(o - L6, groups)["point"],
                       "own_change": float(np.mean([(o - g(a, s, 0)[:n6])[groups == c].mean()
                                                    for c in dict.fromkeys(groups.tolist())])),
                       "drift_theta6": float(np.mean([np.array(recs[a][s]["probes"][str(i)]["old"]
                                                               ["drift_theta6_O6"])[groups == c].mean()
                                                      for c in dict.fromkeys(groups.tolist())])),
                       "D_grid": float((o - L6)[groups == "grid"].mean()),
                       "D_rest5": float(np.mean([(o - L6)[groups == c].mean() for c in dict.fromkeys(
                           groups.tolist()) if c != "grid"])),
                       "cur_loss": float(np.mean(g(a, s, i, part="cur"))),
                       "rep_loss": float(np.mean(g(a, s, i, part="rep")))}
                if a != "ID":
                    row["gap_to_ID"] = boot(o - g("ID", s, i)[:n6], groups)
                    row["gap_to_ID_grid"] = float((o - g("ID", s, i)[:n6])[groups == "grid"].mean())
                    row["gap_to_ID_rest5"] = float(np.mean([(o - g("ID", s, i)[:n6])[groups == c].mean()
                                                            for c in dict.fromkeys(groups.tolist()) if c != "grid"]))
                    row["rep_gap_to_ID"] = float(np.mean(g(a, s, i, part="rep") - g("ID", s, i, part="rep")))
                rows.append(row)
            out["traj"][a][s] = rows

    # stage-8 wash-out split at the first probe (128 steps: warm-up to the peak lr, before any stage-8 event in REP-05)
    out["did_stage8_split"] = {}
    for a in ARMS[1:]:
        pr = lambda arm, i: g(arm, 8, i)[:n6]
        out["did_stage8_split"][a] = {
            "0-128": boot((pr(a, 128) - pr(a, 0)) - (pr("ID", 128) - pr("ID", 0)), groups),
            "128-667": boot((pr(a, M.ITERS) - pr(a, 128)) - (pr("ID", M.ITERS) - pr("ID", 128)), groups),
            "own_0-128": boot(pr(a, 128) - pr(a, 0), groups)}
    out["did_stage8_split"]["ID_own_0-128"] = boot(g("ID", 8, 128)[:n6] - g("ID", 8, 0)[:n6], groups)

    # reversion: does a deviation from ID at the end of stage s come back during stage s+1? Image level (paired with ID)
    # and arm level (stage-end D against the next stage's own change). Descriptive; one trajectory per arm.
    out["reversion"] = {"image_level": {}, "arm_level": {}}
    for a in ARMS[1:]:
        out["reversion"]["image_level"][a] = {}
        for s in (7, 8):
            gap = end(a, s) - end("ID", s)
            dd = (end(a, s + 1) - end(a, s)) - (end("ID", s + 1) - end("ID", s))
            slope, icpt = np.polyfit(gap, dd, 1)
            out["reversion"]["image_level"][a][f"{s}->{s + 1}"] = {
                "slope": float(slope), "intercept": float(icpt), "pearson": float(np.corrcoef(gap, dd)[0, 1])}
    for s in (7, 8):
        lvl = np.array([out["D"][a][s]["point"] for a in ARMS])
        chg = np.array([out["change"][a][f"{s}->{s + 1}"]["point"] for a in ARMS])
        slope, icpt = np.polyfit(lvl, chg, 1)
        out["reversion"]["arm_level"][f"{s}->{s + 1}"] = {"slope": float(slope), "intercept": float(icpt),
                                                         "pearson": float(np.corrcoef(lvl, chg)[0, 1]),
                                                         "spread_before": float(lvl.max() - lvl.min())}
    out["reversion"]["arm_level"]["spread_stage9"] = float(max(out["D"][a][9]["point"] for a in ARMS) -
                                                          min(out["D"][a][9]["point"] for a in ARMS))

    # leather (task 7) as an old task during stages 8-9: its own inflation since the end of stage 7
    out["leather"] = {}
    for a in ARMS:
        gr8 = np.array(recs[a][8]["groups_old"])
        lth = gr8 == "leather"
        l0 = g(a, 8, 0)[lth]
        out["leather"][a] = {"L_end7": float(l0.mean()),
                             "change_8": float((g(a, 8, M.ITERS)[lth] - l0).mean()),
                             "change_8_9": float((g(a, 9, M.ITERS)[np.array(recs[a][9]["groups_old"]) == "leather"]
                                                  - l0).mean())}
    return out


# --- 2. update statistics (saved STEPSTATS, windows, block displacement) ------------------------------------------------

def updates(recs):
    k = {c: i for i, c in enumerate(M.COLS)}
    out = {}
    for a in ARMS:
        kind = M.ARMS[a][0]
        out[a] = {}
        for s in M.STAGES:
            tab = torch.load(D.ROOT / recs[a][s]["stepstats"], weights_only=True)["table"].numpy()
            require(tab.shape == (M.ITERS, len(M.LAYERS), len(M.COLS)), f"{a} {s}: stepstats shape {tab.shape}")
            col = lambda c, sl=slice(None): tab[sl, :, k[c]]
            eff_rep_known = kind in ("id", "rep", "norm")       # acting basis == own REP basis: real_res is in REP
            wins = []
            for a0 in range(0, M.ITERS, M.EVERY):
                sl = slice(a0, a0 + M.EVERY)
                tot, real = col("e_tot", sl).sum(), col("e_real", sl).sum()
                wins.append({"steps": [a0, min(a0 + M.EVERY, M.ITERS)], "lr_mean": float(col("lr", sl)[:, 0].mean()),
                             "raw_energy": float(tot), "eff_energy": float(real),
                             "raw_in_rep": float(col("e_rep", sl).sum() / tot),
                             "raw_in_act": float(col("e_act", sl).sum() / tot),
                             "eff_in_act": float(col("real_res", sl).sum() / real),
                             "eff_in_rep": float(col("real_res", sl).sum() / real) if eff_rep_known else None,
                             "retained_energy": float(real / tot)})
            tot, real = col("e_tot").sum(0), col("e_real").sum(0)
            out[a][s] = {
                "raw_energy": float(tot.sum()), "eff_energy": float(real.sum()),
                "raw_in_rep": float(col("e_rep").sum() / tot.sum()),
                "raw_in_act": float(col("e_act").sum() / tot.sum()),
                "eff_in_act": float(col("real_res").sum() / real.sum()),
                "eff_in_rep": float(col("real_res").sum() / real.sum()) if eff_rep_known else None,
                "eff_in_rep_energy": float(col("real_res").sum()) if eff_rep_known else None,
                "retained_energy": float(real.sum() / tot.sum()),
                "removed_vector_energy": float(col("removed").sum() / tot.sum()),
                "decay_energy": float(col("e_decay").sum() / tot.sum()),
                "per_layer": {l: {"raw_in_rep": float(col("e_rep")[:, j].sum() / tot[j]),
                                  "eff_energy": float(real[j]), "path_length": float(np.sqrt(col("e_real")[:, j]).sum()),
                                  "eff_in_rep_energy": float(col("real_res")[:, j].sum()) if eff_rep_known else None}
                              for j, l in enumerate(M.LAYERS)},
                "windows": wins,
                "block_displacement": recs[a][s]["block_displacement"]}
    return out


# --- 2b. transient instability events ------------------------------------------------------------------------------

EVENT_X = 100.0           # an event step: post-warm-up attn.proj raw update energy > 100 x that stage's median (100-667)


def events(recs, fn):
    """Per arm and stage: training-loss peak (50-step means), attn.proj update-energy events (STEPSTATS) and the O6
    transient at the 128-step probes. Plus the same training-loss scan of the three historical R15 seeds (RH-01b)."""
    k = {c: i for i, c in enumerate(M.COLS)}
    out = {"threshold_x_median": EVENT_X, "arms": {}, "r15_seeds": {}}

    def loss_peak(c):
        pk = max(range(2, len(c)), key=lambda i: c[i])
        return {"ratio": float(c[pk] / min(c[1:pk])), "window_start": 50 * pk}

    for a in ARMS:
        out["arms"][a] = {}
        for s in M.STAGES:
            tab = torch.load(D.ROOT / recs[a][s]["stepstats"], weights_only=True)["table"].numpy()
            E, Er = tab[..., k["e_tot"]].sum(1), tab[..., k["e_real"]].sum(1)
            med = float(np.median(E[100:]))
            ev = np.zeros(len(E), bool)
            ev[100:] = E[100:] > EVENT_X * med
            idx = np.where(ev)[0]
            traj = fn["traj"][a][s]
            rec = {"train_loss": loss_peak(recs[a][s]["losses_every50"]),
                   "energy_peak_x_median": float(E[100:].max() / med), "event_steps": int(ev.sum()),
                   "event_first_last": [int(idx[0]), int(idx[-1])] if len(idx) else None,
                   "event_share_raw_energy": float(E[ev].sum() / E.sum()),
                   "event_share_eff_energy": float(Er[ev].sum() / Er.sum()),
                   "o6_peak_probe": max((r["D"], r["step"]) for r in traj),
                   "o6_transient_above_end": float(max(r["D"] for r in traj) - traj[-1]["D"]),
                   "o6_D_end": traj[-1]["D"]}
            if ev.any():
                rec["raw_in_rep_event"] = float(tab[ev][..., k["e_rep"]].sum() / tab[ev][..., k["e_tot"]].sum())
                rec["raw_in_rep_rest"] = float(tab[~ev][..., k["e_rep"]].sum() / tab[~ev][..., k["e_tot"]].sum())
            out["arms"][a][s] = rec
    for path in sorted((C.OUT / "runs").glob("R15_seed*")):
        out["r15_seeds"][path.name] = {s: loss_peak(D.read(path / f"STAGE{s}.json")["loss_curve_every50"])
                                      for s in range(2, 16)}
    # stage 7: stage-end O6 inflation against the size of the O6 transient at the step-256 probe, across arms
    sev = np.array([fn["traj"][a][7][2]["D"] for a in ARMS])
    d7 = np.array([fn["D"][a][7]["point"] for a in ARMS])
    slope, icpt = np.polyfit(sev, d7, 1)
    out["stage7_end_vs_step256"] = {"slope": float(slope), "intercept": float(icpt),
                                    "pearson": float(np.corrcoef(sev, d7)[0, 1]),
                                    "points": {a: [float(x), float(y)] for a, x, y in zip(ARMS, sev, d7)}}
    return out


# --- 3. parameters: stage displacement by component, arm-vs-ID gaps, recomputed REP bases -----------------------------

def group_of(name):
    if name.startswith("bottleneck"):
        return "bottleneck"
    part = name.split(".", 2)[2]                          # e.g. attn.proj.weight, mlp.fc1.bias, norm1.weight
    if part == "attn.proj.weight":
        return "attn.proj.weight"
    if part.startswith("attn.proj"):
        return "attn.proj.bias"
    if part.startswith("attn.qkv"):
        return "attn.qkv"
    if part.startswith("mlp"):
        return "mlp"
    return "norm"


def group_norms(sd):
    acc = {}
    for kk, v in sd.items():
        acc[group_of(kk)] = acc.get(group_of(kk), 0.0) + float(v.double().pow(2).sum())
        if group_of(kk) != "bottleneck":
            b = "block." + kk.split(".")[1]
            acc[b] = acc.get(b, 0.0) + float(v.double().pow(2).sum())
    return {g: float(np.sqrt(x)) for g, x in acc.items()}


def diff(sa, sb):
    return {kk: sa[kk] - sb[kk] for kk in sa}


def proj_weights(sd):
    return {l: sd[l + ".weight"].double().clone() for l in M.LAYERS}


def rep_basis(sd, s, n):
    """The REP basis of stage s from theta_{s-1} = sd (locked code: Replay-24 members of stage s, tau .97), on CPU."""
    dec = O.Decoder(768, 12)
    dec.load_state_dict(sd)
    old = list(C.STREAM[:s - 1])
    st = {"old": old, "idx": C.buffer_index(M.SEED, n, old)}
    cov = M.rep_cov(dec, st, "cpu")
    U = M.bases(cov)
    keys = [(d, int(i)) for d in old for i in st["idx"][d]]
    return U, M.capture(cov, U), keys


def fro2(x):
    return float(sum(v.pow(2).sum() for v in x.values()))


def inner(x, y):
    return float(sum((x[l] * y[l]).sum() for l in x))


def params(recs):
    NUM.use(True)
    n = R1.n_train(RB.verify_lock())
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    W = {a: {6: None} for a in ARMS}
    out = {"displacement": {}, "gap_to_ID": {}, "bases": {}, "in_rep": {}}
    sd6 = state(recs, "ID", 6)
    W6 = proj_weights(sd6)
    for a in ARMS:
        W[a][6] = W6
    # stage displacement per component and arm-vs-ID gap, one stage at a time (bounded RAM)
    for s in M.STAGES:
        id_prev, id_now = state(recs, "ID", s - 1), state(recs, "ID", s)
        out["displacement"].setdefault("ID", {})[s] = {"delta": group_norms(diff(id_now, id_prev)),
                                                      "ref": group_norms(id_prev)}
        W["ID"][s] = proj_weights(id_now)
        for a in ARMS[1:]:
            prev = state(recs, a, s - 1) if s > 7 else id_prev
            now = state(recs, a, s)
            out["displacement"].setdefault(a, {})[s] = {"delta": group_norms(diff(now, prev)),
                                                       "ref": group_norms(prev)}
            out["gap_to_ID"].setdefault(a, {})[s] = group_norms(diff(now, id_now))
            W[a][s] = proj_weights(now)
            del prev, now
        del id_prev, id_now

    # are the four covariance-subspace arms independent observations at stage 7? attn.proj distance between them
    sub = ["REP-05", "REP-10", "ACQ-05", "ORACLE-05"]
    dist = lambda x, y: float(np.sqrt(sum((W[x][7][l] - W[y][7][l]).pow(2).sum() for l in M.LAYERS)))
    out["subspace_arms_theta7_attn_proj_distance"] = {
        "pairs": {f"{x}~{y}": dist(x, y) for i, x in enumerate(sub) for y in sub[i + 1:]},
        "to_ID": {x: dist(x, "ID") for x in sub}}

    # REP bases: stage 7 from the shared theta_6; stages 8-9 from each arm's own theta_7, theta_8
    U = {a: {} for a in ARMS}
    U7, cap7, keys7 = rep_basis(sd6, 7, n)
    del sd6
    for a in ARMS:
        U[a][7] = U7
    val = {"stage7": {"ranks": {l: int(U7[l].shape[1]) for l in M.LAYERS}, "capture": cap7}}
    for a in ARMS:
        for s in (8, 9):
            sd = state(recs, a, s - 1)
            U[a][s], cap, keys = rep_basis(sd, s, n)
            del sd
            val[f"{a}_stage{s}"] = {"ranks": {l: int(U[a][s][l].shape[1]) for l in M.LAYERS}, "capture": cap}
            require(keys == [tuple(x) for x in recs[a][s]["replay_keys"]], f"{a} {s}: replay keys differ")
    require(keys7 == [tuple(x) for x in recs["ID"][7]["replay_keys"]], "stage 7 replay keys differ")
    # validation against the GPU run: logged ranks and captures, and the REP-10 identity
    checks = {}
    for a in ARMS:
        for s in M.STAGES:
            mem = recs[a][s]["memory"]
            r_cpu = {l: int(U[a][s][l].shape[1]) for l in M.LAYERS}
            checks[f"{a}_{s}"] = {"max_abs_rank_diff": max(abs(r_cpu[l] - mem["rep_rank"][l]) for l in M.LAYERS),
                                  "total_rank_cpu": sum(r_cpu.values()), "total_rank_gpu": sum(mem["rep_rank"].values())}
            if M.ARMS[a][0] in ("id", "rep", "norm"):
                cap = val["stage7"]["capture"] if s == 7 else val[f"{a}_stage{s}"]["capture"]
                checks[f"{a}_{s}"]["max_abs_capture_diff"] = max(abs(cap[l] - mem["act_capture_of_replay_cov"][l])
                                                                  for l in M.LAYERS)
    out["bases"] = {"validation": checks, "detail": val}

    P = lambda X, Uu: {l: X[l] @ Uu[l] for l in M.LAYERS}           # coordinates inside the basis
    PP = lambda X, Uu: {l: X[l] @ Uu[l] @ Uu[l].T for l in M.LAYERS}  # component inside the basis
    dW = {a: {s: {l: W[a][s][l] - W[a][s - 1][l] for l in M.LAYERS} for s in M.STAGES} for a in ARMS}
    res = {}
    for a in ARMS:
        res[a] = {}
        for s in M.STAGES:
            e = fro2(dW[a][s])
            own = fro2(P(dW[a][s], U[a][s]))
            u7 = fro2(P(dW[a][s], U7))
            res[a][s] = {"disp_energy": e, "frac_in_own_rep": own / e, "frac_in_U7": u7 / e,
                         "norm_in_own_rep": float(np.sqrt(own)), "norm_in_U7": float(np.sqrt(u7)),
                         "per_layer_frac_in_own_rep": {l: float((dW[a][s][l] @ U[a][s][l]).pow(2).sum()
                                                                / dW[a][s][l].pow(2).sum()) for l in M.LAYERS}}
    out["in_rep"] = res
    # in-subspace displacement relative to ID (stage 7: identical basis U7 for every arm)
    out["kappa7"] = {a: res[a][7]["norm_in_U7"] / res["ID"][7]["norm_in_U7"] for a in ARMS}
    out["kappa_own"] = {a: {s: res[a][s]["norm_in_own_rep"] / res["ID"][s]["norm_in_own_rep"] for s in M.STAGES}
                        for a in ARMS}

    # basis refresh: how much of the stage-7 protected subspace is still protected at stages 8 and 9
    ov = lambda A, B: float(sum((A[l].T @ B[l]).pow(2).sum() for l in M.LAYERS) / sum(A[l].shape[1] for l in M.LAYERS))
    out["basis_overlap"] = {a: {"U7_in_U8": ov(U7, U[a][8]), "U8_in_U7": ov(U[a][8], U7),
                                "U8_in_U9": ov(U[a][8], U[a][9]), "U7_in_U9": ov(U7, U[a][9])} for a in ARMS}
    out["basis_overlap"]["REP-05_vs_ID"] = {"U8": ov(U["REP-05"][8], U["ID"][8]), "U9": ov(U["REP-05"][9], U["ID"][9])}

    # catch-up along the stage-7 withheld component. For REP-05 (alpha .5, U7 fixed during stage 7) the withheld part
    # equals the effective in-U7 stage displacement: w7 = dW7 P7 (both are 0.5 * sum_t raw_t P7).
    out["catchup"] = {}
    for a in ARMS[1:]:
        w7 = PP(dW[a][7], U7)
        gaps = {s: {l: W[a][s][l] - W["ID"][s][l] for l in M.LAYERS} for s in M.STAGES}
        row = {}
        for s in M.STAGES:
            gp = PP(gaps[s], U7)
            row[s] = {"gap_in_U7": float(np.sqrt(fro2(gp))),
                      "gap_perp_U7": float(np.sqrt(fro2(gaps[s]) - fro2(gp))),
                      "gap_total": float(np.sqrt(fro2(gaps[s])))}
        id7 = PP(dW["ID"][7], U7)
        row["gap7_in_U7_dir"] = {s: inner(PP(gaps[s], U7), id7) / fro2(id7) for s in M.STAGES}
        if a == "REP-05":
            row["withheld_norm"] = float(np.sqrt(fro2(w7)))
            for s in (8, 9):
                c = PP(dW[a][s], U7)
                c_id = PP(dW["ID"][s], U7)
                row[f"cos_stage{s}_U7_with_withheld"] = inner(c, w7) / np.sqrt(fro2(c) * fro2(w7))
                row[f"cos_stage{s}_U7_ID_with_own7"] = inner(c_id, id7) / np.sqrt(fro2(c_id) * fro2(id7))
        out["catchup"][a] = row
    return out


def strkeys(x):
    """JSON keys as strings (stage numbers are ints in places; D.write_json sorts keys)."""
    if isinstance(x, dict):
        return {str(k): strkeys(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strkeys(v) for v in x]
    return x


def main():
    require(not OUT.exists(), f"exists: {OUT}")
    recs, base = records()
    fn = functional(recs, base)
    up = updates(recs)
    ev = events(recs, fn)
    pa = params(recs)
    D.write_json(OUT, strkeys({"task": "agents/tasks/task_nlcad_06b_washout_forensics.md", "utc": M.DR.now(),
                       "inputs": LOCKED, "source_sha256": sha(D.ROOT / "exps/nlcad/washout_06b.py"),
                       "functional": fn, "updates": up, "events": ev, "params": pa,
                       "not_available": ["optimizer moments (never saved; not reconstructed)",
                                         "per-step update vectors and gradients (energies only)",
                                         "06B GPU bases (recomputed on CPU here, validated)",
                                         "functional effect of grafting blocks (would be a new causal experiment)"]}))
    print("written", OUT, sha(OUT))


if __name__ == "__main__":
    main()
