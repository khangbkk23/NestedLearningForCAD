# exps/nlcad/analyze_02.py
"""NLCAD-02 analysis. Identity = the exact NLCAD-01 identity (RH-01b R15 seed 2711, stages 1-15).

  python -m exps.nlcad.analyze_02 dynamics   # CPU: controller dynamics, task dependence, parity, integrity, penalty
  python -m exps.nlcad.analyze_02 gpu        # forward only: buffer coverage at stage 15, stage-end synthetic Q2/Q3
  python -m exps.nlcad.analyze_02 compare    # CPU: paired arm-vs-arm native AP on the identical P1 draws

`gpu` and `compare` require LOCK_02 (written before any NLCAD-02 score). The post-hoc synthetic set is fixed: every
bank item whose host category is in the development cohort (no real labels; nothing is selected from it).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import analysis as DA
from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as RO
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import analyze_c as A
from exps.nlcad import bank as BK
from exps.nlcad import openloop as OL
from exps.nlcad import run_02 as R2
from exps.nlcad import stage_d as SD
from exps.nlcad.controller import BLOCKS, BlockController
from exps.nlcad.objectives import ranking_loss
from exps.rh01b import common as C
from exps.rh01b.analysis import ACQ, FINAL
from exps.task_p.p2.metrics import exact_ranking

OUT = R2.OUT
STAGE_C = "N_per_task_seed2711_from6"                                # NLCAD-01 arm N (lambda = 1e-3), for context
PAIRS = [("N1", "ID"), ("N1", "F1"), ("N1", "RND"), ("F1", "ID"), ("RND", "ID"), ("F1", "RND"), ("N", "ID")]
TAGS = {"N1": R2.tag("N1"), "F1": R2.tag("F1"), "RND": R2.tag("RND"), "N": STAGE_C}


def verify_lock():
    R2.verify_prereg()
    lk = D.read(R2.LOCK)
    for s, h in lk["sources"].items():
        require(sha(D.ROOT / s) == h, f"post-lock change: {s}")
    for a, h in lk["profiles"].items():
        require(sha(R2.PROFILE[a]) == h, f"profile changed: {a}")
    return lk


# ----------------------------------------------------------------------------------------------- controller dynamics
def centred(a):
    la = np.log(np.asarray(a, dtype=np.float64))
    return la - la.mean(axis=-1, keepdims=True)


def task_dependence(ev_by_stage, f1=None):
    """Stage-mean profiles (centred log a of the multipliers applied during the stage), their cross-stage cosines,
    within-stage drift (first to last applied profile) and the distance of each stage mean from F1."""
    means, drift = {}, {}
    for s, ev in ev_by_stage.items():
        used = centred([[e["a_used"][b] for b in BLOCKS] for e in ev])
        means[s], drift[s] = used.mean(0), float(np.linalg.norm(used[-1] - used[0]))
    M = np.array(list(means.values()))
    norms = np.linalg.norm(M, axis=1)
    cos = (M @ M.T) / (np.outer(norms, norms) + 1e-300)
    iu = np.triu_indices(len(M), 1)
    out = {"stage_mean_profile_a": {s: dict(zip(BLOCKS, np.exp(m - np.log(np.exp(m).mean())).tolist()))
                                    for s, m in means.items()},
           "stage_mean_centred_log_a_norm": dict(zip(means, norms.tolist())),
           "cross_stage_cosine": {"min": float(cos[iu].min()), "mean": float(cos[iu].mean())} if len(M) > 1 else None,
           "within_stage_drift_centred_log_a": drift}
    if f1 is not None:
        c1 = centred([f1[b] for b in BLOCKS])
        out["distance_from_F1_centred_log_a"] = {s: float(np.linalg.norm(m - c1)) for s, m in means.items()}
        out["F1_centred_log_a_norm"] = float(np.linalg.norm(c1))
    return out


def parity(arm, ev_by_stage):
    """The multipliers applied at every slow event equal the controller state that produced them: for a learning arm
    the post-step multipliers of the previous event (identity before the first), for a fixed arm the profile."""
    ev = [e for s in sorted(ev_by_stage) for e in ev_by_stage[s]]
    if arm in ("N1", "N"):
        expect = [{b: 1.0 for b in BLOCKS}] + [dict(zip(BLOCKS, e["a"])) for e in ev[:-1]]
    else:
        prof = D.read(R2.PROFILE[arm])["phi"]
        a = dict(zip(BLOCKS, R2.multipliers([prof[b] for b in BLOCKS]).tolist()))
        expect = [a] * len(ev)
    bad = [i for i, (e, x) in enumerate(zip(ev, expect)) if any(e["a_used"][b] != x[b] for b in BLOCKS)]
    return {"events": len(ev), "mismatches": len(bad), "first_mismatch": bad[:5],
            "nonfinite_events": sum(not e["finite"] for e in ev)}


def dynamics_of(arm):
    tag = TAGS[arm]
    cfg, rows = A.stage_records(tag, R2.START)
    ev_by_stage = {s: r["events"] for s, r in rows.items()}
    f1 = D.read(R2.PROFILE["F1"])["phi"] if R2.PROFILE["F1"].exists() else None
    f1a = None if f1 is None else dict(zip(BLOCKS, R2.multipliers([f1[b] for b in BLOCKS]).tolist()))
    learning = arm in ("N1", "N")
    ctl = A.controller_learning(rows if learning else _as_learning(rows))
    if not learning:                                                 # phi/a/saturation are not recorded for fixed arms
        ctl = {s: {k: v for k, v in c.items() if k not in FIXED_ONLY_LEARNING} for s, c in ctl.items()}
    out = {"config": cfg, "stages": sorted(rows), "controller": ctl,
           "task_dependence": task_dependence(ev_by_stage, f1a),
           "parity": parity(arm, ev_by_stage), "heldout_vs_identity": A.heldout_vs_identity(rows, R2.SEED),
           "integrity": A.integrity(rows, R2.SEED)}
    if learning:
        a = np.array([e["a"] for s in sorted(rows) for e in rows[s]["events"]])
        sat = np.array([e["saturation"] for s in sorted(rows) for e in rows[s]["events"]])
        out["max_abs_a_minus_1"] = float(np.abs(a - 1).max())
        out["saturation"] = {"max": float(sat.max()), "mean": float(sat.mean())}
        out["penalty_vs_signal"] = OL.decompose(OL.events(tag, R2.START), cfg["wd_phi"])
    return out


FIXED_ONLY_LEARNING = ("a_end", "phi_end", "max_abs_a_minus_1", "saturation_max")


def _as_learning(rows):
    """Fixed-arm events in the shape controller_learning reads (placeholders for the fields it drops afterwards)."""
    return {s: {"stage": r["stage"], "events": [dict(e, a=[e["a_used"][b] for b in BLOCKS], phi=[0.0] * len(BLOCKS),
                                                     saturation=0.0) for e in r["events"]]}
            for s, r in rows.items()}


def dynamics():
    R2.verify_prereg()
    res = {arm: dynamics_of(arm) for arm in ("N1", "F1", "RND", "N") if arm == "N" or R2.complete(arm)}
    res["persistent_bytes"] = BlockController().persistent_bytes()
    D.write_json(OUT / "ANALYSIS_02.json", res)
    for arm, r in res.items():
        if isinstance(r, dict) and "parity" in r:
            print(arm, "parity", r["parity"], "max|a-1|", r.get("max_abs_a_minus_1"), "sat", r.get("saturation"),
                  "cos", r["task_dependence"]["cross_stage_cosine"], flush=True)


# ----------------------------------------------------------------------------------------------- forward-only diagnostics
@torch.no_grad()
def synthetic(dec, items, device="cuda", micro=8):
    """Patch-level synthetic AP/AUROC (positives coverage >= 0.5, negatives coverage == 0, partial excluded) and the
    ranking loss with a fixed negative-sampling generator, FP32 forward as in Stage D."""
    grids = []
    for k in range(0, len(items), micro):
        x = torch.stack([it["features"] for it in items[k:k + micro]]).float().to(device)
        en, de = dec(x[:, 0], x[:, 1])
        grids.append(RO.grid(en, de).cpu())
    s = torch.cat(grids)
    cov = torch.stack([it["coverage"] for it in items]).float()
    pos, sel = cov >= 0.5, (cov >= 0.5) | (cov == 0)
    m = exact_ranking(s[sel].double(), pos[sel])
    rl = ranking_loss(s, cov, gen=torch.Generator().manual_seed(0))
    return {"patch_ap": m["pixel_ap"], "patch_auroc": m["pixel_auroc"], "ranking_loss": float(rl),
            "items": len(items), "positive_patches": m["positive_pixels"]}


def synthetic_set(bank):
    return {(g, c): [it for it in bank[g] if it["host"][0] == c] for g in BK.GENERATORS for c in C.EVAL}


def gpu():
    verify_lock()
    DR.configure()
    target = OUT / "SYN_EVAL_02.json"
    require(not target.exists(), "one-shot")
    bank, _ = BK.load_bank(R2.SEED)
    sets = synthetic_set(bank)
    res = {"set": {f"{g}/{c}": len(v) for (g, c), v in sets.items()}, "arms": {}, "coverage_stage15": {}}
    cks = {arm: SD.checkpoints(TAGS[arm])[1] for arm in ("N1", "F1", "RND", "N")}
    cks["ID"] = [x for x in SD.checkpoints(TAGS["N1"], identity=True)[1] if x[2] >= R2.START]
    for arm, rows in cks.items():
        res["arms"][arm] = {}
        for _, _, s, _, path, _ in rows:
            dec = O.Decoder(768, 12).cuda().eval()
            dec.load_state_dict(torch.load(path, weights_only=False)["state"])
            res["arms"][arm][s] = {f"{g}/{c}": synthetic(dec, v) for (g, c), v in sets.items()}
            del dec
        print("synthetic", arm, {s: round(np.mean([v["patch_ap"] for v in r.values()]), 4)
                                 for s, r in res["arms"][arm].items()}, flush=True)
    for arm in ("N1", "F1", "RND"):
        res["coverage_stage15"][arm] = A.coverage(TAGS[arm], FINAL, R2.SEED)
    D.write_json(target, res)


# ----------------------------------------------------------------------------------------------- paired native AP
def load_points():
    """Points and replicates per arm; the identity re-scored in each NLCAD-02 arm directory must equal NLCAD-01's."""
    P, B = {}, {}
    ref = D.read(SD.d(STAGE_C) / "METRICS_identity.json")["points"]
    ref_b = D.read(SD.d(STAGE_C) / "REPLICATES_identity.json")
    for arm, tag in TAGS.items():
        P[arm] = D.read(SD.d(tag) / "METRICS_nl.json")["points"]
        B[arm] = D.read(SD.d(tag) / "REPLICATES_nl.json")
        if arm != "N":
            require(D.read(SD.d(tag) / "METRICS_identity.json")["points"] == ref, f"identity not reproduced in {arm}")
            require(D.read(SD.d(tag) / "REPLICATES_identity.json") == ref_b, f"identity replicates differ in {arm}")
    P["ID"], B["ID"] = ref, ref_b
    return P, B


def key(arm, s, c, ro="RD"):
    return f"{'ID' if arm == 'ID' else 'NL'}/{R2.SEED}/{s}/{c}/{ro}"


def pair(P, B, x, y):
    lo = lambda arm, c: np.array(B[arm][key(arm, FINAL, c)]["ap_low"])
    hi = lambda arm, c: np.array(B[arm][key(arm, FINAL, c)]["ap_high"])
    ap = lambda arm, s, c, ro="RD": P[arm][key(arm, s, c, ro)]["pixel_ap"]
    cats = {}
    for c in C.EVAL:
        acq = ap("ID", ACQ[c], c)
        cats[c] = {"final": [ap(x, FINAL, c), ap(y, FINAL, c)], "diff": ap(x, FINAL, c) - ap(y, FINAL, c),
                   "diff_ci95_images": DA.diff_ci(lo(x, c), hi(x, c), lo(y, c), hi(y, c)),
                   "forgetting": [acq - ap(x, FINAL, c), acq - ap(y, FINAL, c)],
                   "trajectory_diff": {s: ap(x, s, c) - ap(y, s, c) for s in range(R2.START + 1, FINAL + 1)},
                   "image_auroc_final": [P[x][key(x, FINAL, c)]["image_auroc"], P[y][key(y, FINAL, c)]["image_auroc"]],
                   "RP1_final": [ap(x, FINAL, c, "RP1"), ap(y, FINAL, c, "RP1")]}
        cats[c]["trajectory_mean_diff"] = float(np.mean(list(cats[c]["trajectory_diff"].values())))
    mean = lambda f: np.mean([f(c) for c in C.EVAL], axis=0)
    traj = {s: float(np.mean([cats[c]["trajectory_diff"][s] for c in C.EVAL])) for s in range(R2.START + 1, FINAL + 1)}
    pooled = [P[a][f"{'ID' if a == 'ID' else 'NL'}/{R2.SEED}/{FINAL}/pooled/RD"]["pixel_ap"] for a in (x, y)]
    macro = {"final": [float(mean(lambda c: ap(x, FINAL, c))), float(mean(lambda c: ap(y, FINAL, c)))],
             "diff_ci95_images": DA.diff_ci(mean(lambda c: lo(x, c)), mean(lambda c: hi(x, c)),
                                            mean(lambda c: lo(y, c)), mean(lambda c: hi(y, c))),
             "pooled": pooled, "pooled_diff": pooled[0] - pooled[1],
             "trajectory_diff": traj, "trajectory_mean_diff": float(np.mean(list(traj.values()))),
             "Fbar": [float(mean(lambda c: cats[c]["forgetting"][0])), float(mean(lambda c: cats[c]["forgetting"][1]))]}
    macro["diff"] = macro["final"][0] - macro["final"][1]
    return {"categories": cats, "macro": macro}


def compare():
    verify_lock()
    target = OUT / "COMPARE_02.json"
    require(not target.exists(), "one-shot")
    P, B = load_points()
    res = {f"{x}-{y}": pair(P, B, x, y) for x, y in PAIRS}
    n1id, n1r = res["N1-ID"]["macro"], res["N1-RND"]["macro"]
    res["reading"] = {"modest_consistent_positive": bool(n1id["diff"] > 0 and n1id["pooled_diff"] > 0
                                                         and n1id["trajectory_mean_diff"] > 0 and n1r["diff"] > 0),
                      "rule": D.read(R2.PREREG)["reading"],
                      "note": "one seed: descriptive; image-bootstrap intervals are not training-seed variability"}
    D.write_json(target, res)
    for k, r in res.items():
        if k != "reading":
            m = r["macro"]
            print(k, "final", [round(v, 4) for v in m["final"]], "diff", round(m["diff"], 4),
                  "ci", [round(v, 4) for v in m["diff_ci95_images"]], "pooled", round(m["pooled_diff"], 4),
                  "traj", round(m["trajectory_mean_diff"], 4), flush=True)
    print("reading", res["reading"]["modest_consistent_positive"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dynamics", "gpu", "compare"))
    a = ap.parse_args()
    {"dynamics": dynamics, "gpu": gpu, "compare": compare}[a.stage]()


if __name__ == "__main__":
    main()
