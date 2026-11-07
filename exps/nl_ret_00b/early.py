"""NL-RET-00B addendum (owner-authorised, about 10 GPU-min): full-test R15 maps at stages 2-6 and the complete
stage-by-stage native trajectory.

  lock     EARLY_LOCK.json: source hashes, authorisation, cells (before any prediction or new mask statistic)
  predict  label-free R_D maps for every category acquired by stage s, s = 1..6, from the SHA-verified R15
           checkpoints (stage == acquisition cells reproduce the sealed M1 acq maps -> label-free parity)
  seal     EARLY_SEAL.json: completeness, hashes, acq parity (max |diff| <= 1e-3, the M1 rule)
  traj     CPU: exact native AP + positive/negative factorial for every consecutive stage transition, every
           category, 3 seeds, ID (stages a(c)..15) and FS_B (shared R15 states up to stage 6, own states 7..15)
           -> TRAJECTORY.jsonl (resumable) -> STAGE_TRAJECTORY.json

  PYTHONPATH=$PWD python -m exps.nl_ret_00b.early <cmd>
"""
from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import torch

from exps.cadfinal import common as K
from exps.nl_ret_00b import core as X
from exps.nl_ret_00b import decomp as D
from exps.task_p.p0_normal_support import require

ROOT = K.ROOT
EP = D.OUT / "early_predictions"
ELOCK, ESEAL, TRAJ, TRAJ_OUT = (D.OUT / "EARLY_LOCK.json", D.OUT / "EARLY_SEAL.json", D.OUT / "TRAJECTORY.jsonl",
                                D.OUT / "STAGE_TRAJECTORY.json")
SRC = ["exps/nl_ret_00b/early.py", "exps/nl_align_02/m.py", "exps/dinomaly/readout.py", "exps/dinomaly/encoder.py",
       "exps/cadfinal/dino.py", "exps/nl_align_01/g1.py"]


def cells():
    return [(s, c) for s in range(1, 7) for c in D.ORDER[:s]]


def epath(seed, s, c):
    return EP / f"seed{seed}" / f"ID_s{s}_{c}.pt"


def lock():
    require(not ELOCK.exists(), "exists")
    K.write_json(ELOCK, {"authorisation": "owner, 2026-11-05: 'Cho phép. Chạy xong cho đầy đủ' (about 10 GPU-min "
                                          "label-free inference to fill the stage 2-6 full-test gap)",
                         "sources": {s: K.sha(ROOT / s) for s in SRC}, "cells_per_seed": cells(),
                         "parity_rule": "stage == acquisition cells vs sealed M1 acq maps, max |diff| <= 1e-3",
                         "trajectory_rule": "same exact streaming AP and Shapley factorial as decomp.py (frozen "
                                            "AUDIT_LOCK); transitions are descriptive, no new decision rule",
                         "utc": K.now()})


@torch.no_grad()
def predict():
    from exps.dinomaly import encoder as E, readout as R
    from exps.nl_align_02 import m as MM
    lk = K.read(ELOCK)
    for s, h in lk["sources"].items():
        require(K.sha(ROOT / s) == h, f"source changed {s}")
    enc = MM._setup()
    ids = MM.ids_()
    root = ROOT / "data/mvtec"
    for seed in D.SEEDS:
        (EP / f"seed{seed}").mkdir(parents=True, exist_ok=True)
        decs = {s: MM._decoder("ID", seed, s) for s in range(1, 7)}
        for j, t in enumerate(ids["tasks"][:6]):
            c = t["task"]
            mine = [s for s, cc in cells() if cc == c]
            if all(epath(seed, s, c).exists() for s in mine):
                continue
            acc = {s: ([], []) for s in mine}
            for i in range(0, len(t["test"]), 8):
                x = torch.stack([E.preprocess(root / r["relative_path"]) for r in t["test"][i:i + 8]]).cuda()
                f = enc.groups(x).float()
                for s in mine:
                    en, de = decs[s][0](f[:, 0], f[:, 1])
                    m_, im = R.dinomaly(en, de)
                    acc[s][0].append(m_.cpu()); acc[s][1].append(im.cpu())
            for s in mine:
                torch.save({"s256": torch.cat(acc[s][0]), "image_RD": torch.cat(acc[s][1]),
                            "paths": [r["relative_path"] for r in t["test"]], "arm": "ID", "seed": seed, "stage": s,
                            "category": c, "checkpoint_sha256": decs[s][1], "utc": K.now()}, epath(seed, s, c))
            print("predict", seed, c, mine, flush=True)
        del decs
        torch.cuda.empty_cache()


def seal():
    require(not ESEAL.exists(), "exists")
    files, par = {}, {}
    for seed in D.SEEDS:
        for s, c in cells():
            p = epath(seed, s, c)
            require(p.exists(), f"missing {p}")
            files[str(p.relative_to(D.OUT))] = K.sha(p)
            if s == D.stage_of(c):
                a = torch.load(p, weights_only=True)
                ref = torch.load(D.M1DIR / f"DINOMALY_R_seed{seed}" / f"acq_{s - 1:02d}.pt", weights_only=True)
                require(a["paths"] == ref["paths"], "identity")
                par[f"{seed}|{c}"] = max(float((a["s256"] - ref["s256"]).abs().max()),
                                         float((a["image_RD"] - ref["image_RD"]).abs().max()))
    worst = max(par.values())
    K.write_json(ESEAL, {"files": files, "n": len(files), "acq_parity": par, "worst": worst, "pass": worst <= 1e-3,
                         "utc": K.now()})
    print("sealed", len(files), "acq parity worst", worst)
    require(worst <= 1e-3, "acq parity failure")


def state_path(arm, seed, c, s):
    j = D.stage_of(c)
    require(s >= j, "pre-acquisition")
    if s == j:
        return D.src(arm, seed, c, "acq")[0]
    if s <= 6:
        return epath(seed, s, c)
    p = D.stage_map("ID" if arm == "ID" else "FS_B", seed, c, s)
    require(p is not None, f"no map {arm} {seed} {c} {s}")
    return p


def traj():
    D.verify_lock()
    require(K.read(ESEAL)["pass"], "seal first")
    done = set()
    if TRAJ.exists():
        for line in TRAJ.read_text().splitlines():
            r = json.loads(line); done.add((r["arm"], r["seed"], r["category"]))
    for c in D.ORDER:
        if c == "zipper":
            continue
        todo = [(a, s) for s in D.SEEDS for a in D.ARMS if (a, s, c) not in done]
        if not todo:
            continue
        rows, hws, masks = D.category_masks(c)
        j = D.stage_of(c)
        for arm, seed in todo:
            prev, stages, trans = None, [], []
            for s in range(j, 16):
                P, NG, NB, _ = D.state(state_path(arm, seed, c, s), rows, hws, masks, per_image=False)
                q = X.quantiles([NG, NB], (0.99,))
                stages.append({"stage": s, "ap": X.ap_pn(P, [NG, NB]), "pos_mean": float(P.mean(dtype=np.float64)),
                               "pos_q50": float(np.median(P)), "neg_q99": q[0.99],
                               "good_q99": X.quantiles([NG], (0.99,))[0.99] if NG.size else None,
                               "bg_q99": X.quantiles([NB], (0.99,))[0.99]})
                if prev is not None:
                    pP, pNG, pNB = prev
                    aa, ff = stages[-2]["ap"], stages[-1]["ap"]
                    fa, af = X.ap_pn(P, [pNG, pNB]), X.ap_pn(pP, [NG, NB])
                    d, dp, dn = X.shapley(aa, fa, af, ff)
                    trans.append({"from": s - 1, "to": s, "D": d, "D_P": dp, "D_N": dn})
                prev = (P, NG, NB)
                gc.collect()
            del prev
            gc.collect()
            r = {"arm": arm, "seed": seed, "category": c, "acq_stage": j, "stages": stages, "transitions": trans,
                 "telescoping_err": abs(sum(t["D"] for t in trans) - (stages[0]["ap"] - stages[-1]["ap"])),
                 "rss_gib": D.rss_gib(), "peak_gib": D.peak_gib()}
            with open(TRAJ, "a") as f:
                f.write(json.dumps(r) + "\n")
            worst = max(trans, key=lambda t: t["D"])
            print(f"traj {c} {arm} {seed} worst {worst['from']}->{worst['to']} D={worst['D']:+.3f} "
                  f"DP={worst['D_P']:+.3f} DN={worst['D_N']:+.3f} peak={D.peak_gib():.2f}", flush=True)
        del rows, hws, masks
        gc.collect()
    # sealed-value check for stages 7..15 and the compact artifact
    rec = [json.loads(x) for x in TRAJ.read_text().splitlines()]
    m1 = K.read(D.A2 / "M1_METRICS.json")["per_cell"]; m4 = K.read(D.A2 / "M4_METRICS.json")["per_cell"]
    worst = 0.0
    for r in rec:
        for st in r["stages"]:
            if st["stage"] >= 7:
                k = f"{r['arm']}|{r['seed']}|{st['stage']}|{r['category']}"
                worst = max(worst, abs(st["ap"] - (m1 if st["stage"] <= 9 else m4)[k]["ap"]))
    K.write_json(TRAJ_OUT, {"records": rec, "sealed_parity_stages_7_15_worst_abs": worst,
                            "early_seal": K.sha(ESEAL), "utc": K.now()})
    print("trajectory written; sealed parity worst", worst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lock", "predict", "seal", "traj"])
    a = ap.parse_args()
    globals()[a.cmd]()
