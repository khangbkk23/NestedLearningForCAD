# exps/nl_align_02/m4.py
"""NL-ALIGN-02-MVTec M4: full-stream (stages 10-15) descriptive evaluation of ID (R15), FS_B and CN (M3 runs).

BENCHMARK_DESCRIPTIVE / EXPLORATORY. Same scorer, metrics and bootstrap as exps.nl_align_02.m (M1).

  prereg    M4_PREREG.json: final-verdict rule frozen before any stage-10..15 prediction or mask
  predict   --seed S: label-free maps for every arm x stage 10..15 x acquired category
  seal      M4_PRED_SEAL.json (+ label-free parity of ID stage 15 vs the sealed M1 `fin` maps)
  evaluate  single mask pass -> M4_METRICS.json (stages 10-15; forgetting acquisition -> final with paired CIs)
  decide    FINAL_VERDICT.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from exps.cadfinal import common as K
from exps.cadfinal import metrics as M
from exps.nl_align_02 import m as M1
from exps.task_p.p0_normal_support import require

ROOT = K.ROOT
OUT = M1.OUT
PRED = OUT / "predictions_m4"
PREREG, SEAL, METRICS, FINAL = (OUT / "M4_PREREG.json", OUT / "M4_PRED_SEAL.json", OUT / "M4_METRICS.json",
                                OUT / "FINAL_VERDICT.json")
M3 = OUT / "m3"
SEEDS, ARMS = M1.SEEDS, M1.ARMS
STAGES = tuple(range(10, 16))
T = 15
SRC = ["exps/nl_align_02/m4.py", "exps/nl_align_02/m.py", "exps/nl_align_02/m3.py", "exps/cadfinal/metrics.py",
       "exps/cadfinal/common.py", "exps/dinomaly/readout.py", "exps/dinomaly/official.py", "exps/dinomaly/encoder.py",
       "exps/calm/stats.py", "exps/calm02/native.py"]
RULE = {
    "primary_endpoint": "final (stage-15) category-balanced macro native Pixel AP over all 15 categories; seed mean; "
                        "paired whole-image bootstrap (1,000 reps, category x label strata, identical weights)",
    "supported_exploratory": [f"Delta_ID(final) >= {M1.DELTA_MIN} and CI lower > 0", "Delta_ID > 0 in >= 2/3 seeds",
                              "Delta_CN(final) > 0 and CI lower > 0",
                              f"no category with final seed-mean Delta_ID < -{M1.CAT_HARM}",
                              f"test/good q99 ratio FS_B/ID <= {M1.TAIL_RATIO} in every category at stage 15",
                              "integrity (M3 lock, runner equivalence, seals, ID parity vs M1 fin maps)"],
    "not_supported": f"CI upper(Delta_ID final) < {M1.DELTA_MIN}, or Delta_ID <= 0 with <= 1/3 seeds positive",
    "inconclusive": "otherwise",
    "labels": {"supported_exploratory": "FIXED_Q2_REAL_TRANSFER_SUPPORTED_EXPLORATORY",
               "not_supported": "FIXED_Q2_REAL_TRANSFER_NOT_SUPPORTED", "inconclusive": "FIXED_Q2_REAL_TRANSFER_INCONCLUSIVE",
               "blocked": "EXECUTION_OR_INTEGRITY_BLOCKED"},
    "secondary_descriptive": ["macro acquisition->final forgetting (signed, positive part, worst) with paired CI",
                              "stage-averaged macro AP over stages 7-15", "per-stage trajectory", "pooled AP",
                              "Image AUROC", "per-category and per-seed effects", "normal-score tails"],
    "note": "same thresholds as the frozen M2 rule; M3 was owner-authorized after M2 = INCONCLUSIVE, so the full-stream "
            "result is exploratory and selection-influenced",
}


def cells(seed):
    return [(a, s, c) for a in ARMS for s in STAGES for c in M1.acquired(s)]


def ckpt(arm, seed, stage):
    if arm == "ID":
        return M1.ckpt("ID", seed, stage)
    rec = K.read(M3 / "runs" / f"{arm}_seed{seed}" / f"STAGE{stage}.json")
    p = ROOT / rec["checkpoint"]
    require(K.sha(p) == rec["checkpoint_sha256"], f"checkpoint changed: {p}")
    return p, rec["checkpoint_sha256"]


def pred_path(seed, arm, stage, cat):
    return PRED / f"seed{seed}" / f"{arm}_s{stage}_{cat}.pt"


def prereg():
    require(not PREREG.exists(), "exists")
    for a in ("FS_B", "CN"):
        for sd in SEEDS:
            for s in STAGES:
                ckpt(a, sd, s)
    K.write_json(PREREG, {"rule": RULE, "sources": {s: K.sha(ROOT / s) for s in SRC},
                          "m3_lock": K.sha(M3 / "M3_LOCK.json"), "rerun_check": K.read(M3 / "RERUN_CHECK.json"),
                          "m1_metrics": K.sha(M1.METRICS), "m2_decision": K.sha(M1.DECISION), "utc": K.now()})


def verify_prereg():
    p = K.read(PREREG)
    bad = [s for s, d in p["sources"].items() if K.sha(ROOT / s) != d]
    require(not bad and p["rule"] == RULE, f"changed after M4 prereg: {bad}")


@torch.no_grad()
def predict(seed):
    from exps.dinomaly import encoder as E, readout as R
    verify_prereg()
    enc = M1._setup()
    ids = M1.ids_()
    root = ROOT / "data/mvtec"
    (PRED / f"seed{seed}").mkdir(parents=True, exist_ok=True)
    for s in STAGES:
        mine = [c for c in M1.acquired(s)]
        if all(pred_path(seed, a, s, c).exists() for a in ARMS for c in mine):
            continue
        decs = {}
        for a in ARMS:
            from exps.cadfinal import dino as DN
            p, d = ckpt(a, seed, s)
            dec = DN.O_decoder()
            dec.load_state_dict(torch.load(p, weights_only=False)["state"])
            decs[a] = (dec.eval(), d)
        for j, c in enumerate(mine):
            if all(pred_path(seed, a, s, c).exists() for a in ARMS):
                continue
            require(not any(pred_path(seed, a, s, c).exists() for a in ARMS), f"partial cell {s} {c}")
            t = ids["tasks"][j]
            acc = {a: ([], []) for a in ARMS}
            sec = {a: 0.0 for a in ARMS}
            for i in range(0, len(t["test"]), 8):
                x = torch.stack([E.preprocess(root / r["relative_path"]) for r in t["test"][i:i + 8]]).cuda()
                f = enc.groups(x).float()
                for a in ARMS:
                    t0 = time.perf_counter()
                    en, de = decs[a][0](f[:, 0], f[:, 1])
                    sm, im = R.dinomaly(en, de)
                    torch.cuda.synchronize(); sec[a] += time.perf_counter() - t0
                    acc[a][0].append(sm.cpu()); acc[a][1].append(im.cpu())
            for a in ARMS:
                torch.save({"s256": torch.cat(acc[a][0]), "image_RD": torch.cat(acc[a][1]),
                            "paths": [r["relative_path"] for r in t["test"]], "arm": a, "seed": seed, "stage": s,
                            "category": c, "checkpoint_sha256": decs[a][1], "decoder_seconds": sec[a],
                            "n_images": len(t["test"]), "utc": K.now()}, pred_path(seed, a, s, c))
        del decs
        torch.cuda.empty_cache()
        print("predict", seed, "stage", s, flush=True)


def seal():
    verify_prereg()
    require(not SEAL.exists(), "sealed")
    files, missing = {}, []
    for sd in SEEDS:
        for a, s, c in cells(sd):
            p = pred_path(sd, a, s, c)
            if not p.exists():
                missing.append(str(p)); continue
            files[str(p.relative_to(OUT))] = K.sha(p)
    extra = [str(p.relative_to(OUT)) for p in PRED.rglob("*.pt") if str(p.relative_to(OUT)) not in files]
    require(not missing and not extra, f"incomplete/extra: {missing[:3]} {extra[:3]}")
    # label-free parity: ID stage-15 maps vs the sealed M1 `fin` maps (same R15 checkpoints, same scorer)
    par = {}
    for sd in SEEDS:
        for j, c in enumerate(M1.acquired(15)):
            mine = torch.load(pred_path(sd, "ID", 15, c), weights_only=False)
            ref = torch.load(K.OUT / "scores/mvtec" / f"DINOMALY_R_seed{sd}" / f"fin_{j:02d}.pt", weights_only=True)
            require(mine["paths"] == ref["paths"], "identity order")
            par[f"{sd}|{c}"] = max(float((mine["s256"] - ref["s256"]).abs().max()),
                                   float((mine["image_RD"] - ref["image_RD"]).abs().max()))
    worst = max(par.values())
    K.write_json(SEAL, {"prereg": K.sha(PREREG), "files": files, "n_cells": len(files),
                        "expected": 3 * len(cells(SEEDS[0])), "id_parity_vs_m1_fin": par, "id_parity_worst": worst,
                        "id_parity_pass": worst <= 1e-3, "utc": K.now()})
    print("sealed", len(files), "ID parity worst", worst)
    require(worst <= 1e-3, "ID stage-15 parity vs M1 failed")


def _load_cell(seed, arm, stage, cat):
    """Prediction file of any sealed cell (M1 for stages 6-9, M4 for 10-15, M1 acq maps for R15 acquisition)."""
    if arm == "M1ACQ":
        j = M1.acquired(15).index(cat)
        return torch.load(K.OUT / "scores/mvtec" / f"DINOMALY_R_seed{seed}" / f"acq_{j:02d}.pt", weights_only=True)
    if stage <= 9:
        return torch.load(M1.pred_path(seed, arm, stage, cat), weights_only=False)
    return torch.load(pred_path(seed, arm, stage, cat), weights_only=False)


def evaluate():
    verify_prereg()
    lk = K.read(SEAL)
    require(not METRICS.exists(), "one-shot")
    require(all(K.sha(OUT / n) == d for n, d in lk["files"].items()), "post-seal change")
    l1 = K.read(M1.SEAL)
    require(all(K.sha(OUT / n) == d for n, d in l1["files"].items()), "M1 seal")
    ids = M1.ids_()
    root = ROOT / "data/mvtec"
    names = M1.acquired(15)
    ap, iau, q99, boot, pool = {}, {}, {}, {}, {}
    acqb = {}                                  # (arm, seed, cat) -> (ap, boot) at the arm's acquisition state
    for j, c in enumerate(names):
        t = ids["tasks"][j]
        rows = t["test"]
        labels = np.array([r["label"] for r in rows])
        require(0 < labels.sum() < len(labels), f"invalid AUROC cohort {c}")
        hws = [M.native_hw(root / r["relative_path"]) for r in rows]
        masks = [M.native_mask(root / r["mask_path"] if r["mask_path"] else None, hw) for r, hw in zip(rows, hws)]
        W = M.bootstrap_weights(labels, M1.REPS, M1.SEED0 + j)
        jobs = [(a, sd, s) for sd in SEEDS for a in ARMS for s in STAGES if c in M1.acquired(s)]
        acq_stage = j + 1
        for sd in SEEDS:
            for a in ARMS:
                if acq_stage <= 6:
                    src = ("M1ACQ", acq_stage)
                elif a == "ID":
                    src = ("M1ACQ", acq_stage)                     # R15 = ID lineage at every stage
                else:
                    src = (a, acq_stage)
                jobs.append((("ACQ", a, src), sd, acq_stage))
        cache = {}
        for a, sd, s in jobs:
            if isinstance(a, tuple):
                _, arm, (src, st) = a
                key = (src, sd, st)
                if key not in cache:
                    p = _load_cell(sd, src, st, c)
                    cache[key] = _ap_boot(p, rows, hws, masks, labels, W, need_pool=False)[:2]
                acqb[(arm, sd, c)] = cache[key]
                continue
            p = _load_cell(sd, a, s, c)
            v, b, img, pp = _ap_boot(p, rows, hws, masks, labels, W, need_pool=True)
            ap[(a, sd, s, c)], boot[(a, sd, s, c)] = v, b
            iau[(a, sd, s, c)] = M.auroc(labels, img)
            q99[(a, sd, s, c)] = float(np.quantile(img[labels == 0], 0.99))
            pk = (a, sd, s)
            if pk not in pool:
                pool[pk] = [np.zeros(M1.POOL_BINS), np.zeros(M1.POOL_BINS)]
            pool[pk][0] += pp[0]; pool[pk][1] += pp[1]
        print("eval", c, {a: round(np.mean([ap[(a, sd, 15, c)] for sd in SEEDS]), 4) for a in ARMS}, flush=True)
        del masks
    res = _summarize(ap, iau, q99, boot, pool, acqb, names)
    res.update({"seal": K.sha(SEAL), "prereg": K.sha(PREREG), "utc": K.now()})
    K.write_json(METRICS, res)
    print("written", METRICS)


def _ap_boot(p, rows, hws, masks, labels, W, need_pool):
    require(p["paths"] == [r["relative_path"] for r in rows], "identity order")
    low, img = p["s256"].numpy(), p["image_RD"].numpy()
    maps = [M.to_native("dinomaly", x, hw) for x, hw in zip(low, hws)]
    v = M.exact_ap(maps, masks)["ap"]
    lo, hi = max(float(low.min()), 0.0), float(low.max())
    pos, tot = np.zeros((len(rows), M1.BINS)), np.zeros((len(rows), M1.BINS))
    for n, (m, y) in enumerate(zip(maps, masks)):
        pos[n], tot[n] = M.hist(m, y, lo, hi, M1.BINS)
    b = np.array([M.ap_from_hist(pos.T @ w, tot.T @ w) for w in W])
    pp = None
    if need_pool:
        pp = [np.zeros(M1.POOL_BINS), np.zeros(M1.POOL_BINS)]
        for m, y in zip(maps, masks):
            h1, h2 = M.hist(m, y, 0.0, 2.0, M1.POOL_BINS); pp[0] += h1; pp[1] += h2
    return v, b, img, pp


def _ci(x):
    return np.percentile(x, [2.5, 97.5]).tolist()


def _summarize(ap, iau, q99, boot, pool, acqb, names):
    def macro(a, sd, s):
        return float(np.mean([ap[(a, sd, s, c)] for c in M1.acquired(s)]))

    def bmacro(a, sd, s):
        return np.mean([boot[(a, sd, s, c)] for c in M1.acquired(s)], 0)

    out = {"per_cell": {f"{a}|{sd}|{s}|{c}": {"ap": ap[k], "image_auroc": iau[k], "good_q99": q99[k]}
                        for k in ap for (a, sd, s, c) in [k]}}
    out["per_stage"] = {f"{a}|{s}": {"macro_ap": float(np.mean([macro(a, sd, s) for sd in SEEDS])),
                                     "macro_ap_per_seed": [macro(a, sd, s) for sd in SEEDS],
                                     "macro_image_auroc": float(np.mean([np.mean([iau[(a, sd, s, c)]
                                                                                  for c in M1.acquired(s)])
                                                                         for sd in SEEDS])),
                                     "pooled_ap_hist": float(np.mean([M.ap_from_hist(*pool[(a, sd, s)])
                                                                      for sd in SEEDS]))}
                        for a in ARMS for s in STAGES}

    def contrast(a, b):
        by = {}
        for s in STAGES:
            ps = [macro(a, sd, s) - macro(b, sd, s) for sd in SEEDS]
            by[str(s)] = {"delta": float(np.mean(ps)), "per_seed": ps,
                          "ci95": _ci(np.mean([bmacro(a, sd, s) - bmacro(b, sd, s) for sd in SEEDS], 0))}
        pc = {c: {"delta": float(np.mean([ap[(a, sd, 15, c)] - ap[(b, sd, 15, c)] for sd in SEEDS])),
                  "per_seed": [ap[(a, sd, 15, c)] - ap[(b, sd, 15, c)] for sd in SEEDS],
                  "ci95": _ci(np.mean([boot[(a, sd, 15, c)] - boot[(b, sd, 15, c)] for sd in SEEDS], 0))}
              for c in names}
        ia = float(np.mean([np.mean([iau[(a, sd, 15, c)] - iau[(b, sd, 15, c)] for c in names]) for sd in SEEDS]))
        return {"final": by["15"], "by_stage": by, "per_category_final": pc, "image_auroc_delta_final": ia,
                "stage_avg_10_15": float(np.mean([by[str(s)]["delta"] for s in STAGES]))}
    out["contrasts"] = {"FS_B-ID": contrast("FS_B", "ID"), "FS_B-CN": contrast("FS_B", "CN"),
                        "CN-ID": contrast("CN", "ID")}
    fg = {}
    old = names[:-1]
    for a in ARMS:
        F = {c: [acqb[(a, sd, c)][0] - ap[(a, sd, 15, c)] for sd in SEEDS] for c in old}
        bF = np.mean([np.mean([acqb[(a, sd, c)][1] - boot[(a, sd, 15, c)] for c in old], 0) for sd in SEEDS], 0)
        Fm = {c: float(np.mean(v)) for c, v in F.items()}
        fg[a] = {"per_category": Fm, "macro_F": float(np.mean(list(Fm.values()))), "macro_F_ci95": _ci(bF),
                 "macro_F_plus": float(np.mean(np.maximum(list(Fm.values()), 0))),
                 "worst": [max(Fm, key=Fm.get), max(Fm.values())],
                 "per_seed_macro_F": [float(np.mean([F[c][i] for c in old])) for i in range(len(SEEDS))],
                 "acquisition_ap": {c: float(np.mean([acqb[(a, sd, c)][0] for sd in SEEDS])) for c in names}}
    for a, b in (("FS_B", "ID"), ("FS_B", "CN")):
        d = np.mean([np.mean([(acqb[(a, sd, c)][1] - boot[(a, sd, 15, c)]) - (acqb[(b, sd, c)][1] - boot[(b, sd, 15, c)])
                              for c in old], 0) for sd in SEEDS], 0)
        fg[f"{a}-{b}"] = {"delta_macro_F": fg[a]["macro_F"] - fg[b]["macro_F"], "ci95": _ci(d)}
    out["forgetting_acq_to_final"] = fg
    out["tail_ratio_q99_FS_B_over_ID_s15"] = {c: float(np.mean([q99[("FS_B", sd, 15, c)] for sd in SEEDS]) /
                                                     np.mean([q99[("ID", sd, 15, c)] for sd in SEEDS])) for c in names}
    return out


def decide():
    verify_prereg()
    require(METRICS.exists() and not FINAL.exists(), "evaluate first / decided")
    m, sl = K.read(METRICS), K.read(SEAL)
    rr = K.read(M3 / "RERUN_CHECK.json")
    integ = rr["pass"] and sl["id_parity_pass"] and sl["n_cells"] == sl["expected"]
    d, dc = m["contrasts"]["FS_B-ID"]["final"], m["contrasts"]["FS_B-CN"]["final"]
    pc = m["contrasts"]["FS_B-ID"]["per_category_final"]
    worst = min((v["delta"], c) for c, v in pc.items())
    tails = m["tail_ratio_q99_FS_B_over_ID_s15"]
    npos = sum(x > 0 for x in d["per_seed"])
    cond = {"delta_ci": d["delta"] >= M1.DELTA_MIN and d["ci95"][0] > 0, "seeds": npos >= 2,
            "beats_cn": dc["delta"] > 0 and dc["ci95"][0] > 0, "no_cat_harm": worst[0] >= -M1.CAT_HARM,
            "tails": max(tails.values()) <= M1.TAIL_RATIO, "integrity": integ}
    if not integ:
        key = "blocked"
    elif all(cond.values()):
        key = "supported_exploratory"
    elif d["ci95"][1] < M1.DELTA_MIN or (d["delta"] <= 0 and npos <= 1):
        key = "not_supported"
    else:
        key = "inconclusive"
    out = {"label": RULE["labels"][key], "scope": "completed stage-15 stream (MVTec, exploratory)",
           "conditions": cond, "final_delta_id": d, "final_delta_cn": dc, "worst_category_final": worst,
           "max_tail_ratio": max(tails.values()), "forgetting": {k: m["forgetting_acq_to_final"][k] for k in
                                                                ("FS_B-ID", "FS_B-CN")},
           "metrics": K.sha(METRICS), "utc": K.now()}
    K.write_json(FINAL, out)
    print(out)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("cmd", choices=["prereg", "predict", "seal", "evaluate", "decide"])
    ap_.add_argument("--seed", type=int)
    a_ = ap_.parse_args()
    if a_.cmd == "predict":
        predict(a_.seed)
    else:
        {"prereg": prereg, "seal": seal, "evaluate": evaluate, "decide": decide}[a_.cmd]()
