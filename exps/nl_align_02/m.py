# exps/nl_align_02/m.py
"""NL-ALIGN-02-MVTec: fixed-Q2 (G1 FS_B) real-anomaly transfer on existing stage 7-9 checkpoints.

BENCHMARK_DESCRIPTIVE / EXPLORATORY: the MVTec official test is historically exposed. No training here.

Arms: ID = RH-01b R15; FS_B = G1 fixed Q2 mixing; CN = G1 compute-matched normal control; S6 = the shared R15
stage-6 start (all arms share stages 1-6), predicted only for categories 1-6 to measure stage-6 -> s retention.

  g0        G0_AUDIT.json + CUSTODY.json: branch/HEAD, source/checkpoint/encoder hashes vs their locks (G1 seal,
            RH-01b records), MVTec identities with image/mask hashes, P1-DEV identities, state-restoration audit
  prereg    M_PREREG.json: protocol, metrics, the frozen M2 continuation rule (before any new prediction/mask)
  parity    label-free: ID stage-a(c) maps of c in {leather, metal_nut, pill} vs the sealed M1 acq maps
  predict   --seed S: label-free maps (R_D 256^2) + image scores, every arm x stage x acquired category
  seal      PRED_SEAL.json: completeness (expected = actual cells) + hashes
  evaluate  the single mask pass -> M1_METRICS.json
  decide    M2_DECISION.json (applies the frozen rule mechanically)

  CUDA: PYTHONPATH=$PWD python -m exps.nl_align_02.m <cmd>
"""
from __future__ import annotations

import argparse
import subprocess
import time

import numpy as np
import torch

from exps.cadfinal import common as K
from exps.cadfinal import metrics as M
from exps.task_p.p0_normal_support import require

ROOT = K.ROOT
OUT = ROOT / "results/hope_cad/nl_align_02_mvtec"
PRED = OUT / "predictions"
G0, CUSTODY, PREREG = OUT / "G0_AUDIT.json", OUT / "CUSTODY.json", OUT / "M_PREREG.json"
PARITY, SEAL, METRICS, DECISION = OUT / "M0_PARITY.json", OUT / "PRED_SEAL.json", OUT / "M1_METRICS.json", OUT / "M2_DECISION.json"
SEEDS = (2711, 2712, 2713)
ARMS = ("ID", "FS_B", "CN")
STAGES = (7, 8, 9)
START = 6
REPS, SEED0, BINS, POOL_BINS = 1000, 20261105, 8192, 65536
CAP_GIB = 4.5            # runtime VRAM cap for inference (G1 `real` OOMed at 2.6 GiB); no numeric effect
SRC = ["exps/nl_align_02/m.py", "exps/cadfinal/m1.py", "exps/cadfinal/common.py", "exps/cadfinal/metrics.py",
       "exps/cadfinal/dino.py", "exps/dinomaly/readout.py", "exps/dinomaly/official.py", "exps/dinomaly/encoder.py",
       "exps/calm/stats.py", "exps/calm02/native.py", "exps/calm02/data.py", "exps/nl_align_01/g1.py"]

# ---- the frozen M2 continuation rule (written before any new prediction or mask read) ----------------------
DELTA_MIN = 0.005        # meaningful macro AP benefit; = the half-width of the G1-real paired image CI (~0.005) and
#                          about one R15 seed SD of stage-9 macro AP (0.0045); both known before this task
CAT_HARM = 0.03          # worst-category harm margin: 6 x DELTA_MIN (per-category image noise is ~3-6x the macro's)
TAIL_RATIO = 1.25        # test/good image-score q99 ratio FS_B/ID per category (FS_A carpet was x5-7 in G1)
RULE = {
    "primary_endpoint": "stage-averaged macro native Pixel AP: mean over s in {7,8,9} of the category-balanced macro "
                        "over the acquired set C_s = {c_1..c_s}; seed mean; paired whole-image bootstrap (1,000 reps, "
                        "stratified by category x label, identical weights for every arm/stage/seed)",
    "supported": [f"(1) Delta_ID = FS_B - ID >= {DELTA_MIN}", "(2) CI lower(Delta_ID) > 0",
                  "(3) Delta_CN = FS_B - CN > 0 and CI lower(Delta_CN) > 0",
                  "(4) Delta_ID > 0 in >= 2 of 3 seeds",
                  f"(5) no category with stage-9 seed-mean Delta_ID < -{CAT_HARM}",
                  f"(6) test/good image-score q99 ratio FS_B/ID <= {TAIL_RATIO} in every category at stage 9 (seed mean)",
                  "(7) integrity: G0 pass, M0 parity pass, prediction seal complete"],
    "not_supported": f"CI upper(Delta_ID) < {DELTA_MIN} (a benefit of the meaningful size is excluded), or "
                     "Delta_ID <= 0 with at most 1 of 3 seeds positive",
    "inconclusive": "neither supported nor not_supported",
    "integrity_blocked": "any of G0 / M0 / seal fails",
    "labels": {"supported": "Q2_MVTEC_EARLY_TRANSFER_SUPPORTED", "not_supported": "Q2_MVTEC_EARLY_TRANSFER_NOT_SUPPORTED",
               "inconclusive": "Q2_MVTEC_EARLY_INCONCLUSIVE", "integrity_blocked": "M1_INTEGRITY_BLOCKED"},
    "continuation": "M3 (stages 10-15) only on Q2_MVTEC_EARLY_TRANSFER_SUPPORTED and a measured projection <= 12 GPU-h",
    "prior_exposure_disclosure": "stage-9 / categories 1-9 real AP of ID, FS_B, CN was already computed descriptively in "
                                 "NL-ALIGN-01 (G1_REAL.json: FS_B - ID +0.0004 [-0.0049, +0.0056], FS_B - CN about "
                                 "+0.008). Thresholds above are derived from that run's noise level, not from its effect. "
                                 "Stages 7-8, per-seed/per-stage cells, forgetting and tails are new.",
}


def cmd_out(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def ids_():
    ids = K.benchmark_identities("mvtec")
    require([t["task"] for t in ids["tasks"]] == list(K.ORDERS["mvtec"]), "stream order")
    return ids


def acquired(stage):
    """C_s: categories acquired by `stage` (1-based stream positions); never future tasks."""
    require(1 <= stage <= 15, "stage")
    return list(K.ORDERS["mvtec"][:stage])


def cells(seed):
    """Every expected (arm, stage, category) prediction cell for one seed."""
    out = [(a, s, c) for a in ARMS for s in STAGES for c in acquired(s)]
    out += [("S6", START, c) for c in acquired(START)]
    return out


def ckpt(arm, seed, stage):
    from exps.nl_align_01 import g1 as G
    if arm in ("ID", "S6"):
        return G.r15_record(seed, stage)
    return G.checkpoint(arm, seed, stage)


def pred_path(seed, arm, stage, cat):
    return PRED / f"seed{seed}" / f"{arm}_s{stage}_{cat}.pt"


# ---- G0 ----------------------------------------------------------------------------------------------------

def g0():
    from exps.dinomaly import common as D
    from exps.nl_align_01 import g1 as G
    require(not G0.exists(), "exists")
    seal = K.read(G.SEAL)
    bad_seal = [n for n, d in seal["files"].items() if K.sha(ROOT / n) != d]
    pre = K.read(G.PREREG)
    bad_src = [s for s, d in pre["sources"].items() if K.sha(ROOT / s) != d]
    ck = {}
    for seed in SEEDS:
        for s in range(1, 16):
            ck[f"R15|{seed}|{s}"] = G.r15_record(seed, s)[1]          # raises on mismatch
        for a in ("FS_B", "CN", "FS_A"):
            for s in STAGES:
                ck[f"{a}|{seed}|{s}"] = G.checkpoint(a, seed, s)[1]
    enc_ok = K.sha(D.CKPT["base"][0]) == D.CKPT["base"][1]
    m1 = K.read(K.OUT / "M1_LOCK.json")
    bad_m1 = [n for n, d in m1["files"].items() if K.sha(K.OUT / n) != d]
    # state restoration (source audit, recorded): G1 train() builds a fresh block optimizer and scheduler every
    # stage (train_stage -> block_optimizer / O.make_scheduler), reseeds torch and the sampler from
    # RB.stage_seeds(R15, seed, s), recomputes replay from C.buffer_index(seed, n, done) and seeds the auxiliary
    # sampler from (seed, stage). PARITY.json shows that a stage-6 decoder state alone reproduces R15 stage 7 bitwise.
    restoration = {"decoder_state": "in checkpoint",
                   "optimizer_moments": "reset per stage by protocol (fresh StableAdamW in train_stage)",
                   "scheduler": "reset per stage (O.make_scheduler(opt, iters))",
                   "rng": "reseeded per stage from RB.stage_seeds('R15', seed, stage); aux samplers from (seed, stage)",
                   "replay": "deterministic C.buffer_index(seed, n_train, done)",
                   "stream_position": "stage index in STAGE*.json / checkpoint name",
                   "synthetic_bank": "BANK_seed*.json covers stage_max = 9 only: stages 10-15 need the bank extended "
                                     "with the unchanged builder before any M3 run",
                   "evidence": "results/hope_cad/nl_align_01/g1/PARITY.json (bitwise, max |diff| 0)"}
    bank_max = {s: K.read(G.OUT / f"BANK_seed{s}.json")["stage_max"] for s in SEEDS}
    out = {"branch": cmd_out("git", "branch", "--show-current"), "head": cmd_out("git", "rev-parse", "HEAD"),
           "git_status": cmd_out("git", "status", "--short"),
           "agents_md_sha": K.sha(ROOT / "AGENTS.md"),
           "g1_seal_ok": not bad_seal, "g1_seal_sha": K.sha(G.SEAL), "g1_prereg_sources_ok": not bad_src,
           "g1_decision_sha": K.sha(G.DECISION), "g1_real_sha": K.sha(G.REAL),
           "checkpoints": ck, "encoder_ok": enc_ok, "encoder_sha": D.CKPT["base"][1],
           "m1_lock_ok": not bad_m1, "m1_parity": K.read(K.OUT / "M1_PARITY.json"),
           "rh01b_protocol_sha": K.sha(ROOT / "results/hope_cad/rh01b/PROTOCOL.json"),
           "state_restoration": restoration, "bank_stage_max": bank_max,
           "scorer": "exps.dinomaly.readout.dinomaly (official chain 392->256, Gaussian 5x5 sigma 4, image = top-1% "
                     "mean) -> exps.cadfinal.metrics.to_native (bilinear, align_corners=False) -> exact_ap",
           "utc": K.now()}
    out["pass"] = bool(out["g1_seal_ok"] and out["g1_prereg_sources_ok"] and enc_ok and out["m1_lock_ok"])
    # custody
    ids = ids_()
    root = ROOT / "data/mvtec"
    from exps.calm02 import data as CD
    p1 = set(CD.p1_development_paths())
    cus = {"order": [t["task"] for t in ids["tasks"]], "tasks": {}}
    for t in ids["tasks"]:
        cus["tasks"][t["task"]] = {
            "train_r15": {r["relative_path"]: K.sha(root / r["relative_path"]) for r in t["train"]},
            "test": [{"path": r["relative_path"], "label": r["label"], "sha": K.sha(root / r["relative_path"]),
                      "mask": r["mask_path"], "mask_sha": K.sha(root / r["mask_path"]) if r["mask_path"] else None,
                      "p1_dev": r["relative_path"] in p1} for r in t["test"]],
            "n_train": len(t["train"]), "n_test_good": sum(r["label"] == 0 for r in t["test"]),
            "n_test_anomalous": sum(r["label"] == 1 for r in t["test"])}
    cus["p1_dev"] = sorted(p1)
    cus["utc"] = K.now()
    K.write_json(CUSTODY, cus)
    out["custody_sha"] = K.sha(CUSTODY)
    K.write_json(G0, out)
    print({k: v for k, v in out.items() if k not in ("checkpoints", "state_restoration", "git_status")})
    require(out["pass"], "G0 integrity failure")


# ---- prereg ------------------------------------------------------------------------------------------------

def prereg():
    require(not PREREG.exists(), "exists")
    require(K.read(G0)["pass"], "G0 first")
    K.write_json(PREREG, {"task": "agents/tasks/task_nl_align_02_mvtec.md", "rule": RULE,
                          "thresholds": {"delta_min": DELTA_MIN, "cat_harm": CAT_HARM, "tail_ratio": TAIL_RATIO},
                          "arms": ARMS, "stages": STAGES, "seeds": SEEDS, "cells_per_seed": len(cells(SEEDS[0])),
                          "metrics": "exact native Pixel AP (ties grouped; masks never resized; R_D 256^2 bilinear to "
                                     "native), Image AUROC (top-1% R_D), pooled AP from 65,536-bin histograms, "
                                     "bootstrap CIs from 8,192-bin per-image histograms with paired whole-image weights",
                          "forgetting": "F_{c,s} = AP_{c,a(c)} - AP_{c,s}, s > a(c). For c <= 6 the acquisition state is "
                                        "the shared R15 stage-a(c) checkpoint (common lineage of all arms), scored "
                                        "from the sealed M1 acq maps; for c in 7..9 each arm's own stage-c state. "
                                        "Stage-6 -> s retention uses the S6 predictions.",
                          "sources": {s: K.sha(ROOT / s) for s in SRC}, "g0": K.sha(G0), "utc": K.now()})


def verify_prereg():
    p = K.read(PREREG)
    bad = [s for s, d in p["sources"].items() if K.sha(ROOT / s) != d]
    require(not bad and p["rule"] == RULE, f"changed after prereg: {bad}")
    return p


# ---- inference ---------------------------------------------------------------------------------------------

def _setup():
    from exps.dinomaly import common as D, encoder as E, run as DR
    from exps.rh01 import numerics as NUM
    DR.CAP_GIB = CAP_GIB
    DR.configure()
    NUM.use(True)
    return E.load(D.CKPT["base"][0], "base", "cuda")


def _decoder(arm, seed, stage):
    from exps.cadfinal import dino as DN
    path, digest = ckpt(arm, seed, stage)
    dec = DN.O_decoder()
    dec.load_state_dict(torch.load(path, weights_only=False)["state"])
    return dec.eval(), digest


@torch.no_grad()
def parity():
    """Label-free: this module's ID maps at the acquisition stage of leather/metal_nut/pill vs sealed M1 acq maps."""
    from exps.dinomaly import encoder as E, readout as R
    verify_prereg()
    require(not PARITY.exists(), "exists")
    enc = _setup()
    ids = ids_()
    root = ROOT / "data/mvtec"
    res = {}
    for seed in SEEDS:
        for j in (6, 7, 8):
            t = ids["tasks"][j]
            ref = torch.load(K.OUT / "scores/mvtec" / f"DINOMALY_R_seed{seed}" / f"acq_{j:02d}.pt", weights_only=True)
            require(ref["paths"] == [r["relative_path"] for r in t["test"]], "identity order")
            dec, _ = _decoder("ID", seed, j + 1)
            maps, img = [], []
            for i in range(0, len(t["test"]), 8):
                x = torch.stack([E.preprocess(root / r["relative_path"]) for r in t["test"][i:i + 8]]).cuda()
                f = enc.groups(x).float()
                en, de = dec(f[:, 0], f[:, 1])
                s, im = R.dinomaly(en, de)
                maps.append(s.cpu()); img.append(im.cpu())
            res[f"{seed}|{t['task']}"] = {"max_abs_map": float((torch.cat(maps) - ref["s256"]).abs().max()),
                                          "max_abs_image": float((torch.cat(img) - ref["image_RD"]).abs().max()),
                                          "n": len(t["test"])}
            del dec
    worst = max(max(v["max_abs_map"], v["max_abs_image"]) for v in res.values())
    out = {"cells": res, "worst_abs": worst, "tolerance": 1e-3, "tolerance_source": "M1_PARITY (same oracle rule)",
           "pass": worst <= 1e-3, "utc": K.now()}
    K.write_json(PARITY, out)
    print({"worst_abs": worst, "pass": out["pass"]})
    require(out["pass"], "scorer parity failure")


@torch.no_grad()
def predict(seed):
    from exps.dinomaly import encoder as E, readout as R
    verify_prereg()
    require(K.read(PARITY)["pass"], "parity first")
    require(seed in SEEDS, "seed")
    enc = _setup()
    ids = ids_()
    root = ROOT / "data/mvtec"
    want = cells(seed)
    decs = {}
    for a, s, _ in want:
        if (a, s) not in decs:
            decs[(a, s)] = _decoder(a, seed, s)
    (PRED / f"seed{seed}").mkdir(parents=True, exist_ok=True)
    srch = {s: K.sha(ROOT / s) for s in SRC}
    for j, t in enumerate(ids["tasks"][:max(STAGES)]):
        c = t["task"]
        mine = [(a, s) for a, s, cc in want if cc == c]
        if all(pred_path(seed, a, s, c).exists() for a, s in mine):
            continue
        require(not any(pred_path(seed, a, s, c).exists() for a, s in mine), f"partial category {c}")
        acc = {k: ([], []) for k in mine}
        sec = {k: 0.0 for k in mine}
        enc_sec = 0.0
        for i in range(0, len(t["test"]), 8):
            x = torch.stack([E.preprocess(root / r["relative_path"]) for r in t["test"][i:i + 8]]).cuda()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            f = enc.groups(x).float()
            torch.cuda.synchronize(); enc_sec += time.perf_counter() - t0
            for k in mine:
                t0 = time.perf_counter()
                en, de = decs[k][0](f[:, 0], f[:, 1])
                s_, im = R.dinomaly(en, de)
                torch.cuda.synchronize(); sec[k] += time.perf_counter() - t0
                acc[k][0].append(s_.cpu()); acc[k][1].append(im.cpu())
        for (a, s) in mine:
            torch.save({"s256": torch.cat(acc[(a, s)][0]), "image_RD": torch.cat(acc[(a, s)][1]),
                        "paths": [r["relative_path"] for r in t["test"]], "arm": a, "seed": seed, "stage": s,
                        "category": c, "checkpoint_sha256": decs[(a, s)][1], "sources": srch,
                        "decoder_seconds": sec[(a, s)], "encoder_seconds_shared": enc_sec,
                        "n_images": len(t["test"]), "utc": K.now()}, pred_path(seed, a, s, c))
        print("predict", seed, c, len(mine), f"enc {enc_sec:.1f}s", flush=True)


def seal():
    verify_prereg()
    require(not SEAL.exists(), "sealed")
    files, missing = {}, []
    for seed in SEEDS:
        for a, s, c in cells(seed):
            p = pred_path(seed, a, s, c)
            if not p.exists():
                missing.append(str(p)); continue
            files[str(p.relative_to(OUT))] = K.sha(p)
    extra = [str(p.relative_to(OUT)) for p in PRED.rglob("*.pt") if str(p.relative_to(OUT)) not in files]
    require(not missing and not extra, f"incomplete or extra cells: {missing[:3]} {extra[:3]}")
    K.write_json(SEAL, {"prereg": K.sha(PREREG), "parity": K.sha(PARITY), "g0": K.sha(G0), "files": files,
                        "n_cells": len(files), "expected": 3 * len(cells(SEEDS[0])), "utc": K.now()})
    print("sealed", len(files))


# ---- evaluation (the single mask pass) ---------------------------------------------------------------------

def evaluate():
    verify_prereg()
    lk = K.read(SEAL)
    require(not METRICS.exists(), "one-shot")
    require(all(K.sha(OUT / n) == d for n, d in lk["files"].items()), "post-seal change")
    ml = K.read(K.OUT / "M1_LOCK.json")
    ids = ids_()
    root = ROOT / "data/mvtec"
    names = list(K.ORDERS["mvtec"][:max(STAGES)])
    nop1 = K.cohort_rows("mvtec", "no_p1dev", ids)
    ap, iau, q90, q99 = {}, {}, {}, {}
    boot = {}                                  # (arm, seed, stage, cat) -> [REPS]
    pool = {}                                  # (arm, seed, stage) -> [pos, tot]
    acq_m1 = {}                                # (seed, cat) -> AP at its own acquisition stage (R15 lineage), c<=6
    ap_nop1 = {}
    for j, c in enumerate(names):
        t = ids["tasks"][j]
        rows = t["test"]
        labels = np.array([r["label"] for r in rows])
        require(0 < labels.sum() < len(labels), f"invalid AUROC cohort {c}")   # sklearn returns nan, not an error
        hws = [M.native_hw(root / r["relative_path"]) for r in rows]
        masks = [M.native_mask(root / r["mask_path"] if r["mask_path"] else None, hw) for r, hw in zip(rows, hws)]
        W = M.bootstrap_weights(labels, REPS, SEED0 + j)
        keep = np.array(nop1[c])
        jobs = []
        for seed in SEEDS:
            for a, s, cc in cells(seed):
                if cc == c:
                    jobs.append((a, seed, s, pred_path(seed, a, s, c)))
            if j < START:
                m1 = K.OUT / "scores/mvtec" / f"DINOMALY_R_seed{seed}" / f"acq_{j:02d}.pt"
                require(K.sha(m1) == ml["files"][str(m1.relative_to(K.OUT))], "M1 lock")
                jobs.append(("M1ACQ", seed, j + 1, m1))
        for a, seed, s, path in jobs:
            p = torch.load(path, weights_only=False)
            require(p["paths"] == [r["relative_path"] for r in rows], "identity order")
            low, img = p["s256"].numpy(), p["image_RD"].numpy()
            maps = [M.to_native("dinomaly", x, hw) for x, hw in zip(low, hws)]
            v = M.exact_ap(maps, masks)["ap"]
            if a == "M1ACQ":
                acq_m1[(seed, c)] = v
                continue
            k = (a, seed, s, c)
            ap[k] = v
            ap_nop1[k] = (v if len(keep) == len(rows) else
                          M.exact_ap([maps[i] for i in keep], [masks[i] for i in keep])["ap"])
            iau[k] = M.auroc(labels, img)
            g = img[labels == 0]
            q90[k], q99[k] = float(np.quantile(g, 0.90)), float(np.quantile(g, 0.99))
            lo, hi = max(float(low.min()), 0.0), float(low.max())
            pos, tot = np.zeros((len(rows), BINS)), np.zeros((len(rows), BINS))
            for n, (m, y) in enumerate(zip(maps, masks)):
                pos[n], tot[n] = M.hist(m, y, lo, hi, BINS)
            boot[k] = np.array([M.ap_from_hist(pos.T @ w, tot.T @ w) for w in W])
            pp, pt = np.zeros(POOL_BINS), np.zeros(POOL_BINS)
            for m, y in zip(maps, masks):
                h1, h2 = M.hist(m, y, 0.0, 2.0, POOL_BINS); pp += h1; pt += h2
            pk = (a, seed, s)
            if pk not in pool:
                pool[pk] = [np.zeros(POOL_BINS), np.zeros(POOL_BINS)]
            pool[pk][0] += pp; pool[pk][1] += pt
            del maps
        print("eval", c, {a: round(np.mean([ap[(a, sd, 9, c)] for sd in SEEDS]), 4) for a in ARMS}, flush=True)
        del masks
    res = _summarize(ap, ap_nop1, iau, q90, q99, boot, pool, acq_m1, names)
    res.update({"seal": K.sha(SEAL), "prereg": K.sha(PREREG), "utc": K.now()})
    K.write_json(METRICS, res)
    print("written", METRICS)


def _ci(x):
    return np.percentile(x, [2.5, 97.5]).tolist()


def _summarize(ap, ap_nop1, iau, q90, q99, boot, pool, acq_m1, names):
    def macro(a, sd, s, src=ap):
        return float(np.mean([src[(a, sd, s, c)] for c in acquired(s)]))

    def bmacro(a, sd, s):
        return np.mean([boot[(a, sd, s, c)] for c in acquired(s)], 0)

    out = {"per_cell": {f"{a}|{sd}|{s}|{c}": {"ap": ap[(a, sd, s, c)], "ap_no_p1dev": ap_nop1[(a, sd, s, c)],
                                               "image_auroc": iau[(a, sd, s, c)], "good_q90": q90[(a, sd, s, c)],
                                               "good_q99": q99[(a, sd, s, c)]} for (a, sd, s, c) in ap},
           "acquisition_ap_m1_r15": {f"{sd}|{c}": v for (sd, c), v in acq_m1.items()}}
    arms = list(ARMS)
    stg = {}
    for a in arms:
        for s in STAGES:
            stg[f"{a}|{s}"] = {"macro_ap": float(np.mean([macro(a, sd, s) for sd in SEEDS])),
                               "macro_ap_per_seed": [macro(a, sd, s) for sd in SEEDS],
                               "macro_ap_no_p1dev": float(np.mean([macro(a, sd, s, ap_nop1) for sd in SEEDS])),
                               "macro_image_auroc": float(np.mean([np.mean([iau[(a, sd, s, c)] for c in acquired(s)])
                                                                   for sd in SEEDS])),
                               "pooled_ap_hist": float(np.mean([M.ap_from_hist(*pool[(a, sd, s)]) for sd in SEEDS])),
                               "macro_ap_ci95": _ci(np.mean([bmacro(a, sd, s) for sd in SEEDS], 0))}
    out["per_stage"] = stg

    def contrast(a, b):
        per_seed = [float(np.mean([macro(a, sd, s) - macro(b, sd, s) for s in STAGES])) for sd in SEEDS]
        bb = np.mean([np.mean([bmacro(a, sd, s) - bmacro(b, sd, s) for s in STAGES], 0) for sd in SEEDS], 0)
        by_stage = {}
        for s in STAGES:
            ps = [macro(a, sd, s) - macro(b, sd, s) for sd in SEEDS]
            by_stage[str(s)] = {"delta": float(np.mean(ps)), "per_seed": ps,
                                "ci95": _ci(np.mean([bmacro(a, sd, s) - bmacro(b, sd, s) for sd in SEEDS], 0))}
        per_cat = {}
        for c in names:
            ss = [s for s in STAGES if c in acquired(s)]
            per_cat[c] = {str(s): {"delta": float(np.mean([ap[(a, sd, s, c)] - ap[(b, sd, s, c)] for sd in SEEDS])),
                                   "per_seed": [ap[(a, sd, s, c)] - ap[(b, sd, s, c)] for sd in SEEDS],
                                   "ci95": _ci(np.mean([boot[(a, sd, s, c)] - boot[(b, sd, s, c)] for sd in SEEDS], 0))}
                          for s in ss}
        return {"delta": float(np.mean(per_seed)), "per_seed": per_seed, "ci95": _ci(bb), "by_stage": by_stage,
                "per_category": per_cat,
                "image_auroc_delta_s9": float(np.mean([np.mean([iau[(a, sd, 9, c)] - iau[(b, sd, 9, c)]
                                                                for c in acquired(9)]) for sd in SEEDS]))}
    out["contrasts"] = {"FS_B-ID": contrast("FS_B", "ID"), "FS_B-CN": contrast("FS_B", "CN"),
                        "CN-ID": contrast("CN", "ID")}
    # forgetting / retention
    fg = {}
    for a in arms:
        rec = {}
        for c in names:
            j = names.index(c) + 1
            for s in STAGES:
                if s <= j:
                    continue
                vals = []
                for sd in SEEDS:
                    acq_ = acq_m1[(sd, c)] if j <= START else ap[(a, sd, j, c)]
                    vals.append(acq_ - ap[(a, sd, s, c)])
                rec[f"{c}|{s}"] = {"F": float(np.mean(vals)), "per_seed": vals,
                                   "acq_source": "R15 shared lineage (M1 acq maps)" if j <= START else f"{a} own stage {j}"}
        s9 = [rec[f"{c}|9"]["F"] for c in names[:-1]]
        fg[a] = {"cells": rec, "mean_F_s9": float(np.mean(s9)), "mean_Fplus_s9": float(np.mean(np.maximum(s9, 0))),
                 "worst_F_s9": [names[int(np.argmax(s9))], float(np.max(s9))],
                 "per_seed_mean_F_s9": [float(np.mean([rec[f"{c}|9"]["per_seed"][i] for c in names[:-1]]))
                                        for i in range(len(SEEDS))]}
        ret = {}
        for c in acquired(START):
            for s in STAGES:
                ret[f"{c}|{s}"] = float(np.mean([ap[("S6", sd, START, c)] - ap[(a, sd, s, c)] for sd in SEEDS]))
        fg[a]["stage6_to_s_loss"] = ret
        fg[a]["stage6_to_9_mean_loss_cats1_6"] = float(np.mean([ret[f"{c}|9"] for c in acquired(START)]))
    out["forgetting"] = fg
    out["acquisition_new_categories"] = {a: {c: float(np.mean([ap[(a, sd, names.index(c) + 1, c)] for sd in SEEDS]))
                                             for c in names[START:]} for a in arms}
    out["tails_s9"] = {a: {c: {"q90": float(np.mean([q90[(a, sd, 9, c)] for sd in SEEDS])),
                               "q99": float(np.mean([q99[(a, sd, 9, c)] for sd in SEEDS]))} for c in names} for a in arms}
    out["tail_ratio_q99_FS_B_over_ID_s9"] = {c: out["tails_s9"]["FS_B"][c]["q99"] / out["tails_s9"]["ID"][c]["q99"]
                                             for c in names}
    return out


def decide():
    verify_prereg()
    require(METRICS.exists() and not DECISION.exists(), "evaluate first / decided")
    m = K.read(METRICS)
    g0ok, parok = K.read(G0)["pass"], K.read(PARITY)["pass"]
    sl = K.read(SEAL)
    sealok = sl["n_cells"] == sl["expected"]
    d, dc = m["contrasts"]["FS_B-ID"], m["contrasts"]["FS_B-CN"]
    worst_cat = min((v["9"]["delta"], c) for c, v in d["per_category"].items())
    tails = m["tail_ratio_q99_FS_B_over_ID_s9"]
    cond = {"1_delta_id_ge_min": d["delta"] >= DELTA_MIN, "2_ci_low_id_gt0": d["ci95"][0] > 0,
            "3_beats_cn": dc["delta"] > 0 and dc["ci95"][0] > 0,
            "4_seeds_pos_ge2": sum(x > 0 for x in d["per_seed"]) >= 2,
            "5_no_cat_harm": worst_cat[0] >= -CAT_HARM, "6_tails_ok": max(tails.values()) <= TAIL_RATIO,
            "7_integrity": g0ok and parok and sealok}
    if not cond["7_integrity"]:
        key = "integrity_blocked"
    elif all(cond.values()):
        key = "supported"
    elif d["ci95"][1] < DELTA_MIN or (d["delta"] <= 0 and sum(x > 0 for x in d["per_seed"]) <= 1):
        key = "not_supported"
    else:
        key = "inconclusive"
    out = {"label": RULE["labels"][key], "conditions": cond, "delta_id": d["delta"], "delta_id_ci95": d["ci95"],
           "delta_id_per_seed": d["per_seed"], "delta_cn": dc["delta"], "delta_cn_ci95": dc["ci95"],
           "worst_category_s9": worst_cat, "max_tail_ratio": max(tails.values()),
           "m3_authorized": key == "supported", "metrics": K.sha(METRICS), "utc": K.now()}
    K.write_json(DECISION, out)
    print(out)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("cmd", choices=["g0", "prereg", "parity", "predict", "seal", "evaluate", "decide"])
    ap_.add_argument("--seed", type=int)
    a_ = ap_.parse_args()
    if a_.cmd == "predict":
        predict(a_.seed)
    else:
        {"g0": g0, "prereg": prereg, "parity": parity, "seal": seal, "evaluate": evaluate, "decide": decide}[a_.cmd]()
