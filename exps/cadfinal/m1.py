# exps/cadfinal/m1.py
"""CAD-RESET-FINAL-01 step M1: full-MVTec-test measurement of the canonical baseline (RH-01b R15, 3 seeds).

No training. BENCHMARK_DESCRIPTIVE / EXPLORATORY (the MVTec official test is historically exposed).

  prereg   M1_PREREG.json: reading rule, metrics, sources (before any mask)
  parity   on-the-fly R_D maps vs the stored RH-01b maps of the 60 P1-DEV images (label-free)
  score    exps.cadfinal.dino.score('mvtec', seed) for the three seeds (label-free)
  lock     M1_LOCK.json: hashes of every score file
  evaluate the single mask pass: per-category native Pixel AP at acquisition and final, macro, pooled
           (histogram), Image AUROC, forgetting; seed-mean with a paired image bootstrap; full test and
           without P1-DEV -> M1_EVAL.json
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.cadfinal import common as K
from exps.cadfinal import metrics as M
from exps.task_p.p0_normal_support import require

PREREG, LOCK, EVAL = K.OUT / "M1_PREREG.json", K.OUT / "M1_LOCK.json", K.OUT / "M1_EVAL.json"
SRC = ["exps/cadfinal/m1.py", "exps/cadfinal/common.py", "exps/cadfinal/metrics.py", "exps/cadfinal/dino.py",
       "exps/dinomaly/readout.py", "exps/dinomaly/official.py", "exps/dinomaly/encoder.py", "exps/task_p/p1_readout.py",
       "exps/calm/stats.py", "exps/calm02/native.py", "exps/calm02/data.py"]
REPS, SEED0, BINS, POOL_BINS = 1000, 20261104, 8192, 65536
RULE = ("Retention headroom of the baseline on the full MVTec test (seed-mean macro native forgetting "
        "F = mean_{j<15} AP[j,j] - AP[15,j], 95% paired image-bootstrap CI): MATERIAL iff F >= 0.02 and CI lower > 0; "
        "SMALL iff CI upper < 0.02; otherwise AMBIGUOUS. Descriptive; the official test is historically exposed.")


def prereg():
    require(not PREREG.exists(), "exists")
    K.write_json(PREREG, {"rule": RULE, "sources": {s: K.sha(K.ROOT / s) for s in SRC},
                          "protocol_lock_v1": K.sha(K.OUT / "PROTOCOL_LOCK_v1.json"),
                          "metrics": "exact native Pixel AP (ties grouped), masks never resized; R_D 256^2 maps bilinear "
                                     "to native; Image AUROC on the official top-1% R_D score; pooled AP from 65,536-bin "
                                     "histograms; bootstrap 1000 reps stratified within task x label, seed 20261104+j",
                          "cohorts": ["full", "no_p1dev"], "seeds": K.SEEDS, "utc": K.now()})


def verify_prereg():
    p = K.read(PREREG)
    bad = [s for s, d in p["sources"].items() if K.sha(K.ROOT / s) != d]
    require(not bad and p["rule"] == RULE, f"changed after M1 prereg: {bad}")


@torch.no_grad()
def parity():
    """Label-free: R15 seed-2711 stage-15 maps of the 60 P1-DEV images, on the fly vs RH-01b's stored maps."""
    from exps.cadfinal import dino as DN
    from exps.dinomaly import common as D, encoder as E, readout as R, run as DR
    from exps.rh01 import numerics as NUM
    verify_prereg()
    DR.configure(); NUM.use(True)
    ref = torch.load(K.ROOT / "results/hope_cad/rh01b/scores/maps/R15_seed2711_stage15.pt", weights_only=True)
    enc = E.load(D.CKPT["base"][0], "base", "cuda")
    dec = DN.O_decoder(); dec.load_state_dict(torch.load(DN.checkpoint("mvtec", 2711, 15)[0], weights_only=True)["state"]); dec.eval()
    maps, imgs = [], []
    for i in range(0, len(ref["paths"]), 8):
        x = torch.stack([E.preprocess(K.ROOT / "data/mvtec" / p) for p in ref["paths"][i:i + 8]]).cuda()
        f = enc.groups(x).float(); en, de = dec(f[:, 0], f[:, 1]); s, im = R.dinomaly(en, de)
        maps.append(s.cpu()); imgs.append(im.cpu())
    d_map = float((torch.cat(maps) - ref["s256"]).abs().max()); d_img = float((torch.cat(imgs) - ref["image_RD"]).abs().max())
    out = {"n": len(ref["paths"]), "max_abs_map": d_map, "max_abs_image": d_img,
           "map_scale": float(ref["s256"].abs().max()), "pass": d_map <= 1e-3 and d_img <= 1e-3, "utc": K.now()}
    K.write_json(K.OUT / "M1_PARITY.json", out)
    print(out)
    require(out["pass"], "on-the-fly scoring differs from RH-01b maps")


def score(seed):
    from exps.cadfinal import dino as DN
    verify_prereg()
    require(K.read(K.OUT / "M1_PARITY.json")["pass"], "parity first")
    DN.score("mvtec", seed)


def lock():
    require(not LOCK.exists(), "exists")
    verify_prereg()
    files = {}
    for s in K.SEEDS:
        d = K.OUT / "scores/mvtec" / f"DINOMALY_R_seed{s}"
        for p in sorted(d.glob("*")):
            files[str(p.relative_to(K.OUT))] = K.sha(p)
    require(len(files) == 3 * 31, f"incomplete scores: {len(files)}")
    K.write_json(LOCK, {"prereg": K.sha(PREREG), "files": files, "utc": K.now()})


def evaluate():
    verify_prereg()
    lk = K.read(LOCK)
    require(not EVAL.exists(), "one-shot")
    require(all(K.sha(K.OUT / n) == d for n, d in lk["files"].items()), "post-lock change")
    ids = K.benchmark_identities("mvtec")
    names, T, root = [t["task"] for t in ids["tasks"]], len(ids["tasks"]), K.ROOT / "data/mvtec"
    load = lambda s, kind, j: torch.load(K.OUT / "scores/mvtec" / f"DINOMALY_R_seed{s}" / f"{kind}_{j:02d}.pt", weights_only=True)
    hi_pool = {s: max(float(load(s, "fin", j)["s256"].max()) for j in range(T)) for s in K.SEEDS}
    res = {"rule": RULE, "lock": K.sha(LOCK), "tasks": names, "cohorts": {}}
    for coh in ("full", "no_p1dev"):
        rows_idx = K.cohort_rows("mvtec", coh, ids)
        z = {s: {k: np.zeros(T) for k in ("acq", "fin", "iau", "iau_acq", "binerr")} for s in K.SEEDS}
        B = {s: {k: np.zeros((REPS, T)) for k in ("acq", "fin")} for s in K.SEEDS}
        pool = {s: [np.zeros(POOL_BINS), np.zeros(POOL_BINS)] for s in K.SEEDS}
        for j, t in enumerate(ids["tasks"]):
            I = rows_idx[t["task"]]
            rows = [t["test"][k] for k in I]
            labels = np.array([r["label"] for r in rows])
            hws = [M.native_hw(root / r["relative_path"]) for r in rows]
            masks = [M.native_mask(root / r["mask_path"] if r["mask_path"] else None, hw) for r, hw in zip(rows, hws)]
            W = M.bootstrap_weights(labels, REPS, SEED0 + j)
            for s in K.SEEDS:
                for kind in ("acq", "fin"):
                    c = load(s, kind, j)
                    low, img = c["s256"].numpy()[I], c["image_RD"].numpy()[I]
                    maps = [M.to_native("dinomaly", x, hw) for x, hw in zip(low, hws)]
                    z[s][kind][j] = M.exact_ap(maps, masks)["ap"]
                    z[s]["iau" if kind == "fin" else "iau_acq"][j] = M.auroc(labels, img)
                    lo, hi = max(float(low.min()), 0.0), float(low.max())
                    pos = np.zeros((len(rows), BINS)); tot = np.zeros((len(rows), BINS))
                    for n, (m, y) in enumerate(zip(maps, masks)):
                        pos[n], tot[n] = M.hist(m, y, lo, hi, BINS)
                    B[s][kind][:, j] = [M.ap_from_hist(pos.T @ w, tot.T @ w) for w in W]
                    if kind == "fin":
                        pp, pt = np.zeros(POOL_BINS), np.zeros(POOL_BINS)
                        for m, y in zip(maps, masks):
                            a, b = M.hist(m, y, 0.0, hi_pool[s], POOL_BINS); pp += a; pt += b
                        pool[s][0] += pp; pool[s][1] += pt
                        z[s]["binerr"][j] = abs(M.ap_from_hist(pp, pt) - z[s]["fin"][j])
                    del maps
            print(coh, t["task"], {s: (round(z[s]["acq"][j], 4), round(z[s]["fin"][j], 4)) for s in K.SEEDS}, flush=True)
            del masks
        old = slice(0, T - 1)
        acq = np.mean([z[s]["acq"] for s in K.SEEDS], 0); fin = np.mean([z[s]["fin"] for s in K.SEEDS], 0)
        bacq = np.mean([B[s]["acq"] for s in K.SEEDS], 0); bfin = np.mean([B[s]["fin"] for s in K.SEEDS], 0)
        F = acq[old] - fin[old]
        bF = (bacq[:, old] - bfin[:, old]).mean(1)
        ci_F, ci_A = np.percentile(bF, [2.5, 97.5]).tolist(), np.percentile(bfin.mean(1), [2.5, 97.5]).tolist()
        Fm = float(F.mean())
        reading = "MATERIAL" if Fm >= 0.02 and ci_F[0] > 0 else "SMALL" if ci_F[1] < 0.02 else "AMBIGUOUS"
        per_cat_ci = {names[j]: np.percentile(bacq[:, j] - bfin[:, j], [2.5, 97.5]).tolist() for j in range(T - 1)}
        res["cohorts"][coh] = {
            "n_images": sum(len(v) for v in rows_idx.values()),
            "final_ap": dict(zip(names, fin.tolist())), "acquisition_ap": dict(zip(names, acq.tolist())),
            "forgetting": dict(zip(names[:-1], F.tolist())), "forgetting_ci95": per_cat_ci,
            "macro_final_ap": float(fin.mean()), "macro_final_ap_ci95": ci_A,
            "macro_acquisition_ap": float(acq.mean()),
            "macro_forgetting": Fm, "macro_forgetting_ci95": ci_F, "headroom_reading": reading,
            "worst_final": [names[int(fin.argmin())], float(fin.min())], "worst_forgetting": [names[int(F.argmax())], float(F.max())],
            "macro_image_auroc": float(np.mean([z[s]["iau"] for s in K.SEEDS])),
            "macro_image_auroc_acq": float(np.mean([z[s]["iau_acq"] for s in K.SEEDS])),
            "image_auroc": dict(zip(names, np.mean([z[s]["iau"] for s in K.SEEDS], 0).tolist())),
            "pooled_final_ap_hist": float(np.mean([M.ap_from_hist(*pool[s]) for s in K.SEEDS])),
            "pooled_binning_error_max": float(max(z[s]["binerr"].max() for s in K.SEEDS)),
            "per_seed": {str(s): {"macro_final_ap": float(z[s]["fin"].mean()),
                                  "macro_forgetting": float((z[s]["acq"][old] - z[s]["fin"][old]).mean()),
                                  "macro_image_auroc": float(z[s]["iau"].mean())} for s in K.SEEDS}}
    K.write_json(EVAL, res)
    print({c: {k: v for k, v in r.items() if k.startswith("macro") or k in ("headroom_reading", "pooled_final_ap_hist")}
           for c, r in res["cohorts"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "parity", "score", "lock", "evaluate"])
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    {"prereg": prereg, "parity": parity, "lock": lock, "evaluate": evaluate}.get(a.cmd, lambda: score(a.seed))()
