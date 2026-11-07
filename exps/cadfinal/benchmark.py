# exps/cadfinal/benchmark.py
"""CAD-RESET-FINAL-01 benchmark driver. Irreversible steps last; every output is create-exclusive.

  freeze                         FREEZE.json: sources, configurations, manifests, seeds, selection and reporting
                                 rules (before any training of this task)
  train-memory / score-memory    label-free (exps.cadfinal.memory)
  cache-visa / train-visa / score-dino   label-free (exps.cadfinal.dino)
  lock                           LOCK.json: FREEZE + every state/checkpoint and score file
  evaluate --cohort visa_dev     reads the 240 VisA-DEV masks only; writes SELECTION.json (declared rule)
  evaluate --cohort visa_final   requires SELECTION.json; reads the VisA-FINAL masks once
  evaluate --cohort mvtec        MVTec official test, BENCHMARK_DESCRIPTIVE (full and without P1-DEV)
  timing                         uncontended latency/update probes (no labels)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from exps.cadfinal import common as K
from exps.cadfinal import metrics as M
from exps.task_p.p0_normal_support import require

FREEZE, LOCK, SELECTION = K.OUT / "FREEZE.json", K.OUT / "LOCK.json", K.OUT / "SELECTION.json"
REPS, BOOT_SEED, BINS, POOL_BINS = 1000, 20261104, 8192, 65536
MARGIN, SELECT_MARGIN = 0.005, 0.01
SOURCES = ["exps/cadfinal/common.py", "exps/cadfinal/metrics.py", "exps/cadfinal/memory.py", "exps/cadfinal/dino.py",
           "exps/cadfinal/benchmark.py", "exps/cadfinal/test_cadfinal.py",
           "exps/calm01/arms.py", "exps/calm01/calm.py", "exps/calm01/closest_pair.py", "exps/calm01/storage.py",
           "exps/calm02/score.py", "exps/calm02/native.py", "exps/calm02/data.py", "exps/calm02/extract.py",
           "exps/calm/stats.py", "exps/task_p/p1_readout.py", "exps/task_p/p0_normal_support.py",
           "exps/dinomaly/official.py", "exps/dinomaly/encoder.py", "exps/dinomaly/readout.py", "exps/dinomaly/run.py",
           "exps/dinomaly/common.py", "exps/rh01/run.py", "exps/rh01/common.py", "exps/rh01/numerics.py",
           "exps/rh01b/run.py", "exps/rh01b/common.py", "models/feature_extractors/cadic_vit_v1.py",
           "dataset/benchmark_protocol_v1.py", K.TASK]
ARMS = {  # instance -> (family, kind, seeds or None)
    "DINOMALY_R": ("A", "dinomaly", K.SEEDS),
    "CALM_L1_B1": ("B", "patch", None), "CALM_L1_B2": ("B", "patch", None),
    "CADIC_COMPAT_B1": ("C", "patch", None), "CADIC_COMPAT_B2": ("C", "patch", None)}
CONTRASTS = [("DINOMALY_R", "CALM_L1_B2"), ("DINOMALY_R", "CALM_L1_B1"), ("DINOMALY_R", "CADIC_COMPAT_B2"),
             ("DINOMALY_R", "CADIC_COMPAT_B1"), ("CALM_L1_B2", "CADIC_COMPAT_B2"), ("CALM_L1_B1", "CADIC_COMPAT_B1"),
             ("CALM_L1_B2", "CALM_L1_B1"), ("CADIC_COMPAT_B2", "CADIC_COMPAT_B1")]
RULES = {
    "selection": "On VisA-DEV only: final macro native Pixel AP (Dinomaly = mean of 3 seeds) of family A "
                 "(DINOMALY_R) vs family B at its CADIC-paper budget (CALM_L1_B2, fixed a priori). Select A iff "
                 "A - B >= +0.01; select B iff A - B <= -0.01; otherwise select the candidate with fewer persistent "
                 "bytes (B). Written to SELECTION.json before any VisA-FINAL mask is read. No configuration changes.",
    "contrast_label": "Per contrast and endpoint on VisA-FINAL (paired image bootstrap, 1000 reps, stratified within "
                      "task x label): BETTER/LOWER iff |diff| >= 0.005 and the 95% CI excludes 0 in that direction; "
                      "otherwise NO_DIFFERENCE_ESTABLISHED. A small family of contrasts; no multiplicity correction.",
    "ranking": "Rankings only within one benchmark and cohort, same training identities, same test images, "
               "same metric code. Readouts and backbones are part of each method and stated.",
    "mvtec": "BENCHMARK_DESCRIPTIVE (historically exposed official test); never used for selection; reported on "
             "the full test and without the 60 P1-DEV images.",
    "forgetting": "native AP[j,j] - AP[T,j] over tasks j < T (acquisition -> final), macro and worst",
    "pooled": "pooled native Pixel AP from 65,536-bin histograms on each arm's global score range; binning error "
              "reported per category against the exact AP",
    "image_auroc": "Dinomaly: official top-1% mean of the R_D map; memories: CADIC Eq. (9) b=9 (primary) and max patch",
    "stop": "integrity / numerical / resource only; an unfavourable result never stops or repeats a run"}


def sources():
    return {s: K.sha(K.ROOT / s) for s in SOURCES}


def manifests():
    out = {}
    for b in ("mvtec", "visa"):
        ids = K.benchmark_identities(b)
        out[b] = {"order": [t["task"] for t in ids["tasks"]],
                  "train_sha256": K.text_sha("\n".join(r["relative_path"] for t in ids["tasks"] for r in t["train"])),
                  "test_sha256": K.text_sha("\n".join(r["relative_path"] for t in ids["tasks"] for r in t["test"])),
                  "n_train": sum(len(t["train"]) for t in ids["tasks"]), "n_test": sum(len(t["test"]) for t in ids["tasks"])}
    sp = K.visa_split()
    out["visa_split"] = {"dev_sha256": K.text_sha("\n".join(p for t in sp.values() for p in t["dev"])),
                         "final_sha256": K.text_sha("\n".join(p for t in sp.values() for p in t["final"])),
                         "n_dev": sum(len(t["dev"]) for t in sp.values()), "n_final": sum(len(t["final"]) for t in sp.values()),
                         "dev": {k: v["dev"] for k, v in sp.items()}}
    return out


def freeze():
    from exps.dinomaly import common as D
    require(not FREEZE.exists(), "FREEZE exists")
    K.write_json(FREEZE, {
        "task": K.TASK, "sources": sources(), "manifests": manifests(), "arms": ARMS, "contrasts": CONTRASTS,
        "budgets": K.BUDGETS, "memory_arms": K.MEMORY_ARMS, "seeds": K.SEEDS, "rules": RULES,
        "constants": {"reps": REPS, "boot_seed": BOOT_SEED, "bins": BINS, "pool_bins": POOL_BINS, "margin": MARGIN,
                      "select_margin": SELECT_MARGIN, "dev_per_label": K.DEV_PER_LABEL},
        "encoders": {"vit_b8": K.sha(K.ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"),
                     "dinov2_b_reg4": K.sha(D.CKPT["base"][0])},
        "feature_manifests": {b: K.sha(K.ROOT / f"results/hope_cad/calm02/features/{b}/MANIFEST.json") for b in ("mvtec", "visa")},
        "r15_lock": K.sha(K.ROOT / "results/hope_cad/rh01b/PREREG_LOCK.json"), "frozen_utc": K.now()})
    print("frozen", K.sha(FREEZE))


def verify_freeze():
    fz = K.read(FREEZE)
    bad = [s for s, d in fz["sources"].items() if K.sha(K.ROOT / s) != d]
    require(not bad, f"source changed after FREEZE: {bad}")
    require(fz["arms"] == {k: [v[0], v[1], list(v[2]) if v[2] else None] for k, v in ARMS.items()}
            and fz["rules"] == RULES, "configuration changed after FREEZE")
    return fz


# --- score loading ------------------------------------------------------------------------------------------

def score_files(bench, inst):
    fam, kind, seeds = ARMS[inst]
    if kind == "dinomaly":
        return {s: K.OUT / "scores" / bench / f"DINOMALY_R_seed{s}" for s in seeds}
    return {None: K.OUT / "runs" / bench / inst}


def load_cell(bench, inst, seed, kind, j):
    d = score_files(bench, inst)[seed]
    if ARMS[inst][1] == "dinomaly":
        z = torch.load(d / f"{kind}_{j:02d}.pt", weights_only=True)
        return z["s256"].numpy(), z["image_RD"].numpy(), None
    z = np.load(d / f"scores_{kind}_{j:02d}.npz")
    return z["patch"], z["image_cadic"], z["image_max"]


def lock():
    require(not LOCK.exists(), "LOCK exists")
    fz = verify_freeze()
    files = {}
    for bench in ("mvtec", "visa"):
        T = len(K.ORDERS[bench])
        for inst, (fam, kind, seeds) in ARMS.items():
            for seed, d in score_files(bench, inst).items():
                req = [f"{k}_{j:02d}.pt" for k in ("acq", "fin") for j in range(T)] if kind == "dinomaly" else \
                      [f"scores_{k}_{j:02d}.npz" for k in ("acq", "fin") for j in range(T)] + [f"task_{j:02d}.pt" for j in range(T)]
                for name in req + ["SCORES.json"] + ([] if kind == "dinomaly" else ["TRAIN.json"]):
                    require((d / name).exists(), f"incomplete: {d / name}")
                    files[str((d / name).relative_to(K.OUT))] = K.sha(d / name)
        if bench == "visa":
            for s in K.SEEDS:
                d = K.OUT / "runs/visa" / f"DINOMALY_R_seed{s}"
                for p in sorted(d.glob("*")):
                    files[str(p.relative_to(K.OUT))] = K.sha(p)
    K.write_json(LOCK, {"freeze_sha256": K.sha(FREEZE), "files": files, "locked_utc": K.now(),
                        "cache_visa_dinomaly": K.sha(K.OUT / "CACHE_visa_dinomaly.json")})
    print("locked", len(files), K.sha(LOCK))


def verify_lock():
    lk = K.read(LOCK)
    require(lk["freeze_sha256"] == K.sha(FREEZE), "FREEZE changed")
    verify_freeze()
    bad = [n for n, d in lk["files"].items() if K.sha(K.OUT / n) != d]
    require(not bad, f"post-LOCK change: {bad[:5]}")
    return lk


# --- evaluation ---------------------------------------------------------------------------------------------

def _boot(pos, tot, W):
    return np.array([M.ap_from_hist(pos.T @ w, tot.T @ w) for w in W])


def evaluate(cohort):
    from PIL import Image  # noqa: F401
    verify_lock()
    bench = "mvtec" if cohort.startswith("mvtec") else "visa"
    if cohort == "visa_final":
        require(SELECTION.exists(), "SELECTION.json must exist before VisA-FINAL is read")
    target = K.OUT / f"EVAL_{cohort}.json"
    require(not target.exists(), "one-shot evaluation")
    ids = K.benchmark_identities(bench)
    names, T, root = [t["task"] for t in ids["tasks"]], len(ids["tasks"]), K.ROOT / ("data/visa" if bench == "visa" else "data/mvtec")
    subsets = {"mvtec": ["full", "no_p1dev"], "visa_dev": ["dev"], "visa_final": ["final"]}[cohort]
    insts = list(ARMS)
    seeds_of = {i: (list(ARMS[i][2]) if ARMS[i][2] else [None]) for i in insts}
    res = {"cohort": cohort, "bench": bench, "tasks": names, "lock_sha256": K.sha(LOCK), "subsets": {}}
    # global score ranges for pooled AP (bilinear maps stay within the low-res range)
    rng_pool = {}
    for i in insts:
        for s in seeds_of[i]:
            lo, hi = np.inf, -np.inf
            for j in range(T):
                m, _, _ = load_cell(bench, i, s, "fin", j)
                lo, hi = min(lo, float(m.min())), max(hi, float(m.max()))
            rng_pool[(i, s)] = (max(lo, 0.0), hi)
    for sub in subsets:
        rows_idx = K.cohort_rows(bench, sub, ids)
        cell = {(i, s): {"acq": np.zeros(T), "fin": np.zeros(T), "iau": np.zeros(T), "iau_max": np.zeros(T),
                         "iau_acq": np.zeros(T), "boot_acq": np.zeros((REPS, T)), "boot_fin": np.zeros((REPS, T)),
                         "pool_pos": np.zeros(POOL_BINS), "pool_tot": np.zeros(POOL_BINS), "bin_err": np.zeros(T)}
                for i in insts for s in seeds_of[i]}
        for j, t in enumerate(ids["tasks"]):
            I = rows_idx[t["task"]]
            rows = [t["test"][k] for k in I]
            labels = np.array([r["label"] for r in rows])
            hws = [M.native_hw(root / r["relative_path"]) for r in rows]
            masks = [M.native_mask(root / r["mask_path"] if r["mask_path"] else None, hw) for r, hw in zip(rows, hws)]  # mask read
            W = M.bootstrap_weights(labels, REPS, BOOT_SEED + j)
            for i in insts:
                kindmap = ARMS[i][1]
                for s in seeds_of[i]:
                    c = cell[(i, s)]
                    for kind in ("acq", "fin"):
                        lowres, img, imax = load_cell(bench, i, s, kind, j)
                        lowres = lowres[I]
                        maps = [M.to_native(kindmap, x, hw) for x, hw in zip(lowres, hws)]
                        c[kind][j] = M.exact_ap(maps, masks)["ap"]
                        lo, hi = max(float(lowres.min()), 0.0), float(lowres.max())
                        pos = np.zeros((len(rows), BINS)); tot = np.zeros((len(rows), BINS))
                        for n, (m, y) in enumerate(zip(maps, masks)):
                            pos[n], tot[n] = M.hist(m, y, lo, hi, BINS)
                        c[f"boot_{kind}"][:, j] = _boot(pos, tot, W)
                        c["iau_acq" if kind == "acq" else "iau"][j] = M.auroc(labels, img[I])
                        if kind == "fin":
                            if imax is not None:
                                c["iau_max"][j] = M.auroc(labels, imax[I])
                            plo, phi = rng_pool[(i, s)]
                            pp, pt = np.zeros(POOL_BINS), np.zeros(POOL_BINS)
                            for m, y in zip(maps, masks):
                                a, b = M.hist(m, y, plo, phi, POOL_BINS)
                                pp += a; pt += b
                            c["pool_pos"] += pp; c["pool_tot"] += pt
                            c["bin_err"][j] = abs(M.ap_from_hist(pp, pt) - c["fin"][j])
                        del maps
            print(cohort, sub, t["task"], {f"{i}{'' if s is None else s}": round(cell[(i, s)]["fin"][j], 4)
                                          for i in insts for s in seeds_of[i]}, flush=True)
            del masks
        res["subsets"][sub] = summarize(cell, insts, seeds_of, names)
    if cohort == "visa_dev":
        res["selection"] = select(res["subsets"]["dev"])
    K.write_json(target, res)
    if cohort == "visa_dev":
        K.write_json(SELECTION, {"eval_dev_sha256": K.sha(target), "rule": RULES["selection"], **res["selection"],
                                 "utc": K.now()})
    print("written", target.name, K.sha(target))


def summarize(cell, insts, seeds_of, names):
    T, old = len(names), slice(0, len(names) - 1)
    arms, boots = {}, {}
    for i in insts:
        cs = [cell[(i, s)] for s in seeds_of[i]]
        mean = lambda key: np.mean([c[key] for c in cs], axis=0)
        acq, fin = mean("acq"), mean("fin")
        b_acq, b_fin = mean("boot_acq"), mean("boot_fin")
        forget = acq[old] - fin[old]
        pooled = [M.ap_from_hist(c["pool_pos"], c["pool_tot"]) for c in cs]
        boots[i] = {"fin": b_fin.mean(1), "forget": (b_acq[:, old] - b_fin[:, old]).mean(1)}
        arms[i] = {
            "family": ARMS[i][0], "seeds": [s for s in seeds_of[i] if s is not None],
            "final_ap": dict(zip(names, fin.tolist())), "acquisition_ap": dict(zip(names, acq.tolist())),
            "macro_final_ap": float(fin.mean()), "macro_final_ap_ci95": np.percentile(boots[i]["fin"], [2.5, 97.5]).tolist(),
            "macro_forgetting": float(forget.mean()), "macro_forgetting_ci95": np.percentile(boots[i]["forget"], [2.5, 97.5]).tolist(),
            "forgetting": dict(zip(names[:-1], forget.tolist())),
            "worst_final_ap": [names[int(fin.argmin())], float(fin.min())],
            "worst_forgetting": [names[int(forget.argmax())], float(forget.max())],
            "pooled_final_ap_hist": float(np.mean(pooled)), "pooled_per_seed": pooled,
            "pooled_binning_error_max": float(max(c["bin_err"].max() for c in cs)),
            "macro_image_auroc": float(mean("iau").mean()), "image_auroc": dict(zip(names, mean("iau").tolist())),
            "macro_image_auroc_acq": float(mean("iau_acq").mean()),
            "macro_image_auroc_max_patch": float(mean("iau_max").mean()) if ARMS[i][1] == "patch" else None}
        if len(cs) > 1:
            arms[i]["per_seed_macro_final_ap"] = [float(c["fin"].mean()) for c in cs]
            arms[i]["per_seed_macro_forgetting"] = [float((c["acq"][old] - c["fin"][old]).mean()) for c in cs]
    con = {}
    for a, b in CONTRASTS:
        out = {}
        for ep, key, up, down in (("final_ap", "fin", "BETTER", "WORSE"), ("forgetting", "forget", "HIGHER", "LOWER")):
            d = boots[a][key] - boots[b][key]
            pt = (arms[a]["macro_final_ap"] - arms[b]["macro_final_ap"]) if key == "fin" else \
                 (arms[a]["macro_forgetting"] - arms[b]["macro_forgetting"])
            ci = np.percentile(d, [2.5, 97.5]).tolist()
            lab = up if pt >= MARGIN and ci[0] > 0 else down if pt <= -MARGIN and ci[1] < 0 else "NO_DIFFERENCE_ESTABLISHED"
            out[ep] = {"diff": float(pt), "ci95": ci, "label": lab}
        con[f"{a} - {b}"] = out
    return {"arms": arms, "contrasts": con}


def select(dev):
    a, b = dev["arms"]["DINOMALY_R"]["macro_final_ap"], dev["arms"]["CALM_L1_B2"]["macro_final_ap"]
    d = a - b
    pick = "DINOMALY_R" if d >= SELECT_MARGIN else "CALM_L1_B2" if d <= -SELECT_MARGIN else "CALM_L1_B2"
    why = "A - B >= +0.01" if d >= SELECT_MARGIN else "A - B <= -0.01" if d <= -SELECT_MARGIN else \
          "|A - B| < 0.01: fewer persistent bytes"
    return {"dev_macro_final_ap": {"DINOMALY_R": a, "CALM_L1_B2": b}, "diff": d, "selected": pick, "reason": why}


# --- timing (uncontended, label-free) -------------------------------------------------------------------------

@torch.no_grad()
def timing(n=40, warm=5):
    """Per-image inference latency (batch 1, CUDA synchronised) and per-step / per-image update cost on VisA
    candle test/train images. Scores are not stored and no label is read."""
    from exps.calm01 import arms as A
    from exps.calm02.score import Scorer, decode
    from exps.calm02.extract import extractor
    from exps.cadfinal import dino as DN, memory as MEM
    from exps.dinomaly import common as D, encoder as E, readout as R, run as DR
    from exps.dinomaly import official as O
    from exps.rh01 import numerics as NUM, run as R1
    target = K.OUT / "TIMING.json"
    require(not target.exists(), "timing exists")
    DR.configure(); NUM.use(True)
    ids = K.benchmark_identities("visa")
    t0r = ids["tasks"][0]
    root = K.ROOT / "data/visa"
    paths = [root / r["relative_path"] for r in t0r["test"][:n + warm]]
    sync = torch.cuda.synchronize
    out = {"device": torch.cuda.get_device_name(0), "n": n, "warmup": warm, "batch": 1}
    # Dinomaly inference: preprocess + encoder + decoder + R_D map
    enc = E.load(D.CKPT["base"][0], "base", "cuda")
    dec = O.Decoder(768, 12).cuda().eval()
    dec.load_state_dict(torch.load(DN.checkpoint("visa", K.SEEDS[0], len(ids["tasks"]))[0], weights_only=True)["state"])
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for k, p in enumerate(paths):
        sync(); t = time.perf_counter()
        x = E.preprocess(p)[None].cuda(); f = enc.groups(x).float(); en, de = dec(f[:, 0], f[:, 1]); R.dinomaly(en, de)
        sync(); ts.append(time.perf_counter() - t)
    out["DINOMALY_R"] = {"latency_s_median": float(np.median(ts[warm:])), "latency_s_p95": float(np.percentile(ts[warm:], 95)),
                         "peak_infer_mib": torch.cuda.max_memory_allocated() / 2**20}
    # Dinomaly update: s/step on a VisA task with replay (20 steps after 5 warm-up), training memory
    feats = DN.load_visa(ids["tasks"][1]["task"]); rep = DN.load_visa(ids["tasks"][0]["task"])[:24]
    d2 = DR.new_decoder(1); gen = torch.Generator().manual_seed(1)
    R1.train_stage_guarded(d2, feats, 5, gen, rep); sync(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    R1.train_stage_guarded(d2, feats, 20, gen, rep); sync()
    out["DINOMALY_R"].update({"update_s_per_step": (time.perf_counter() - t) / 20, "steps_per_task": DN.ITERS,
                              "peak_train_mib": torch.cuda.max_memory_allocated() / 2**20})
    del enc, dec, d2, feats, rep; torch.cuda.empty_cache()
    # memory arms: ViT-B/8 extraction + exact 1-NN + Eq. 9 against the final VisA bank
    vit = extractor("cuda")
    from dataset.benchmark_protocol_v1 import _ImageDataset
    from exps.calm02.data import MEAN, STD
    ds = _ImageDataset(root, t0r["test"][:n + warm], 224, MEAN, STD, training=True)
    for inst in ("CALM_L1_B1", "CALM_L1_B2", "CADIC_COMPAT_B1", "CADIC_COMPAT_B2"):
        ck = torch.load(K.OUT / "runs/visa" / inst / f"task_{len(ids['tasks']) - 1:02d}.pt", map_location="cpu", weights_only=False)
        sc = Scorer(decode(ck["state"])[ck["occupied"]], "cuda")
        torch.cuda.reset_peak_memory_stats(); ts = []
        for k in range(n + warm):
            sync(); t = time.perf_counter()
            f = vit.extract_patch_features(ds[k]["images"][None].cuda()).float().cpu()[0]
            d, idx = sc.patch(f); sc.image_cadic(f, d, idx)
            sync(); ts.append(time.perf_counter() - t)
        tr = K.read(K.OUT / "runs/visa" / inst / "TRAIN.json")
        out[inst] = {"latency_s_median": float(np.median(ts[warm:])), "latency_s_p95": float(np.percentile(ts[warm:], 95)),
                     "peak_infer_mib": torch.cuda.max_memory_allocated() / 2**20,
                     "update_s_per_image_stream_mean": sum(x["seconds"] for x in tr["tasks"]) / sum(x["images"] for x in tr["tasks"]),
                     "note": "update time from the recorded stream (feature extraction excluded; see report)"}
        del sc
    K.write_json(target, {**out, "utc": K.now()})
    print("timing written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "train-memory", "score-memory", "cache-visa", "train-visa", "score-dino",
                                    "lock", "evaluate", "timing"])
    ap.add_argument("--bench", choices=["mvtec", "visa"])
    ap.add_argument("--arm", choices=list(K.MEMORY_ARMS))
    ap.add_argument("--budget", choices=list(K.BUDGETS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cohort", choices=["visa_dev", "visa_final", "mvtec"])
    a = ap.parse_args()
    from exps.cadfinal import dino as DN, memory as MEM
    if a.cmd == "freeze":
        freeze()
    elif a.cmd in ("train-memory", "score-memory"):
        verify_freeze()
        (MEM.train if a.cmd == "train-memory" else MEM.score)(a.bench, a.arm, a.budget)
    elif a.cmd == "cache-visa":
        verify_freeze(); DN.cache_visa()
    elif a.cmd == "train-visa":
        verify_freeze(); DN.train_visa(a.seed)
    elif a.cmd == "score-dino":
        verify_freeze(); DN.score(a.bench, a.seed)
    elif a.cmd == "lock":
        lock()
    elif a.cmd == "evaluate":
        evaluate(a.cohort)
    elif a.cmd == "timing":
        verify_lock(); timing()


if __name__ == "__main__":
    main()
