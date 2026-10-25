# exps/ptag/run.py
"""PTAG-01 stages: manifest -> pilot -> lock -> generate -> score -> proxies -> analyze -> audit.

Run: python -m exps.ptag.run <stage> [--device cuda]. Stage outputs are create-exclusive; `analyze` is one-shot.
No real defect mask or defect image is opened by any stage.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from exps.ptag import common as C
from exps.ptag import metrics as M
from exps.ptag import scoring as S
from exps.ptag import synth as Y
from exps.task_p.p0_normal_support import require
from exps.task_p.p2.metrics import bootstrap_bounds

OUT = C.OUT
SOURCES = ["exps/ptag/__init__.py", "exps/ptag/common.py", "exps/ptag/synth.py", "exps/ptag/scoring.py",
           "exps/ptag/metrics.py", "exps/ptag/run.py", "exps/ptag/parity.py", "exps/ptag/analysis.py",
           "exps/ptag/test_ptag.py",
           "agents/tasks/task_ptag_01_alignment_gate.md", "dataset/anomaly_generators/perlin.py",
           "dataset/anomaly_generators/base.py", "exps/task_p/p0_normal_support.py", "exps/task_p/p1_readout.py",
           "exps/task_p/p2/metrics.py", "exps/task_p/p2/model.py", "exps/task_p/p2d/model.py",
           "exps/task_p/p2e/model.py", "exps/task_p/p2e/data.py", "exps/task_p/p2f/model.py", "exps/task_p/p2f/data.py",
           "exps/calm/decompose.py", "exps/calm/load.py", "exps/calm02/score.py", "exps/calm02/extract.py",
           "exps/ntm/stage0.py", "models/feature_extractors/cadic_vit_v1.py"]


def gpu_peak():
    return torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0


# ---------------------------------------------------------------- Stage A

def manifest():
    man = C.build_manifest()
    man["hosts"] = C.select_hosts()
    man["contrasts"] = {s: [[f, n, c, p] for f, n, c, p in C.contrasts(man, s)] for s in ("selection", "heldout")}
    states = set()
    for cell in man["cells"]:
        sc = cell["scorer"]
        if sc["kind"] == "p2":
            states.add(sc["state_file"])
        elif sc["kind"] == "nn64":
            states |= {f"results/hope_cad/task_p/p1/states_t{t}.pt" for t in (100, 200, 300, 350)}
        else:
            states |= {f"results/hope_cad/calm02/runs/mvtec/{a}_B1/task_{k:02d}.pt"
                       for a in C.CALM_ARMS for k in range(C.CALM_FINAL + 1)}
    real = {cell["real_source"][0] for cell in man["cells"]} | {cell["rep_source"][0] for cell in man["cells"]
                                                                 if cell["rep_source"]}
    real |= {"results/hope_cad/ntm/stage0/prepared.json"} | {f"results/hope_cad/calm02/runs/mvtec/{a}_B1/train_log.json"
                                                            for a in C.CALM_ARMS}
    man["sha256"] = {p: C.sha(C.ROOT / p) for p in sorted(states | real)}
    man["excluded"] = {
        "P2h": "no stored real AP (stopped RESOURCE_INFEASIBLE / exploratory)",
        "NPIM, CMS-NM, NCSL, CSG, CCNM, AD-02": "outside the pre-registered PTAG design list; not added",
        "real-AP replicates": "only P1 and P2 family store paired replicates; C00/CALM/NTM point-only"}
    man["retrospective"] = "all real AP values come from the consumed P1 development cohort and are reused, not re-measured"
    C.write_json(OUT / "MANIFEST.json", man)
    by = {}
    for cell in man["cells"]:
        by.setdefault(f"{cell['family']}/{cell['split']}", 0)
        by[f"{cell['family']}/{cell['split']}"] += 1
    print(json.dumps({"cells": by, "hosts": {c: v["pool_size"] for c, v in man["hosts"].items()},
                      "contrasts": {s: len(v) for s, v in man["contrasts"].items()}}, indent=1))


def pilot(device):
    """Cost projection on clean hosts and random masks only: no proxy is computed against any detector set."""
    man = C.read(OUT / "MANIFEST.json")
    t = {}
    torch.cuda.reset_peak_memory_stats()
    model = Y.extractor(device)
    imgs = [Y.load_rgb(p) for p in man["hosts"]["carpet"]["hosts"][:8]]
    s0 = time.perf_counter(); X = Y.extract(model, imgs, device); t["extract_per_image"] = (time.perf_counter() - s0) / 8
    del model
    X = X.repeat(2, 1, 1)
    cell = next(c for c in man["cells"] if c["id"] == "P1/FULL784_GREEDY/t350/carpet")
    s0 = time.perf_counter(); S.NN64(S.nn64_bank(cell["scorer"]["bank"]), device)(X); t["nn64_per_image"] = (time.perf_counter() - s0) / 16
    rows, own, ids = S.calm_bank("CALM_L1", 14)
    s0 = time.perf_counter()
    S.calm_scores(rows, own, ids, X, np.zeros(16, int), [-1], device)
    t["calm_per_image"] = (time.perf_counter() - s0) / 16
    p2 = next(c for c in man["cells"] if c["family"] == "P2" and c["category"] == "carpet")
    s0 = time.perf_counter(); S.p2_scores(S.p2_learner(p2["scorer"], device), X, device); t["p2_per_cell16"] = time.perf_counter() - s0
    g = torch.Generator().manual_seed(0)
    patch = torch.rand(120, 784, generator=g) * 10
    labels = torch.zeros(120, 1024, 1024, dtype=torch.bool)
    labels[20:, 300:420, 300:420] = True
    s0 = time.perf_counter()
    maps = M.native_maps(patch, (1024, 1024), device)
    M.exact_ap(maps, labels.to(device))
    lo, hi = bootstrap_bounds(list(maps), list(labels.to(device)), M.host_weights(1, 20)[:, np.r_[0:20, np.repeat(np.arange(20), 5)]])
    torch.cuda.synchronize(); t["ap_cell_family"] = time.perf_counter() - s0
    n = {k: sum(1 for c in man["cells"] if c["scorer"]["kind"] == k) for k in ("nn64", "calm", "ntm", "p2")}
    images = C.HOSTS * (1 + C.VARIANTS * len(C.FAMILIES))
    proj = {"extract_s": t["extract_per_image"] * len(C.CATS) * C.HOSTS * (1 + 2 * C.VARIANTS),
            "nn64_s": t["nn64_per_image"] * n["nn64"] * images, "calm_s": t["calm_per_image"] * n["calm"] * images,
            "p2_s": t["p2_per_cell16"] / 16 * n["p2"] * images,
            "ap_s": t["ap_cell_family"] * len(man["cells"]) * len(C.FAMILIES)}
    proj["total_gpu_hours"] = sum(proj.values()) / 3600
    res = {"timings": t, "cells_by_kind": n, "projection": proj, "peak_vram_mib": gpu_peak(),
           "budget": {"gpu_hours": 4, "vram_mib": 2560}, "within_2x_budget": proj["total_gpu_hours"] <= 8}
    C.write_json(OUT / "PILOT.json", res)
    print(json.dumps(res, indent=1))


def lock():
    man = C.read(OUT / "MANIFEST.json")
    files = {s: C.sha(C.ROOT / s) for s in SOURCES}
    files.update({"results/hope_cad/ptag/MANIFEST.json": C.sha(OUT / "MANIFEST.json"),
                  "results/hope_cad/ptag/PILOT.json": C.sha(OUT / "PILOT.json")})
    dtd = sorted(p.relative_to(C.ROOT).as_posix() for p in (C.ROOT / "data/dtd/images").glob("*/*.*"))
    constants = {k: getattr(C, k) for k in ("SEED", "HOSTS", "VARIANTS", "MAX_ATTEMPTS", "FAMILIES", "PROXIES",
                                            "REPS", "BINS")}
    C.write_json(OUT / "PREREG_LOCK.json", {"sha256": files, "detector_and_real_ap_sha256": man["sha256"],
                                            "dtd_list_sha256": C.sha256_text("\n".join(dtd)), "dtd_files": len(dtd),
                                            "constants": constants, "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def verify_lock(deep=False):
    """Locked digests, except files changed by a recorded engineering amendment (AMENDMENTS.json)."""
    lk = C.read(OUT / "PREREG_LOCK.json")
    digests = dict(lk["sha256"])
    records = [OUT / "AMENDMENTS.json"] + sorted(OUT.glob("AMENDMENT_*.json"))
    for rec in [r for r in records if r.exists()]:                    # amendments chain in order
        for name, a in C.read(rec)["files"].items():
            require(digests[name] == a["locked_sha256"], f"amendment {rec.name} does not chain: {name}")
            digests[name] = a["amended_sha256"]
    for name, digest in digests.items():
        require(C.sha(C.ROOT / name) == digest, f"post-LOCK change: {name}")
    if deep:
        for name, digest in lk["detector_and_real_ap_sha256"].items():
            require(C.sha(C.ROOT / name) == digest, f"frozen artifact changed: {name}")


# ---------------------------------------------------------------- Stage B: data

def generate(device):
    verify_lock(deep=True)
    man = C.read(OUT / "MANIFEST.json")
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, gen = Y.extractor(device), Y.NativePerlin()
    (OUT / "synthetic").mkdir(exist_ok=True)
    report = {}
    for ci, c in enumerate(C.CATS):
        target = OUT / "synthetic" / f"{c}.pt"
        require(not target.exists(), f"exists: {target}")
        start = time.perf_counter()
        hosts = man["hosts"][c]["hosts"]
        imgs = [Y.load_rgb(p) for p in hosts]
        hw = imgs[0].shape[:2]
        require(all(i.shape[:2] == hw for i in imgs), "mixed native sizes")
        clean = Y.extract(model, imgs, device)
        sigma = Y.normal_scale(clean.to(device)) / 768 ** 0.5
        feats, masks, meta, host_of = [clean], [], {"Q1": [], "Q2": [], "Q3": []}, list(range(C.HOSTS))
        q1 = []
        for h in range(C.HOSTS):
            for v in range(C.VARIANTS):
                x, m, info = Y.q1(clean[h], hw, sigma, ci, h, v)
                q1.append(x); masks.append(m); meta["Q1"].append(info); host_of.append(h)
        feats.append(torch.stack(q1))
        for fam in ("Q2", "Q3"):
            out = []
            for h in range(C.HOSTS):
                for v in range(C.VARIANTS):
                    a, m, info = Y.q2(gen, imgs[h], c, ci, h, v) if fam == "Q2" else Y.q3(imgs[h], ci, h, v)
                    out.append(a); masks.append(m); meta[fam].append(info); host_of.append(h)
            feats.append(Y.extract(model, out, device))
        X = torch.cat(feats)
        require(X.shape == (C.HOSTS * 16, 784, 768) and len(masks) == C.HOSTS * 15, "synthetic set shape")
        frac = {fam: [float(m.mean()) for m in masks[j * 100:(j + 1) * 100]] for j, fam in enumerate(C.FAMILIES)}
        parity = feature_parity(c, hosts, clean, model, device)
        # AMENDMENT_2: pipeline identity is bitwise equality on an identically composed CALM-02 batch; host
        # deviation (GPU batch-composition numerics) must not exceed the two historical caches' own disagreement.
        require(parity["calm02_batch16_reextract_max_abs"] == 0.0
                and parity["calm02_max_abs"] <= parity["historical_intercache_max_abs"],
                f"feature parity failure {c}: {parity}")
        torch.save({"features": X, "masks": np.stack([np.packbits(m.ravel()) for m in masks]), "hw": list(hw),
                    "host_of_image": host_of, "sigma": sigma, "meta": meta, "hosts": hosts}, target)
        report[c] = {"sigma": sigma, "normal_scale_m": sigma * 768 ** 0.5, "parity": parity,
                     "mask_fraction": {f: {"min": min(v), "median": float(np.median(v)), "max": max(v)} for f, v in frac.items()},
                     "q2_attempts_max": max(i["attempts"] for i in meta["Q2"]), "seconds": time.perf_counter() - start,
                     "file_sha256": C.sha(target)}
        print(c, json.dumps(report[c]), flush=True)
    report["peak_vram_mib"] = gpu_peak()
    C.write_json(OUT / "GENERATE.json", report)


def feature_parity(cat, hosts, clean, model, device):
    obj = torch.load(C.CALM / f"features/mvtec/{cat}_train.pt", map_location="cpu", weights_only=False, mmap=True)
    calm, rows = obj["features"], obj["rows"]
    idx = {r["relative_path"]: i for i, r in enumerate(rows)}
    a = max(float((clean[j] - calm[idx[p]]).abs().max()) for j, p in enumerate(hosts))
    ff = torch.load(C.ROOT / f"results/hope_cad/frozen_features_v1/clean/{cat}_train.pt", weights_only=True, mmap=True)
    jdx = {p: i for i, p in enumerate(ff["relative_paths"])}
    b = max(float((clean[j] - ff["patches"][jdx[p]]).abs().max()) for j, p in enumerate(hosts))
    inter = max(float((calm[i] - ff["patches"][jdx[r["relative_path"]]]).abs().max()) for i, r in enumerate(rows))
    s = (idx[hosts[0]] // 16) * 16                                     # the CALM-02 batch holding the first host
    f16 = Y.extract(model, [Y.load_rgb(r["relative_path"]) for r in rows[s:s + 16]], device, batch=16)
    return {"calm02_max_abs": a, "frozen_features_v1_max_abs": b, "historical_intercache_max_abs": inter,
            "calm02_batch16_reextract_max_abs": float((f16 - calm[s:s + 16]).abs().max())}


# ---------------------------------------------------------------- Stage B: scoring (with parity gates)

def score(device):
    verify_lock(deep=True)
    man = C.read(OUT / "MANIFEST.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    from exps.ptag.parity import run_parity
    par = run_parity(man, device)
    C.write_json(OUT / "PARITY.json", par)
    require(par["pass"], f"parity failure: {json.dumps(par['max_abs'])}")
    tau, gidx = S.ntm_tau(), C.calm_global_index()
    (OUT / "scores").mkdir(exist_ok=True)
    timing = {}
    for c in C.CATS:
        target = OUT / "scores" / f"{c}.npz"
        require(not target.exists(), f"exists: {target}")
        syn = torch.load(OUT / "synthetic" / f"{c}.pt", weights_only=False)
        X, hoi = syn["features"], np.asarray(syn["host_of_image"])
        host_global = [gidx[p] for p in syn["hosts"]]
        cells = [x for x in man["cells"] if x["category"] == c]
        res, mse = {}, {}
        s0 = time.perf_counter()
        bank_cache = {}
        for x in [x for x in cells if x["scorer"]["kind"] == "nn64"]:
            key = json.dumps(x["scorer"]["bank"])
            if key not in bank_cache:
                bank_cache[key] = S.NN64(S.nn64_bank(x["scorer"]["bank"]), device)(X).float()
            res[x["id"]] = bank_cache[key]
        timing[f"{c}/nn64"] = time.perf_counter() - s0; s0 = time.perf_counter()
        l1 = {}
        for x in [x for x in cells if x["scorer"]["kind"] == "calm"]:
            sc = x["scorer"]
            d, own = S.calm_scores(*S.calm_bank(sc["arm"], sc["k"]), X, hoi, host_global, device)
            res[x["id"]] = d.float()
            if sc["arm"] == "CALM_L1":
                l1[sc["k"]] = (d, own)
        timing[f"{c}/calm"] = time.perf_counter() - s0; s0 = time.perf_counter()
        dx = S.context(X, device)
        for x in [x for x in cells if x["scorer"]["kind"] == "ntm"]:
            sc = x["scorer"]
            if sc["arm"] == "CTX":
                res[x["id"]] = dx.float()
            else:
                d, own = l1[sc["k"]]
                res[x["id"]] = torch.from_numpy(S.ntm_scores(sc["arm"], d, own, dx, tau)).float()
        timing[f"{c}/ntm"] = time.perf_counter() - s0; s0 = time.perf_counter()
        for x in [x for x in cells if x["scorer"]["kind"] == "p2"]:
            s = S.p2_scores(S.p2_learner(x["scorer"], device), X, device)
            res[x["id"]] = s
            mse[x["id"]] = (s[:C.HOSTS].double().square().mean(1) / 768).numpy()
        timing[f"{c}/p2"] = time.perf_counter() - s0
        ids = [x["id"] for x in cells]
        require(all(bool(torch.isfinite(res[i]).all()) and res[i].shape == (C.HOSTS * 16, 784) for i in ids), "bad scores")
        np.savez(target, ids=np.array(ids), scores=torch.stack([res[i] for i in ids]).numpy(),
                 mse_ids=np.array(list(mse)), mse=np.stack(list(mse.values())) if mse else np.zeros((0, C.HOSTS)))
        print(c, {k: round(v, 1) for k, v in timing.items() if k.startswith(c)}, flush=True)
    timing["peak_vram_mib"] = gpu_peak()
    C.write_json(OUT / "SCORE.json", {"timing": timing, "files": {c: C.sha(OUT / "scores" / f"{c}.npz") for c in C.CATS}})


# ---------------------------------------------------------------- Stage B: proxies

def proxies(device):
    verify_lock(deep=True)
    man = C.read(OUT / "MANIFEST.json")
    cells = {x["id"]: x for x in man["cells"]}
    torch.cuda.reset_peak_memory_stats()
    point, reps = {}, {}
    start = time.perf_counter()
    for ci, c in enumerate(C.CATS):
        syn = torch.load(OUT / "synthetic" / f"{c}.pt", weights_only=False)
        hw, hoi = tuple(syn["hw"]), np.asarray(syn["host_of_image"])
        n = hw[0] * hw[1]
        masks = torch.from_numpy(np.stack([np.unpackbits(m)[:n].reshape(hw) for m in syn["masks"]]).astype(bool))
        z = np.load(OUT / "scores" / f"{c}.npz")
        mse = dict(zip(z["mse_ids"].tolist(), z["mse"]))
        W = M.host_weights(ci, C.HOSTS)
        IW = W[:, hoi]
        zeros = torch.zeros((C.HOSTS, *hw), dtype=torch.bool)
        labels = {f: torch.cat((zeros, masks[j * 100:(j + 1) * 100])).to(device) for j, f in enumerate(C.FAMILIES)}
        index = {f: np.r_[0:C.HOSTS, C.HOSTS + j * 100:C.HOSTS + (j + 1) * 100] for j, f in enumerate(C.FAMILIES)}
        patch_host = np.repeat(np.arange(C.HOSTS), 784)
        for j, cid in enumerate(z["ids"].tolist()):
            s = z["scores"][j]
            p, r = {}, {}
            clean = s[:C.HOSTS].ravel()
            order = np.argsort(clean, kind="stable")
            p["Q4"] = -M.weighted_quantile(clean, np.ones_like(clean), 0.99)
            hs, vs = patch_host[order], clean[order]
            r["Q4"] = np.array([-vs[np.searchsorted(np.cumsum(w[hs]), 0.99 * w[hs].sum(), side="left")] for w in W])
            for f in C.FAMILIES:
                maps = M.native_maps(s[index[f]], hw, device)
                p[f] = M.exact_ap(maps, labels[f])
                lo, hi = bootstrap_bounds(list(maps), list(labels[f]), IW[:, index[f]])
                r[f] = (lo + hi) / 2
                del maps
            if cid in mse:
                p["Q5"] = -float(mse[cid].mean())
                r["Q5"] = -(W @ mse[cid]) / W.sum(1)
            if cells[cid]["family"] == "CALM":
                p["Q6"] = -cells[cid]["deficit"]
                r["Q6"] = np.full(C.REPS, p["Q6"])
            point[cid], reps[cid] = p, r
        print(c, f"{time.perf_counter() - start:.0f}s", flush=True)
    ids = sorted(point)
    P = np.full((len(ids), len(C.PROXIES)), np.nan)
    Rp = np.full((len(ids), len(C.PROXIES), C.REPS), np.nan, dtype=np.float64)
    for i, cid in enumerate(ids):
        for k, q in enumerate(C.PROXIES):
            if q in point[cid]:
                P[i, k], Rp[i, k] = point[cid][q], reps[cid][q]
    target = OUT / "PROXIES.npz"
    require(not target.exists(), "PROXIES exists")
    np.savez(target, ids=np.array(ids), proxies=np.array(C.PROXIES), point=P, reps=Rp)
    C.write_json(OUT / "PROXIES.json", {"seconds": time.perf_counter() - start, "peak_vram_mib": gpu_peak(),
                                        "sha256": C.sha(target), "cells": len(ids)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("manifest", "pilot", "lock", "generate", "score", "proxies", "analyze", "audit"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage == "manifest":
        manifest()
    elif a.stage == "pilot":
        pilot(a.device)
    elif a.stage == "lock":
        lock()
    elif a.stage == "generate":
        generate(a.device)
    elif a.stage == "score":
        score(a.device)
    elif a.stage == "proxies":
        proxies(a.device)
    elif a.stage == "analyze":
        from exps.ptag.analysis import analyze
        verify_lock(deep=True)
        analyze()
    else:
        from exps.ptag.analysis import audit
        verify_lock(deep=True)
        audit(a.device)


if __name__ == "__main__":
    main()
