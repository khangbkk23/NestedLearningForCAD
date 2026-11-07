# exps/cadfinal/memory.py
"""Bounded nearest-neighbour support memories on the frozen ViT-B/8 (layer 9) CALM-02 feature caches.

Arms (unchanged implementations, imported):
  CALM_L1       exps.calm01 ``L1_FIXED_U8`` (fixed CALM fast level, eta = 0; shortlist 16, floor 1)
  CADIC_COMPAT  exps.calm01 ``A_CADIC_FP32`` (bitwise-equal to models/cadic_patch_coreset_v1.py; CADIC
                Eq. 1-6 as implemented in this repository; author code unavailable -> project-adapted)
Scorer (identical for both, exps.calm02.score): exact FP64 1-NN Euclidean patch distance; image score CADIC
Eq. (9) with b = 9 (primary for Image AUROC) and max patch (secondary).
The stream follows exps.calm02.run.train exactly (sorted paths, one image per update), restricted to the
benchmark's training identities. States are saved after every task; scoring needs only acquisition (k = j)
and final (k = T - 1) states.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.cadfinal import common as K
from exps.calm01 import arms as A
from exps.calm02.score import Scorer, decode
from exps.task_p.p0_normal_support import require

FEAT = K.ROOT / "results/hope_cad/calm02/features"


def run_dir(bench, arm, budget):
    return K.OUT / "runs" / bench / f"{arm}_{budget}"


def features(bench, task, split):
    obj = torch.load(FEAT / bench / f"{task}_{split}.pt", map_location="cpu", weights_only=False, mmap=True)
    return obj["features"], obj["rows"]


def train(bench, arm, budget, device="cuda"):
    ids = K.benchmark_identities(bench)
    n_cat, name = len(ids["tasks"]), K.MEMORY_ARMS[arm]
    m = A.build(name, K.BUDGETS[budget], n_cat, cache_device=device)
    out = run_dir(bench, arm, budget)
    out.mkdir(parents=True, exist_ok=False)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log, serial = [], 0
    for k, t in enumerate(ids["tasks"]):
        feats, rows = features(bench, t["task"], "train")
        index = {r["relative_path"]: i for i, r in enumerate(rows)}
        order = [index[r["relative_path"]] for r in t["train"]]          # sorted path order, matched identities
        m.begin_category(k)
        start, stats = time.perf_counter(), {"inserted": 0, "replaced": 0, "rejected": 0}
        for i in order:
            s = m.update(feats[i].clone(), list(range(serial, serial + 784))); serial += 784
            for key in stats:
                stats[key] += s.get(key, 0)
        m.end_category()
        seconds = time.perf_counter() - start
        path = out / f"task_{k:02d}.pt"
        torch.save({"state": A.persistent_state(m, name, n_cat), "occupied": m.bank.occupied.clone(),
                    "owner": m.bank.owner.clone(), "ids": m.bank.ids.clone()}, path)
        log.append({"task": t["task"], "images": len(order), "seconds": seconds, **stats,
                    "rows_by_task": {ids["tasks"][j]["task"]: int((m.bank.owner == j).sum()) for j in range(k + 1)},
                    "state_sha256": K.sha(path)})
        print(bench, arm, budget, t["task"], f"{seconds:.0f}s", stats, flush=True)
        del feats
    K.write_json(out / "TRAIN.json", {
        "arm": arm, "implementation": name, "budget": budget, "budget_bytes": K.BUDGETS[budget],
        "capacity_rows": m.capacity, "ledger": A.ledger(name, m.capacity, n_cat), "cache_device": device,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "torch": torch.__version__, "tasks": log, "finished_utc": K.now()})


@torch.no_grad()
def score(bench, arm, budget, device="cuda"):
    """Label-free: patch distances [n,784] (FP32), CADIC Eq. 9 and max-patch image scores, for every test image of
    task j at its acquisition state (k = j) and at the final state (k = T-1)."""
    ids = K.benchmark_identities(bench)
    T = len(ids["tasks"])
    out = run_dir(bench, arm, budget)
    require((out / "TRAIN.json").exists(), "train first")
    tr = K.read(out / "TRAIN.json")
    states = {}
    record = {}
    for j, t in enumerate(ids["tasks"]):
        feats, rows = features(bench, t["task"], "test")
        require([r["relative_path"] for r in rows] == [r["relative_path"] for r in t["test"]], "test row order drift")
        for kind, k in (("acq", j), ("fin", T - 1)):
            target = out / f"scores_{kind}_{j:02d}.npz"
            require(not target.exists(), f"exists: {target}")
            if k not in states:
                ck_path = out / f"task_{k:02d}.pt"
                require(K.sha(ck_path) == tr["tasks"][k]["state_sha256"], "state changed after training")
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                states = {k: Scorer(decode(ck["state"])[ck["occupied"]], device)}       # keep one bank in memory
            sc = states[k]
            pd, ic = np.empty((len(rows), 784), dtype=np.float32), np.empty(len(rows))
            t0 = time.perf_counter()
            for i in range(len(rows)):
                d, idx = sc.patch(feats[i])
                pd[i], ic[i] = d.float().numpy(), sc.image_cadic(feats[i], d, idx)
            np.savez(target, patch=pd, image_cadic=ic, image_max=pd.max(1), paths=np.array([r["relative_path"] for r in rows]))
            record[f"{kind}_{j:02d}"] = {"sha256": K.sha(target), "state": k, "seconds": time.perf_counter() - t0}
        print(bench, arm, budget, "scored", t["task"], flush=True)
        del feats
    K.write_json(out / "SCORES.json", {"arm": arm, "budget": budget, "files": record, "finished_utc": K.now()})
