# exps/rh01/common.py
"""Paths, identities, the generalized Replay-24 buffer and the memory-mapped feature pool for RH-01."""
from __future__ import annotations

import bisect
import os
from pathlib import Path

import torch

from exps.dinomaly import common as D
from exps.dinomaly.common import require, sha

ROOT = D.ROOT
OUT = ROOT / "results/hope_cad/rh01"
DRF = D.OUT
CACHE = Path.home() / "hope_cad_cache/rh01/features"
STREAM = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
          "tile", "toothbrush", "transistor", "wood", "zipper")
EVAL = D.CATS                                        # bottle, carpet, hazelnut: the consumed P1 development cohort
CAPACITY = 24
N_TRAIN = 3099


def identities():
    """Training: P1 identities for the evaluated categories, all train/good otherwise. Held-out/evaluation: DRF-01."""
    base = D.identities()
    sealed = set(D.p1_protocol()["sealed_identity_paths"])
    train = {}
    for c in STREAM:
        if c in EVAL:
            train[c] = [{k: r[k] for k in ("relative_path", "sha256", "bytes", "category")} for r in base["train"][c]]
        else:
            d = D.DATA / c / "train/good"
            train[c] = [{"relative_path": f"{c}/train/good/{f}", "sha256": sha(d / f), "bytes": (d / f).stat().st_size,
                         "category": c} for f in sorted(os.listdir(d))]
    paths = {r["relative_path"] for rows in train.values() for r in rows}
    held = {r["relative_path"] for rows in base["heldout"].values() for r in rows}
    ev = {r["relative_path"] for rows in base["evaluation"].values() for r in rows}
    require(all("/train/good/" in p for p in paths), "training identity outside train/good")
    require(not (paths & held) and not (paths & ev) and not (paths & sealed), "identity overlap or sealed identity")
    require(sum(len(v) for v in train.values()) == N_TRAIN, "training identity count changed")
    return {"train": train, "heldout": base["heldout"], "evaluation": base["evaluation"], "sealed_count": len(sealed)}


# --- Replay-24 ---------------------------------------------------------------------------------------


def buffer_plan(seed, n_train):
    """Per-class member permutations (stream order, seed + 9000) and the remainder priority (seed + 9001)."""
    gen = torch.Generator().manual_seed(seed + 9000)
    perms = {c: torch.randperm(n_train[c], generator=gen).tolist() for c in STREAM}
    order = torch.randperm(len(STREAM), generator=torch.Generator().manual_seed(seed + 9001)).tolist()
    return perms, [STREAM[i] for i in order]


def allocation(done, priority):
    """All 24 slots used: 24 // k each, remainder to the first completed classes in priority order."""
    base, rem = divmod(CAPACITY, len(done))
    extra = set([c for c in priority if c in done][:rem])
    return {c: base + (c in extra) for c in done}


def buffer_index(seed, n_train, done):
    perms, priority = buffer_plan(seed, n_train)
    alloc = allocation(done, priority)
    return {c: perms[c][:alloc[c]] for c in done}


# --- features ----------------------------------------------------------------------------------------


_VERIFIED = set()


def load_features(c, mmap=True):
    """Cached FP32 features [n, 2, 789, 768] from the ext4 cache, SHA-verified against CACHE.json once per process."""
    path = CACHE / f"train_{c}.pt"
    if c not in _VERIFIED:
        require(sha(path) == D.read(OUT / "CACHE.json")[c]["sha256"], f"feature cache changed: {path}")
        _VERIFIED.add(c)
    return torch.load(path, weights_only=True, mmap=mmap)["features"]


class Pool:
    """Indexable view over per-category feature tensors in stream order (no concatenation in RAM)."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.starts = [0]
        for t in self.tensors:
            self.starts.append(self.starts[-1] + len(t))

    def __len__(self):
        return self.starts[-1]

    def __getitem__(self, idx):
        rows = []
        for i in idx:
            k = bisect.bisect_right(self.starts, i) - 1
            rows.append(self.tensors[k][i - self.starts[k]])
        return torch.stack(rows)
