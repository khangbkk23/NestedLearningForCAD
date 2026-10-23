"""Frozen inputs and G0 configuration for NCSL-01 Stage 0."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exps.csg00.features import load_feature_cache, verify_cache_rows
from exps.csg00 import contract as C

ROOT = C.ROOT
P0C = ROOT / "results/hope_cad/task_p/p0c/protocol_resolved.json"
CHECKPOINT_SHA256 = C.CHECKPOINT_SHA256
BOUNDARIES = (200, 300, 350)
SEED = 20261022
TAU = 1e-3
LR = 0.05
EPOCHS = 4
RESELECT_EVERY = 2
PROBE_BATCH = 2048


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(verify: bool = True) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    d = json.loads(P0C.read_text())
    if d["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint lineage mismatch")
    cache = load_feature_cache(verify_hashes=verify)
    checks = verify_cache_rows(cache, d["updates"], d["probes"])
    return d, cache, checks


def split_events(updates: list[dict], t: int) -> tuple[list[int], list[int]]:
    # Image-level, causal, deterministic 80/20 split.  No patch-level split.
    fit, held = [], []
    for i, row in enumerate(updates[:t]):
        # SplitMix-like stable hash; it is independent of feature values.
        x = (int(row["event"]) * 0x9E3779B97F4A7C15 + SEED) & ((1 << 64) - 1)
        x ^= x >> 30; x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        x ^= x >> 27; x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
        x ^= x >> 31
        (held if x % 5 == 0 else fit).append(i)
    if not fit or not held:
        raise RuntimeError("invalid image-level split")
    return fit, held


def bin_positions() -> list[np.ndarray]:
    out = []
    for br in range(7):
        for bc in range(7):
            out.append(np.asarray([(br * 4 + r) * 28 + bc * 4 + c
                                   for r in range(4) for c in range(4)], dtype=np.int64))
    return out


def stack_events(cache, updates, indices: list[int]) -> torch.Tensor:
    return torch.stack([cache.row(updates[i]["category"], int(updates[i]["cache_index"]))
                        for i in indices]).float()


def config_hash() -> str:
    payload = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

