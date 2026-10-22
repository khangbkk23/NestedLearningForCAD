"""Arm C: byte-matched spatial reservoir control.

Uniform reservoir sampling (Algorithm R) per spatial bin over the causal writer
descriptor stream.  The RNG is stateless and counter-based: the draw for item i
of bin b is ``splitmix64(seed + (b << 40) + i)``.  The persistent control state
therefore consists of the reservoir slots, the per-bin item count and the
per-bin seen count; no mutable generator state is needed, and reloading the
state cannot change future draws.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from . import contract as C

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def draws_for(items: np.ndarray, bin_index: int, seed: int = C.RESERVOIR_SEED) -> np.ndarray:
    """Uniform [0,1) draw for every 1-based item index in ``items``."""
    keys = (np.uint64(seed) + (np.uint64(bin_index) << np.uint64(40)) + items.astype(np.uint64))
    z = splitmix64_array(keys)
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def build_reservoir(writer_bins: Sequence[torch.Tensor], t: int,
                    capacity: int = C.RESERVOIR_CAPACITY,
                    seed: int = C.RESERVOIR_SEED) -> Dict[str, np.ndarray]:
    """Build the per-bin reservoir from the first ``t`` writer events."""
    res = np.zeros((C.BINS, capacity, C.DIM), dtype=np.float32)
    count = np.zeros((C.BINS,), dtype=np.int64)
    seen = np.zeros((C.BINS,), dtype=np.int64)
    for b in range(C.BINS):
        D = writer_bins[b]
        n = int(D.shape[0])
        if n == 0:
            raise RuntimeError(f"bin {b}: empty writer stream")
        if n != t * C.TOKENS_PER_BIN:
            raise ValueError(
                f"bin {b}: expected {t * C.TOKENS_PER_BIN} prefix descriptors for t={t}, got {n}")
        seen[b] = n
        initial = min(capacity, n)
        res[b, :initial] = D[:initial].numpy().astype(np.float32)
        count[b] = initial
        if n > capacity:
            items = np.arange(capacity + 1, n + 1, dtype=np.uint64)
            u = draws_for(items, b, seed)
            j = np.floor(u * items.astype(np.float64)).astype(np.int64)
            for local, item in enumerate(items):
                slot = int(j[local])
                if slot < capacity:
                    res[b, slot] = D[int(item) - 1].numpy().astype(np.float32)
            count[b] = capacity
    return {"res": res, "count": count, "seen": seen}


def reservoir_arrays(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    res = np.asarray(state["res"], dtype=np.float32)
    count = np.asarray(state["count"], dtype=np.int64)
    seen = np.asarray(state["seen"], dtype=np.int64)
    if res.shape != (C.BINS, C.RESERVOIR_CAPACITY, C.DIM):
        raise ValueError(f"reservoir shape mismatch: {res.shape}")
    if count.shape != (C.BINS,) or seen.shape != (C.BINS,):
        raise ValueError("reservoir metadata shape mismatch")
    if np.any(count < 1) or np.any(count > C.RESERVOIR_CAPACITY):
        raise ValueError("reservoir count out of range")
    return {"res": res, "count": count, "seen": seen}
