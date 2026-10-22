"""Deterministic support selection for CSG-01.

Arm B/C: exact farthest-first selection (FPS) over the causal writer prefix,
51 descriptors per bin plus one extra in bin 0 (all bins have equal candidate
counts, so the largest-count tie-break resolves to the lowest bin index).

Arm D: deterministic hash-random selection with the same quotas.

Arm E: the same FPS policy with the byte-matched smaller quota vector.

No N x N matrix is constructed.  Each FPS step needs one matrix-vector product
against the current selected point and an O(N) min-distance update.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from . import contract as C1

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def random_priorities(seed: int, global_ids: np.ndarray) -> np.ndarray:
    x = (np.uint64(seed) + global_ids.astype(np.uint64) *
         np.uint64(0x9E3779B97F4A7C15)) & _MASK
    return splitmix64_array(x)


def _dist_to_vector(D: torch.Tensor, v: torch.Tensor, D2: torch.Tensor) -> torch.Tensor:
    return (D2 - 2.0 * (D @ v) + float(v @ v)).clamp_min_(0.0)


def select_fps(D: torch.Tensor, k: int) -> np.ndarray:
    """Exact deterministic farthest-first selection of k local indices."""
    n = D.shape[0]
    if k > n:
        raise ValueError(f"cannot select {k} from {n} candidates")
    if k == n:
        return np.arange(n, dtype=np.int64)
    D = D.to(torch.float64)
    D2 = (D * D).sum(dim=1)
    mean = D.mean(dim=0)
    first = int(_dist_to_vector(D, mean, D2).argmin().item())
    selected: List[int] = [first]
    min_d2 = _dist_to_vector(D, D[first], D2)
    min_d2[first] = -1.0
    selected_mask = torch.zeros(n, dtype=torch.bool)
    selected_mask[first] = True
    for _ in range(1, k):
        j = int(min_d2.argmax().item())
        if float(min_d2[j].item()) < 0.0:
            remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
            if remaining.numel() == 0:
                break
            j = int(remaining[0].item())
        selected.append(j)
        selected_mask[j] = True
        d2j = _dist_to_vector(D, D[j], D2)
        min_d2 = torch.minimum(min_d2, d2j)
        min_d2[selected_mask] = -1.0
    if len(selected) != k:
        raise RuntimeError(f"FPS produced {len(selected)} of {k} selections")
    return np.asarray(selected, dtype=np.int64)


def select_random(n: int, k: int, seed: int, global_ids: np.ndarray) -> np.ndarray:
    """Deterministic hash-random selection of k local indices."""
    if k > n:
        raise ValueError(f"cannot select {k} from {n} candidates")
    if k == n:
        return np.arange(n, dtype=np.int64)
    if global_ids.size != n:
        raise ValueError("global id array length mismatch")
    if len(set(global_ids.tolist())) != n:
        raise ValueError("global ids must be unique within a bin")
    priority = random_priorities(seed, global_ids)
    order = np.lexsort((global_ids, priority))
    return order[:k].astype(np.int64)


def _local_to_global(bin_index: int, local_indices: np.ndarray,
                     positions: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    local = np.asarray(local_indices, dtype=np.int64)
    per_event = C1.TOKENS_PER_BIN
    events = local // per_event
    pos_idx = local % per_event
    patches = np.asarray(positions[bin_index], dtype=np.int64)[pos_idx]
    gids = events * C1.PATCHES + patches
    return {"event": events, "patch": patches, "global_id": gids}


def build_selection(writer_bins: Sequence[torch.Tensor], t: int,
                    quotas: np.ndarray, policy: str,
                    seed: int = C1.RANDOM_SEED,
                    updates: Sequence[Dict[str, Any]] | None = None,
                    positions: Sequence[np.ndarray] | None = None) -> Dict[str, Any]:
    """Select descriptors per bin and return local/global identities and provenance."""
    if policy not in {"fps", "random"}:
        raise ValueError(policy)
    if int(quotas.sum()) > sum(int(x.shape[0]) for x in writer_bins):
        raise ValueError("quota exceeds candidate count")
    started = time.perf_counter()
    selected: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    per_bin_seconds: List[float] = []
    for b in range(C1.BINS):
        D = writer_bins[b].to(torch.float64)
        k = int(quotas[b])
        if k > D.shape[0]:
            raise ValueError(f"bin {b}: quota {k} > candidates {D.shape[0]}")
        t0 = time.perf_counter()
        if policy == "fps":
            local = select_fps(D, k)
        else:
            if positions is None:
                raise ValueError("positions required for random selection")
            events = np.arange(D.shape[0], dtype=np.int64) // C1.TOKENS_PER_BIN
            pos_idx = np.arange(D.shape[0], dtype=np.int64) % C1.TOKENS_PER_BIN
            patches = np.asarray(positions[b], dtype=np.int64)[pos_idx]
            gids = events * C1.PATCHES + patches
            local = select_random(D.shape[0], k, seed, gids)
        per_bin_seconds.append(time.perf_counter() - t0)
        selected.append(local)
        mapped = _local_to_global(b, local, positions)
        for j in range(k):
            event = int(mapped["event"][j])
            row: Dict[str, Any] = {
                "bin": int(b),
                "local_index": int(local[j]),
                "global_id": int(mapped["global_id"][j]),
                "event": event,
                "patch": int(mapped["patch"][j]),
                "category": None,
                "relative_path": None,
            }
            if updates is not None:
                row["category"] = updates[event]["category"]
                row["relative_path"] = updates[event]["relative_path"]
            rows.append(row)
    return {
        "policy": policy,
        "seed": int(seed),
        "quotas": quotas.astype(np.int32),
        "selected_local": selected,
        "provenance": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "per_bin_seconds": per_bin_seconds,
        "n_selected": int(sum(x.size for x in selected)),
        "all_events_causal": bool(all(r["event"] < t for r in rows)),
    }


def selected_global_ids(selection: Dict[str, Any]) -> np.ndarray:
    return np.asarray([r["global_id"] for r in selection["provenance"]], dtype=np.int64)


def take_selected(writer_bins: Sequence[torch.Tensor],
                  selection: Dict[str, Any]) -> List[torch.Tensor]:
    out = []
    for b in range(C1.BINS):
        local = torch.as_tensor(selection["selected_local"][b], dtype=torch.long)
        out.append(writer_bins[b][local].to(torch.float32).contiguous())
    return out


def pack_support(support: Sequence[torch.Tensor]) -> Dict[str, np.ndarray]:
    counts = np.asarray([x.shape[0] for x in support], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    desc = torch.cat([x.to(torch.float32) for x in support], dim=0).numpy().astype(np.float32)
    return {"counts": counts, "offsets": offsets, "desc": desc}


def split_support(desc: np.ndarray, counts: np.ndarray,
                  offsets: np.ndarray) -> List[torch.Tensor]:
    return [torch.as_tensor(desc[offsets[b]:offsets[b] + counts[b]], dtype=torch.float32)
            for b in range(C1.BINS)]
