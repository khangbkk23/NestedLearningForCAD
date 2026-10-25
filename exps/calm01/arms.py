# exps/calm01/arms.py
"""CALM-01 arm registry and persistent-byte ledger (equal bytes, not equal rows).

Charged per row: stored vector (FP32 3072 B, or UINT8 768 B + FP32 xmin/step 8 B); owner 1 B for
every arm whose decisions read it; CALM mass and cost 8 B each (FP64). Charged per category: scale
(scaled/quota) or scale, price, mass, acquisition reference (CALM). Not charged, identically for every
arm: serial ids (provenance only) and the nearest-other graph (derived, recomputable from the bank).
The frozen backbone is shared and reported separately.
"""
from __future__ import annotations

import torch

from exps.task_p.p0_normal_support import require
from exps.calm01.calm import CALMMemory
from exps.calm01.closest_pair import ClosestPairMemory

# Placeholder controller step; Phase 2 selects it on VisA development data only.
ETA_DEFAULT = 2.0

ARMS = {
    "A_CADIC_FP32": {"kind": "closest_pair", "mode": "cadic", "storage": "fp32"},
    "A_CADIC_U8": {"kind": "closest_pair", "mode": "cadic", "storage": "u8"},
    "B_QUOTA_U8": {"kind": "closest_pair", "mode": "quota", "storage": "u8"},
    "C_SCALED_U8": {"kind": "closest_pair", "mode": "scaled", "storage": "u8"},
    "L1_FIXED_U8": {"kind": "calm", "storage": "u8", "eta": 0.0, "clock": "boundary"},
    "L2_NESTED_U8": {"kind": "calm", "storage": "u8", "eta": ETA_DEFAULT, "clock": "boundary"},
    "L2S_SAMECLOCK_U8": {"kind": "calm", "storage": "u8", "eta": ETA_DEFAULT, "clock": "image"},
}

# Reference budgets: CADIC's 2,500 and 10,000 FP32 patch coresets (paper Table VII).
BUDGETS = {"cadic2500": 2_500 * 768 * 4, "cadic10000": 10_000 * 768 * 4}


def row_bytes(arm, dim=768):
    spec = ARMS[arm]
    vec = dim * 4 if spec["storage"] == "fp32" else dim + 8
    owner = 0 if spec.get("mode") == "cadic" else 1
    return vec + owner + (16 if spec["kind"] == "calm" else 0)


def category_bytes(arm):
    spec = ARMS[arm]
    return 32 if spec["kind"] == "calm" else 0 if spec.get("mode") == "cadic" else 8


def capacity_for_budget(arm, budget, n_categories, dim=768):
    k = (int(budget) - category_bytes(arm) * int(n_categories)) // row_bytes(arm, dim)
    require(k >= max(2, n_categories), "budget too small for the arm")
    return int(k)


def ledger(arm, capacity, n_categories, dim=768):
    total = row_bytes(arm, dim) * capacity + category_bytes(arm) * n_categories
    return {"arm": arm, "capacity_rows": capacity, "row_bytes": row_bytes(arm, dim),
            "category_bytes": category_bytes(arm), "n_categories": n_categories,
            "charged_persistent_bytes": total,
            "uncharged": ["serial ids (provenance)", "nearest-other graph (derived)", "frozen backbone"]}


def persistent_state(memory, arm, n_categories):
    """The charged detector state in its charged dtypes (what a deployment would serialise)."""
    spec, b = ARMS[arm], memory.bank
    out = ({"x": b.x.clone()} if spec["storage"] == "fp32" else
           {"q": b.q.clone(), "xmin": b.xmin.clone(), "step": b.step.clone()})
    if spec.get("mode") != "cadic":
        out["owner"] = b.owner.to(torch.int8)
    per_cat = 4 if spec["kind"] == "calm" else 0 if spec.get("mode") == "cadic" else 1
    if per_cat:
        table = torch.zeros(n_categories, per_cat, dtype=torch.float64)
        for k in memory.categories:
            row = [memory.scales.s.get(k, 0.0)]
            if spec["kind"] == "calm":
                row += [memory.price[k], memory.mass[k], memory.j_acq.get(k, float("nan"))]
            table[k] = torch.tensor(row, dtype=torch.float64)
        out["category"] = table
    if spec["kind"] == "calm":
        out["mu"], out["kappa"] = memory.mu.clone(), memory.kappa.clone()
    return out


def state_bytes(state):
    return sum(v.numel() * v.element_size() for v in state.values())


def build(arm, budget, n_categories, dim=768, n_ref=None, cache_device="cpu", **overrides):
    """Instantiate an arm at equal persistent bytes. ``n_ref``: images per category for L2S."""
    spec = dict(ARMS[arm], **overrides)
    k = capacity_for_budget(arm, budget, n_categories, dim)
    if spec["kind"] == "closest_pair":
        return ClosestPairMemory(k, mode=spec["mode"], storage=spec["storage"], dim=dim,
                                 distance=spec.get("distance", "p0_fp64"), cache_device=cache_device)
    eta_image = 0.0
    if spec["clock"] == "image":
        require(n_ref is not None and n_ref > 0, "same-clock arm needs n_ref (protocol constant)")
        eta_image, spec["eta"] = spec["eta"] / float(n_ref), 0.0
    return CALMMemory(k, storage=spec["storage"], eta=spec["eta"], clock=spec["clock"],
                      eta_image=eta_image, dim=dim, shortlist=spec.get("shortlist", 16),
                      floor=spec.get("floor", 1), distance=spec.get("distance", "p0_fp64"),
                      cache_device=cache_device)
