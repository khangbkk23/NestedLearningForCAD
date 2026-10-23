"""CALM-00A: 2x2 cell construction and frozen-scorer evaluation. No labels are used here."""
from __future__ import annotations

import torch

from exps.task_p.p0_normal_support import distances

CELLS = ("00", "01", "10", "11", "oracle")


def _subset(bank, ids, owners, keep: torch.Tensor) -> dict:
    return {"bank": bank[keep].contiguous(), "ids": ids[keep].contiguous(),
            "owners": owners[keep].contiguous()}


def build_cells(acq: dict, final: dict, query_cat: int) -> dict[str, dict]:
    """Five candidate sets defined by set operations on descriptor serial ids.

    00 = acquisition bank; 10 = acquisition rows surviving at final;
    01 = acquisition bank plus every row added afterwards; 11 = final bank;
    oracle = final rows owned by the query category (perfect-router bound).
    """
    a_bank, a_ids, a_own = acq["bank"], acq["ids"], acq["owners"]
    f_bank, f_ids, f_own = final["bank"], final["ids"], final["owners"]

    in_final = torch.isin(a_ids, f_ids)
    in_acq = torch.isin(f_ids, a_ids)

    cells = {
        "00": {"bank": a_bank, "ids": a_ids, "owners": a_own},
        "10": _subset(a_bank, a_ids, a_own, in_final),
        "01": {
            "bank": torch.cat((a_bank, f_bank[~in_acq]), 0),
            "ids": torch.cat((a_ids, f_ids[~in_acq]), 0),
            "owners": torch.cat((a_own, f_own[~in_acq]), 0),
        },
        "11": {"bank": f_bank, "ids": f_ids, "owners": f_own},
        "oracle": _subset(f_bank, f_ids, f_own, f_own == query_cat),
    }
    for name, cell in cells.items():
        if cell["bank"].shape[0] == 0:
            raise RuntimeError(f"empty cell {name}")
        if cell["bank"].shape[0] != cell["ids"].shape[0] != cell["owners"].shape[0]:
            raise RuntimeError(f"cell {name} length mismatch")
    return cells


@torch.no_grad()
def score_cell(cell: dict, feature: torch.Tensor, chunk: int = 64) -> dict:
    """Frozen readout: FP64 raw Euclidean 1-NN with cancellation repair.

    Returns the two smallest distances, the owner of the nearest row and its column index.
    """
    bank = cell["bank"]
    n = bank.shape[0]
    d1 = torch.empty(784, dtype=torch.float64)
    d2 = torch.full((784,), float("inf"), dtype=torch.float64)
    idx1 = torch.zeros(784, dtype=torch.int64)
    for start in range(0, 784, chunk):
        q = feature[start:start + chunk]
        D = distances(q, bank)
        if n >= 2:
            v, i = torch.topk(D, 2, dim=1, largest=False)
            d1[start:start + len(q)] = v[:, 0]
            d2[start:start + len(q)] = v[:, 1]
            idx1[start:start + len(q)] = i[:, 0]
        else:
            d1[start:start + len(q)] = D[:, 0]
            idx1[start:start + len(q)] = 0
    if not bool(torch.isfinite(d1).all()):
        raise RuntimeError("nonfinite nearest distance")
    owner1 = cell["owners"][idx1].to(torch.int8)
    return {"d1": d1, "d2": d2, "idx1": idx1, "owner1": owner1}


def score_group(cells: dict[str, dict], features: list[torch.Tensor]) -> dict[str, dict]:
    out = {name: {"d1": [], "d2": [], "owner1": []} for name in cells}
    for feat in features:
        for name, cell in cells.items():
            s = score_cell(cell, feat)
            out[name]["d1"].append(s["d1"])
            out[name]["d2"].append(s["d2"])
            out[name]["owner1"].append(s["owner1"])
    return {name: {k: torch.stack(v, 0).numpy() for k, v in rec.items()} for name, rec in out.items()}
