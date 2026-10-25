# exps/ptag/scoring.py
"""Frozen-detector scorers for PTAG-01. Detector states are loaded read-only; nothing is updated."""
from __future__ import annotations

import numpy as np
import torch

from exps.calm.decompose import build_cells
from exps.calm.load import CAT_INDEX, load_state
from exps.calm02.score import Scorer, decode
from exps.ntm.stage0 import combine, context_distance, context_mask
from exps.ptag.common import CALM, NTM, ROOT, read
from exps.task_p.p0_normal_support import distances, require


# ---------------------------------------------------------------- raw FP64 1-NN (P1, CALM-00A)

def nn64_bank(spec):
    if spec[0] == "p1":
        _, arm, t = spec
        return load_state(t, arm)["bank"]
    _, arm, acq, T, name, cat = spec
    return build_cells(load_state(acq, arm), load_state(T, arm), CAT_INDEX[cat])[name]["bank"]


class NN64:
    """`p0_normal_support.distances` (FP64, cancellation repair), minimum over the bank."""

    def __init__(self, bank, device, chunk=8):
        self.B, self.device, self.chunk = bank.double().to(device), device, chunk

    @torch.no_grad()
    def __call__(self, X):
        out = []
        for i in range(0, len(X), self.chunk):
            q = X[i:i + self.chunk].reshape(-1, X.shape[-1]).to(self.device)
            out.append(distances(q, self.B).min(1).values.reshape(-1, 784).cpu())
        return torch.cat(out)                     # FP64 [N, 784]


# ---------------------------------------------------------------- CALM-02A banks, leave-host-out

def calm_bank(arm, k):
    ck = torch.load(CALM / f"runs/mvtec/{arm}_B1/task_{k:02d}.pt", map_location="cpu", weights_only=False)
    occ = ck["occupied"]
    return decode(ck["state"])[occ], ck["owner"][occ], ck["ids"][occ]


@torch.no_grad()
def calm_scores(rows, owners, row_ids, X, host_of_image, host_global, device):
    """Exact CALM-02 Scorer; images derived from host h never see bank rows sourced from host h."""
    src = (row_ids // 784).numpy()
    d_out = torch.empty(len(X), 784, dtype=torch.float64)
    own_out = torch.empty(len(X), 784, dtype=torch.int64)
    for h, g in enumerate(host_global):
        keep = torch.from_numpy(src != g)
        sc = Scorer(rows[keep], device)
        own = owners[keep]
        for i in np.flatnonzero(host_of_image == h):
            d, idx = sc.patch(X[i])
            d_out[i], own_out[i] = d, own[idx].long()
    return d_out, own_out


def ntm_tau():
    tau = read(NTM / "prepared.json")["tau"]
    return np.array([tau[str(k)]["tau"] for k in range(len(tau))])


@torch.no_grad()
def context(X, device):
    allowed = context_mask().to(device)
    return torch.stack([context_distance(x.to(device), allowed).cpu() for x in X])


def ntm_scores(arm, d_m, own, d_x, tau):
    return combine(arm, d_m.numpy(), d_x.numpy(), tau[own.numpy()])


# ---------------------------------------------------------------- P2 family

def p2_learner(spec, device):
    st = torch.load(ROOT / spec["state_file"], map_location="cpu", weights_only=False)["states"][spec["arm"]]
    d, arm, seed = spec["dir"], spec["arm"], spec["seed"]
    if d == "p2":
        from exps.task_p.p2.model import Learner
        learner = Learner(arm, seed, device)
    elif d == "p2d":
        from exps.task_p.p2d.model import Learner2
        learner = Learner2(arm, seed, device)
    elif d == "p2e":
        from exps.task_p.p2e.data import model_arguments
        from exps.task_p.p2e.model import Learner2
        learner = Learner2(arm, seed, device, **model_arguments(read(ROOT / "results/hope_cad/task_p/p2e/protocol_resolved.json")))
    else:
        from exps.task_p.p2f.data import model_arguments
        from exps.task_p.p2f.model import LearnerF
        learner = LearnerF(arm, seed, device, **model_arguments(read(ROOT / "results/hope_cad/task_p/p2f/protocol_resolved.json")))
    learner.restore(st)
    return learner


@torch.no_grad()
def p2_scores(learner, X, device, chunk=32):
    out = [learner.scores(X[i:i + chunk].to(device)).float().cpu() for i in range(0, len(X), chunk)]
    s = torch.cat(out)
    require(s.shape == (len(X), 784) and bool(torch.isfinite(s).all()), "bad P2 scores")
    return s
