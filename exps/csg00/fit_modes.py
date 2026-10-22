"""Deterministic two-mode bounded-support fitting (Arm B).

Registered fitting rule (G0):

* per spatial bin, all descriptors of the causal writer prefix (event order, then
  row-major position order);
* K=2 mode clustering by deterministic k-means++ initialisation (4 restarts) and
  full-batch Lloyd iterations, selecting the restart with the lowest inertia
  (ties -> lowest restart index); the "mini-batch" limit with batch = all bin
  descriptors is used because G2 is an OFFLINE STATIC feasibility study;
* per cluster: population covariance (1/n), top q=4 eigenvectors, eigenvalues
  ``lambda_j``; R_j = 2 sqrt(lambda_j + eps), rho = sqrt(trace - sum lambda);
* modes with fewer than N_MIN_MODE assigned descriptors are dropped; a bin with
  no ready mode is a pre-registered BLOCKED condition.

No anomaly value or outcome participates in fitting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from . import contract as C


def _sqdist(D: torch.Tensor, Cc: torch.Tensor, D2: torch.Tensor | None = None) -> torch.Tensor:
    if D2 is None:
        D2 = (D * D).sum(dim=1)
    c2 = (Cc * Cc).sum(dim=1)
    return (D2.unsqueeze(1) - 2.0 * (D @ Cc.transpose(0, 1)) + c2.unsqueeze(0)).clamp_min_(0.0)


def kmeanspp_init(D: torch.Tensor, k: int, rng: np.random.Generator) -> torch.Tensor:
    n = D.shape[0]
    first = int(rng.integers(0, n))
    centers = [D[first]]
    d2 = _sqdist(D, D[first].unsqueeze(0)).squeeze(1)
    for _ in range(1, k):
        total = float(d2.sum().item())
        if total <= 0.0:
            pick = int(rng.integers(0, n))
        else:
            probs = (d2 / total).detach().cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()
            pick = int(rng.choice(n, p=probs))
        centers.append(D[pick])
        d2 = torch.minimum(d2, _sqdist(D, D[pick].unsqueeze(0)).squeeze(1))
    return torch.stack(centers, dim=0)


def lloyd(D: torch.Tensor, centers: torch.Tensor) -> Dict[str, Any]:
    k = centers.shape[0]
    D2 = (D * D).sum(dim=1)
    Cc = centers.clone()
    iterations = 0
    for iterations in range(1, C.KMEANS_MAX_ITERS + 1):
        d2 = _sqdist(D, Cc, D2)
        assign = d2.argmin(dim=1)
        new_centers = []
        for j in range(k):
            mask = assign == j
            n_j = int(mask.sum().item())
            if n_j == 0:
                other = Cc[[x for x in range(k) if x != j][0]]
                reseed = int(_sqdist(D, other.unsqueeze(0)).squeeze(1).argmax().item())
                new_centers.append(D[reseed])
            else:
                new_centers.append(D[mask].mean(dim=0))
        new_C = torch.stack(new_centers, dim=0)
        movement = float((new_C - Cc).norm(dim=1).max().item())
        Cc = new_C
        if movement <= C.KMEANS_TOL:
            break
    d2 = _sqdist(D, Cc, D2)
    assign = d2.argmin(dim=1)
    inertia = float(d2.gather(1, assign.unsqueeze(1)).sum().item())
    return {"centers": Cc, "assign": assign, "inertia": inertia, "iterations": iterations}


def _sign_canonical(U: torch.Tensor) -> torch.Tensor:
    U = U.clone()
    for j in range(U.shape[1]):
        i = int(U[:, j].abs().argmax().item())
        if float(U[i, j].item()) < 0.0:
            U[:, j] = -U[:, j]
    return U


def fit_two_mode(D: torch.Tensor, bin_index: int) -> Dict[str, Any]:
    """Fit the registered two-mode support to one bin's descriptor matrix."""
    if D.ndim != 2 or D.shape[1] != C.DIM:
        raise ValueError(f"expected [n,{C.DIM}] descriptors, got {tuple(D.shape)}")
    if D.shape[0] < 2:
        raise RuntimeError(f"bin {bin_index}: too few descriptors to fit a mode")

    best = None
    for restart in range(C.KMEANS_RESTARTS):
        rng = C.rng_from(C.FIT_SEED, bin_index, restart)
        init = kmeanspp_init(D, C.K_MODES, rng)
        result = lloyd(D, init)
        if best is None or result["inertia"] < best["inertia"]:
            best = dict(result, restart=restart)
    assert best is not None

    assign = best["assign"]
    arrays = {
        "ready": np.zeros((C.K_MODES,), dtype=np.uint8),
        "count": np.zeros((C.K_MODES,), dtype=np.int64),
        "mu": np.zeros((C.K_MODES, C.DIM), dtype=np.float32),
        "U": np.zeros((C.K_MODES, C.DIM, C.Q_TANGENT), dtype=np.float32),
        "lam": np.zeros((C.K_MODES, C.Q_TANGENT), dtype=np.float32),
        "R": np.zeros((C.K_MODES, C.Q_TANGENT), dtype=np.float32),
        "rho": np.zeros((C.K_MODES,), dtype=np.float32),
    }
    modes: List[Dict[str, Any]] = []
    for k in range(C.K_MODES):
        sel = D[assign == k]
        n_k = int(sel.shape[0])
        mode = {"index": k, "count": n_k, "ready": False}
        if n_k < C.N_MIN_MODE:
            modes.append(mode)
            continue
        mu = sel.mean(dim=0)
        Z = sel - mu
        cov = (Z.transpose(0, 1) @ Z) / float(n_k)
        trace = float(torch.diagonal(cov).sum().item())
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)[: C.Q_TANGENT]
        lam = evals[order].clamp_min(0.0)
        U = _sign_canonical(evecs[:, order])
        tr_perp = max(trace - float(lam.sum().item()), 0.0)
        R = C.TANGENT_MULT * torch.sqrt(lam + C.EPS_VAR)
        rho = C.ORTHO_MULT * float(tr_perp) ** 0.5

        ortho_error = float((U.transpose(0, 1) @ U - torch.eye(C.Q_TANGENT, dtype=U.dtype)).abs().max().item())
        rep = int(((sel - mu) ** 2).sum(dim=1).argmin().item())
        # map rep back to the bin descriptor index
        sel_indices = torch.nonzero(assign == k, as_tuple=False).squeeze(1)
        rep_index = int(sel_indices[rep].item())

        arrays["ready"][k] = 1
        arrays["count"][k] = n_k
        arrays["mu"][k] = mu.numpy().astype(np.float32)
        arrays["U"][k] = U.numpy().astype(np.float32)
        arrays["lam"][k] = lam.numpy().astype(np.float32)
        arrays["R"][k] = R.numpy().astype(np.float32)
        arrays["rho"][k] = np.float32(rho)
        mode.update({
            "ready": True,
            "mu": mu, "U": U, "lam": lam, "R": R, "rho": rho,
            "trace": trace, "trace_perp": tr_perp,
            "orthogonality_max_error": ortho_error,
            "rep_index": rep_index,
            "eigenvalues": [float(x) for x in lam.tolist()],
        })
        modes.append(mode)

    ready_modes = [m for m in modes if m["ready"]]
    if not ready_modes:
        raise RuntimeError(f"bin {bin_index}: no ready mode (pre-registered BLOCKED condition)")

    center_distance = float("nan")
    if len(ready_modes) == 2:
        center_distance = float((ready_modes[0]["mu"] - ready_modes[1]["mu"]).norm().item())
    diag = {
        "bin": int(bin_index),
        "n_total": int(D.shape[0]),
        "n_modes_ready": len(ready_modes),
        "n_modes_dropped": C.K_MODES - len(ready_modes),
        "counts": [int(m["count"]) for m in modes],
        "inertia": float(best["inertia"]),
        "iterations": int(best["iterations"]),
        "restart": int(best["restart"]),
        "center_distance": center_distance,
        "orthogonality_max_error": max(float(m.get("orthogonality_max_error", 0.0)) for m in modes),
        "rho": [float(m["rho"]) for m in modes if m["ready"]],
        "R_max": [float(m["R"].max().item()) for m in modes if m["ready"]],
        "rep_index": [int(m["rep_index"]) for m in ready_modes],
        "ready_mode_indices": [int(m["index"]) for m in ready_modes],
    }
    return {"arrays": arrays, "modes": modes, "diag": diag, "assign": assign}


def fit_all_bins(writer_bins: Sequence[torch.Tensor], t: int,
                 category_of_event: Sequence[str],
                 categories: Sequence[str] = C.CATEGORIES) -> Dict[str, Any]:
    """Fit every bin and assemble the fixed-shape serialized state arrays."""
    state = {
        "ready": np.zeros((C.BINS, C.K_MODES), dtype=np.uint8),
        "count": np.zeros((C.BINS, C.K_MODES), dtype=np.int64),
        "n_total": np.zeros((C.BINS,), dtype=np.int64),
        "mu": np.zeros((C.BINS, C.K_MODES, C.DIM), dtype=np.float32),
        "U": np.zeros((C.BINS, C.K_MODES, C.DIM, C.Q_TANGENT), dtype=np.float32),
        "lam": np.zeros((C.BINS, C.K_MODES, C.Q_TANGENT), dtype=np.float32),
        "R": np.zeros((C.BINS, C.K_MODES, C.Q_TANGENT), dtype=np.float32),
        "rho": np.zeros((C.BINS, C.K_MODES), dtype=np.float32),
    }
    bins_diag = []
    probe_index: Dict[str, Any] = {}
    for b in range(C.BINS):
        D = writer_bins[b].to(torch.float64)
        result = fit_two_mode(D, b)
        for key in ("ready", "count"):
            state[key][b] = result["arrays"][key]
        for key in ("mu", "U", "lam", "R", "rho"):
            state[key][b] = result["arrays"][key]
        state["n_total"][b] = D.shape[0]
        bins_diag.append(result["diag"])

        # Analysis-only representative indices for feature-space gap probes.
        reps = {}
        for mode in result["modes"]:
            if mode["ready"]:
                reps[f"mode_{mode['index']}"] = int(mode["rep_index"])
        cat_reps = {}
        per_event = writer_bins[b].shape[0] // max(t, 1)
        for cat in categories:
            event_indices = [e for e in range(t) if category_of_event[e] == cat]
            if not event_indices:
                continue
            positions = []
            for e in event_indices:
                positions.extend(range(e * per_event, (e + 1) * per_event))
            pos_t = torch.as_tensor(positions, dtype=torch.long)
            sub = D[pos_t]
            mu_cat = sub.mean(dim=0)
            nearest = int(((sub - mu_cat) ** 2).sum(dim=1).argmin().item())
            cat_reps[cat] = int(pos_t[nearest].item())
        probe_index[str(b)] = {"reps": reps, "cat_reps": cat_reps}

    return {"state": state, "bins": bins_diag, "probe_index": probe_index}
