"""Read-only generator dynamics and retention-credit forensics.

The canonical NL-CORE-01 module is imported as an oracle and never edited.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import torch

from exps.nl_core_01.core import CorrectedSelfGeneratedDGD


def factorization_check(dim: int = 5, seed: int = 13) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    k = torch.randn(dim, generator=g, dtype=torch.float64)
    v = torch.randn(dim, generator=g, dtype=torch.float64)
    alpha, eta = 0.995, 0.05
    lhs = alpha * G - eta * torch.outer(G @ k - G @ v, k)
    A = alpha * torch.eye(dim, dtype=torch.float64) + eta * torch.outer(v - k, k)
    rhs = G @ A
    return {"max_abs_error": float((lhs - rhs).abs().max().item()),
            "relative_error": float((lhs - rhs).norm().item() / (rhs.norm().item() + 1e-30))}


def stationary_theory(alpha: float, eta: float, c: float, dim: int = 4) -> dict[str, Any]:
    k = torch.zeros(dim, dtype=torch.float64)
    k[0] = 1.0
    v = c * k
    A = alpha * torch.eye(dim, dtype=torch.float64) + eta * torch.outer(v - k, k)
    eig = torch.linalg.eigvals(A).real.tolist()
    parallel = alpha + eta * (c - 1.0)
    fixed_nonzero = abs(parallel - 1.0) < 1e-12
    return {"alpha": alpha, "eta": eta, "c": c, "eigenvalues": eig,
            "lambda_parallel": parallel, "lambda_perp": alpha,
            "spectral_radius": max(abs(x) for x in eig),
            "stable": max(abs(x) for x in eig) < 1.0,
            "neutral": abs(max(abs(x) for x in eig) - 1.0) < 1e-12,
            "nonzero_fixed_subspace": fixed_nonzero,
            "fixed_point_scalar": None if fixed_nonzero else 0.0}


def stationary_run(c: float, *, events: int = 5000, alpha: float = 0.995, eta: float = 0.05,
                   dim: int = 4) -> dict[str, Any]:
    key = torch.zeros(dim, dtype=torch.float64)
    key[0] = 1.0
    value = c * key
    core = CorrectedSelfGeneratedDGD(dim, eta_w=0.1, alpha_w=0.99, eta_g=eta,
                                     alpha_g=alpha, slow_period=4).double()
    checkpoints = {0: core.generator.detach().clone()}
    post_losses = []
    for event in range(events):
        result = core.step(key, value)
        if result.slow_update:
            checkpoints[event + 1] = core.generator.detach().clone()
        if event in (0, events // 2, events - 1):
            post_losses.append({"event": event + 1, "generator_norm": float(core.generator.norm()),
                                "target_norm": float((core.generator @ value).norm()),
                                "memory_norm": float(core.memory.norm()), "post_loss": result.post_loss})
    A = alpha * torch.eye(dim, dtype=torch.float64) + eta * torch.outer(value - key, key)
    slow_updates = events // 4
    predicted = torch.linalg.matrix_power(A, slow_updates)
    observed = core.generator.detach()
    return {"c": c, "events": events, "slow_updates": slow_updates,
            "theory": stationary_theory(alpha, eta, c, dim),
            "observed_generator_norm": float(observed.norm()),
            "observed_target_norm": float((observed @ value).norm()),
            "analytic_generator_norm": float(predicted.norm()),
            "max_generator_prediction_error": float((observed - predicted).abs().max()),
            "finite": bool(torch.isfinite(core.generator).all() and torch.isfinite(core.memory).all()),
            "checkpoints": post_losses}


def noncollinear_run(*, events: int = 5000, dim: int = 4) -> dict[str, Any]:
    key = torch.zeros(dim, dtype=torch.float64)
    key[0] = 1.0
    value = torch.zeros(dim, dtype=torch.float64)
    value[0], value[1] = 0.8, 0.6
    alpha, eta = 0.995, 0.05
    core = CorrectedSelfGeneratedDGD(dim, eta_w=0.1, alpha_w=0.99, eta_g=eta,
                                     alpha_g=alpha, slow_period=4).double()
    for _ in range(events):
        core.step(key, value)
    A = alpha * torch.eye(dim, dtype=torch.float64) + eta * torch.outer(value - key, key)
    predicted = torch.linalg.matrix_power(A, events // 4)
    return {"key": key.tolist(), "value": value.tolist(),
            "spectral_radius": float(torch.linalg.eigvals(A).abs().max()),
            "max_generator_prediction_error": float((core.generator - predicted).abs().max()),
            "generator_norm": float(core.generator.norm()),
            "target_norm": float((core.generator @ value).norm()),
            "finite": bool(torch.isfinite(core.generator).all() and torch.isfinite(core.memory).all())}


def checkpoint_parity(events: int = 400, dim: int = 4) -> dict[str, Any]:
    key = torch.zeros(dim, dtype=torch.float64)
    key[0] = 1.0
    value = 0.8 * key
    full = CorrectedSelfGeneratedDGD(dim).double()
    split = CorrectedSelfGeneratedDGD(dim).double()
    for _ in range(events // 2):
        full.step(key, value)
        split.step(key, value)
    payload = io.BytesIO()
    torch.save(split.state_dict(), payload)
    resumed = CorrectedSelfGeneratedDGD(dim).double()
    resumed.load_state_dict(torch.load(io.BytesIO(payload.getvalue()), weights_only=True))
    for _ in range(events // 2):
        full.step(key, value)
        resumed.step(key, value)
    return {"memory_equal": bool(torch.equal(full.memory, resumed.memory)),
            "generator_equal": bool(torch.equal(full.generator, resumed.generator)),
            "event_equal": bool(torch.equal(full.event_count, resumed.event_count)),
            "slow_count_equal": bool(torch.equal(full.slow_update_count, resumed.slow_update_count))}


def artifact_audit() -> dict[str, Any]:
    path = Path("results/hope_cad/nl_core_01/synthetic.json")
    payload = json.loads(path.read_text())
    runs = payload["runs"]
    by_policy = {}
    for policy in payload["summary"]:
        rows = [row for row in runs if row["policy"] == policy]
        by_policy[policy] = {"records": len(rows), "seeds": sorted({row["seed"] for row in rows}),
                             "orders": sorted({tuple(row["order"]) for row in rows}),
                             "topologies": sorted({row["topology"] for row in rows})}
    return {"total_records": len(runs), "per_policy": by_policy,
            "independent_seed_draws_per_policy": 3,
            "designed_order_topology_cells_per_policy": 6}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"factorization": factorization_check(),
              "stationary": {str(c): stationary_run(c) for c in (0.8, 1.0, 1.1, 1.2)},
              "noncollinear": noncollinear_run(),
              "checkpoint": checkpoint_parity(),
              "artifact_audit": artifact_audit()}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
