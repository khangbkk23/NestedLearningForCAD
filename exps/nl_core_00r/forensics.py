"""Focused mathematical and matched-control forensics for NL-CORE-00.

This module never edits ``exps.nl_core_00.core``.  It uses that module as an
oracle and implements controls independently so that an objective mismatch or
an apparent causal gain cannot be hidden by changing the reference code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import functional as F

from exps.nl_core_00.core import CanonicalNLCore


def dgd_oracle(memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               eta: float, alpha: float) -> torch.Tensor:
    """The retained recurrence used by the NL-CORE-00 oracle."""
    k = F.normalize(key, dim=0)
    e = memory @ k - value
    return memory @ (alpha * torch.eye(memory.shape[0], dtype=memory.dtype) - eta * torch.outer(k, k)) - eta * torch.outer(e, k)


def delta_rule(memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor, eta: float) -> torch.Tensor:
    """Standard reconstruction Delta Rule: W <- W - eta (Wk-v) k^T."""
    k = F.normalize(key, dim=0)
    e = memory @ k - value
    return memory - eta * torch.outer(e, k)


def fixed_controls(memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   eta: float = 0.10, alpha: float = 0.95) -> torch.Tensor:
    return dgd_oracle(memory, key, value, eta, alpha)


def controller_controls(core: CanonicalNLCore, memory: torch.Tensor, controller: torch.Tensor,
                        key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = F.normalize(key, dim=0)
    e = memory @ k - value
    eta, alpha = core._controls(controller, k, e)
    return eta, alpha, e


def frozen_controller_step(core: CanonicalNLCore, memory: torch.Tensor, controller: torch.Tensor,
                           key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    eta, alpha, _ = controller_controls(core, memory, controller, key, value)
    return dgd_oracle(memory, key, value, float(eta), float(alpha))


def nonnested_step(core: CanonicalNLCore, memory: torch.Tensor, controller: torch.Tensor,
                   key: torch.Tensor, value: torch.Tensor, beta: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """Parameter-matched control whose controller objective is independent of W'.

    It has the same C shape and input as NL, but updates C toward zero logits
    using only a control regularizer; no gradient passes through the fast
    transition.  This is a deliberately weak but explicit non-nested control.
    """
    k = F.normalize(key, dim=0)
    e = memory @ k - value
    z = torch.cat((k, e.norm().reshape(1)))
    c = controller.detach().clone().requires_grad_(True)
    logits = c @ z
    regularizer = 0.5 * logits.square().sum()
    grad = torch.autograd.grad(regularizer, c)[0]
    eta, alpha = core._controls(c.detach(), k, e)
    next_memory = dgd_oracle(memory, key, value, float(eta), float(alpha))
    return next_memory, (c - beta * grad).detach()


def loss(memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> float:
    k = F.normalize(key, dim=0)
    return float(0.5 * (memory @ k - value).square().sum().item())


def constant_key_fixed_point(dim: int = 4, eta: float = 0.1, alpha: float = 1.0,
                             steps: int = 200) -> dict[str, Any]:
    key = torch.zeros(dim, dtype=torch.float64)
    key[0] = 1.0
    value = torch.zeros(dim, dtype=torch.float64)
    value[0] = 1.0
    memory = torch.zeros(dim, dim, dtype=torch.float64)
    ys = []
    for _ in range(steps):
        memory = dgd_oracle(memory, key, value, eta, alpha)
        ys.append(float((memory @ key)[0]))
    predicted = eta / (1.0 - alpha + 2.0 * eta)
    return {"eta": eta, "alpha": alpha, "observed_final": ys[-1], "predicted_fixed_point": predicted,
            "perfect_association_update": float((dgd_oracle(torch.eye(dim, dtype=torch.float64), key, value, eta, alpha) @ key)[0]),
            "trajectory": ys}


def make_tasks(dim: int, conflicting: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Three task mappings; conflicting=True reuses the same key subspace."""
    tasks = []
    width = dim // 3
    for task in range(3):
        start = 0 if conflicting else task * width
        basis = torch.zeros(dim, width, dtype=torch.float64)
        basis[start:start + width] = torch.eye(width, dtype=torch.float64)
        scale = (1.0 + 0.20 * task) if not conflicting else (1.0 if task % 2 == 0 else -1.0)
        mapping = torch.eye(dim, dtype=torch.float64) * scale
        tasks.append((basis, mapping))
    return tasks


def run_stream(name: str, *, conflicting: bool, seed: int = 20261114, dim: int = 12,
               repeats: int = 24) -> dict[str, Any]:
    torch.manual_seed(seed)
    core = CanonicalNLCore(dim, eta_max=0.20, controller_lr=0.02).double()
    core_template = CanonicalNLCore(dim, eta_max=0.20, controller_lr=0.02).double()
    # A deterministic nonzero controller makes frozen/non-nested controls
    # parameter-matched and distinguishes them from a scalar fixed schedule.
    controller0 = torch.linspace(-0.03, 0.03, 2 * (dim + 1), dtype=torch.float64).reshape(2, dim + 1)
    memory = torch.zeros(dim, dim, dtype=torch.float64)
    controller = controller0.clone()
    tasks = make_tasks(dim, conflicting)
    generator = torch.Generator().manual_seed(seed + (91 if conflicting else 0))

    def probe() -> list[float]:
        values = []
        for _, (basis, mapping) in enumerate(tasks):
            rows = []
            for _ in range(12):
                q = torch.randn(basis.shape[1], generator=generator, dtype=torch.float64)
                key = basis @ q
                rows.append(loss(memory, key, mapping @ key))
            values.append(float(sum(rows) / len(rows)))
        return values

    def update(key: torch.Tensor, value: torch.Tensor) -> None:
        nonlocal memory, controller
        if name == "nl":
            core.memory.copy_(memory)
            core.controller.copy_(controller)
            core.step(key, value)
            memory, controller = core.memory.detach().clone(), core.controller.detach().clone()
        elif name == "fixed_controller":
            memory = fixed_controls(memory, key, value)
        elif name == "frozen_controller":
            memory = frozen_controller_step(core_template, memory, controller, key, value)
        elif name == "delta_rule":
            memory = delta_rule(memory, key, value, eta=0.10)
        elif name == "nonnested_controller":
            memory, controller = nonnested_step(core_template, memory, controller, key, value)
        else:
            raise ValueError(name)

    matrix = [probe()]
    controller_norms = [float(controller.norm().item())]
    for task, (basis, mapping) in enumerate(tasks):
        for _ in range(repeats):
            q = torch.randn(basis.shape[1], generator=generator, dtype=torch.float64)
            key = basis @ q
            update(key, mapping @ key)
        matrix.append(probe())
        controller_norms.append(float(controller.norm().item()))
    return {"policy": name, "conflicting": conflicting, "seed": seed, "dim": dim,
            "repeats": repeats, "probe_matrix": matrix, "controller_norms": controller_norms,
            "persistent_bytes": int(memory.numel() * memory.element_size() + controller.numel() * controller.element_size() + 8),
            "finite": bool(torch.isfinite(memory).all() and torch.isfinite(controller).all())}


def throughput(dim: int = 768, steps: int = 20, dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    torch.manual_seed(7)
    core = CanonicalNLCore(dim).to(dtype=dtype)
    keys = torch.randn(steps + 2, dim, dtype=dtype)
    values = torch.randn(steps + 2, dim, dtype=dtype)
    for i in range(2):
        core.step(keys[i], values[i])
    start = time.perf_counter()
    for i in range(2, steps + 2):
        core.step(keys[i], values[i])
    elapsed = time.perf_counter() - start
    W = torch.zeros(dim, dim, dtype=dtype)
    start_delta = time.perf_counter()
    for i in range(steps):
        W = delta_rule(W, keys[i], values[i], eta=0.1)
    delta_elapsed = time.perf_counter() - start_delta
    return {"dim": dim, "steps": steps, "dtype": str(dtype), "nl_seconds": elapsed,
            "nl_updates_per_second": steps / elapsed, "delta_seconds": delta_elapsed,
            "delta_updates_per_second": steps / delta_elapsed,
            "persistent_bytes_fp32": 4 * (dim * dim + 2 * (dim + 1)) + 8,
            "persistent_bytes_fp64": 8 * (dim * dim + 2 * (dim + 1)) + 8}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "fixed_point_alpha_1": constant_key_fixed_point(alpha=1.0),
        "fixed_point_alpha_0_95": constant_key_fixed_point(alpha=0.95),
        "streams": {f"{'conflict' if c else 'orthogonal'}_{p}": run_stream(p, conflicting=c)
                    for c in (False, True)
                    for p in ("nl", "fixed_controller", "frozen_controller", "delta_rule", "nonnested_controller")},
        "throughput": throughput(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
