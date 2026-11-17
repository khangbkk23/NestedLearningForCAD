"""Tiny synthetic continual associative-memory benchmark for NL-CORE-00."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .core import CanonicalNLCore


def run_benchmark(seed: int = 20261114, dim: int = 12, repeats: int = 24) -> dict[str, Any]:
    """Run three orthogonal task subspaces with exact linear ground truth."""
    if dim % 3 or repeats < 2:
        raise ValueError("dim must be divisible by three and repeats >= 2")
    g = torch.Generator(device="cpu").manual_seed(seed)
    core = CanonicalNLCore(dim, eta_max=0.25, controller_lr=0.03).double()
    tasks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for task in range(3):
        start, stop = task * dim // 3, (task + 1) * dim // 3
        basis = torch.zeros(dim, dim // 3, dtype=torch.float64)
        basis[start:stop] = torch.eye(dim // 3, dtype=torch.float64)
        transform = torch.eye(dim, dtype=torch.float64)
        transform[start:stop, start:stop] *= 1.0 + 0.15 * task
        tasks.append((basis, transform))

    def probe(task: int) -> float:
        basis, transform = tasks[task]
        losses = []
        for _ in range(12):
            q = torch.randn(dim // 3, generator=g, dtype=torch.float64)
            key = basis @ q
            value = transform @ key
            losses.append(core.evaluate(key, value))
        return float(sum(losses) / len(losses))

    initial = [probe(i) for i in range(3)]
    phases: list[dict[str, Any]] = []
    for task, (basis, transform) in enumerate(tasks):
        for _ in range(repeats):
            q = torch.randn(dim // 3, generator=g, dtype=torch.float64)
            key = basis @ q
            core.step(key, transform @ key)
        phases.append({
            "task": task,
            "probe_loss": [probe(i) for i in range(3)],
            "controller_norm": float(core.controller.norm().item()),
            "memory_norm": float(core.memory.norm().item()),
            "updates": int(core.update_count.item()),
        })
    final = phases[-1]["probe_loss"]
    return {
        "schema": "nl_core_00_synthetic_v1",
        "seed": seed,
        "dim": dim,
        "repeats": repeats,
        "persistent_bytes": core.persistent_bytes(),
        "initial_probe_loss": initial,
        "phases": phases,
        "final_probe_loss": final,
        "controller_adapted": bool(phases[-1]["controller_norm"] > 0.0),
        "finite": bool(torch.isfinite(core.memory).all() and torch.isfinite(core.controller).all()),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
