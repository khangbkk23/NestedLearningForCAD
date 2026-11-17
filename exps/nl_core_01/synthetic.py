"""Controlled synthetic continual associative-memory experiment."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

import torch

from .controls import (
    fixed_target_step,
    frozen_generator_step,
    learned_generator_step,
    nonnested_generator_step,
)
from .core import corrected_dgd_transition, standard_delta_transition


POLICIES = ("delta_rule", "fixed_target", "self_frozen", "self_learned", "nonnested")
TOPOLOGIES = ("orthogonal", "overlap", "conflict")
ORDERS = ((0, 1, 2), (2, 1, 0))
ETA_W, ALPHA_W, ETA_G, ALPHA_G, PERIOD = 0.10, 0.99, 0.05, 0.995, 4


def task_basis(dim: int, topology: str, task: int) -> torch.Tensor:
    if topology == "orthogonal":
        width = dim // 3
        indices = list(range(task * width, (task + 1) * width))
    elif topology == "overlap":
        indices = ((0, 1, 2, 3, 4, 5), (4, 5, 6, 7, 8, 9), (6, 7, 8, 9, 10, 11))[task]
    elif topology == "conflict":
        indices = (0, 1, 2, 3)
    else:
        raise ValueError(topology)
    basis = torch.zeros(dim, len(indices), dtype=torch.float64)
    basis[indices, torch.arange(len(indices))] = 1.0
    return basis


def target_map(dim: int, topology: str, task: int) -> torch.Tensor:
    mapping = torch.eye(dim, dtype=torch.float64)
    if topology == "conflict":
        mapping *= 1.0 if task % 2 == 0 else -1.0
    else:
        mapping *= 1.0 + 0.15 * task
    return mapping


def make_probes(dim: int, topology: str, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed + 10000)
    result = []
    for task in range(3):
        basis = task_basis(dim, topology, task)
        mapping = target_map(dim, topology, task)
        keys = [basis @ torch.randn(basis.shape[1], generator=generator, dtype=torch.float64) for _ in range(16)]
        result.append((torch.stack(keys), torch.stack([mapping @ key for key in keys])))
    return result


def mean_probe(memory: torch.Tensor, generator_memory: torch.Tensor, probe: tuple[torch.Tensor, torch.Tensor]) -> float:
    keys, values = probe
    losses = []
    for key, value in zip(keys, values):
        unit = key / key.norm()
        target = generator_memory @ value
        losses.append(float(0.5 * (memory @ unit - target).square().sum().item()))
    return sum(losses) / len(losses)


def run_one(policy: str, topology: str, order: tuple[int, ...], seed: int,
            *, dim: int = 12, repeats: int = 16) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(policy)
    memory = torch.zeros(dim, dim, dtype=torch.float64)
    generator_memory = torch.eye(dim, dtype=torch.float64)
    probes = make_probes(dim, topology, seed)
    train_rng = torch.Generator().manual_seed(seed + 20000 + 100 * TOPOLOGIES.index(topology) + 10 * ORDERS.index(order))
    boundaries = [[mean_probe(memory, generator_memory, p) for p in probes]]
    generator_norms = [float(generator_memory.norm().item())]
    event = 0
    for task in order:
        basis = task_basis(dim, topology, task)
        mapping = target_map(dim, topology, task)
        for _ in range(repeats):
            key = basis @ torch.randn(basis.shape[1], generator=train_rng, dtype=torch.float64)
            value = mapping @ key
            slow = (event + 1) % PERIOD == 0
            if policy == "delta_rule":
                memory, _, _ = standard_delta_transition(memory, key, value, eta=ETA_W)
            elif policy == "fixed_target":
                memory, _, _ = fixed_target_step(memory, key, value, eta=ETA_W, alpha=ALPHA_W)
            elif policy == "self_frozen":
                memory, _, _, _ = frozen_generator_step(memory, generator_memory, key, value, eta=ETA_W, alpha=ALPHA_W)
            elif policy == "self_learned":
                memory, generator_memory, _, _ = learned_generator_step(
                    memory, generator_memory, key, value, eta_w=ETA_W, alpha_w=ALPHA_W,
                    eta_g=ETA_G, alpha_g=ALPHA_G, slow=slow,
                )
            elif policy == "nonnested":
                memory, generator_memory, _, _ = nonnested_generator_step(
                    memory, generator_memory, key, value, eta_w=ETA_W, alpha_w=ALPHA_W,
                    eta_g=ETA_G, slow=slow,
                )
            event += 1
        boundaries.append([mean_probe(memory, generator_memory, p) for p in probes])
        generator_norms.append(float(generator_memory.norm().item()))
    acquisition = {str(task): boundaries[1 + order.index(task)][task] for task in range(3)}
    forgetting = {}
    for task in range(3):
        acquisition_boundary = 1 + order.index(task)
        later = [boundaries[i][task] for i in range(acquisition_boundary + 1, len(boundaries))]
        forgetting[str(task)] = max([0.0] + [x - boundaries[acquisition_boundary][task] for x in later])
    bytes_wg = 8 * (dim * dim + dim * dim) + 16
    bytes_delta = 8 * dim * dim + 8
    return {
        "policy": policy, "topology": topology, "order": list(order), "seed": seed,
        "dim": dim, "repeats": repeats, "boundaries": boundaries,
        "acquisition": acquisition, "forgetting": forgetting,
        "generator_norms": generator_norms, "finite": bool(torch.isfinite(memory).all() and torch.isfinite(generator_memory).all()),
        "persistent_bytes": bytes_delta if policy == "delta_rule" else bytes_wg,
        "events": event,
    }


def throughput(dim: int = 768, steps: int = 12) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(123)
    keys = [torch.randn(dim, generator=generator, dtype=torch.float32) for _ in range(steps)]
    values = [torch.randn(dim, generator=generator, dtype=torch.float32) for _ in range(steps)]
    W = torch.zeros(dim, dim, dtype=torch.float32)
    G = torch.eye(dim, dtype=torch.float32)
    start = time.perf_counter()
    for i in range(steps):
        W, G, _, _ = learned_generator_step(W, G, keys[i], values[i], eta_w=ETA_W, alpha_w=ALPHA_W,
                                            eta_g=ETA_G, alpha_g=ALPHA_G, slow=(i + 1) % PERIOD == 0)
    nl_seconds = time.perf_counter() - start
    W = torch.zeros(dim, dim, dtype=torch.float32)
    start = time.perf_counter()
    for i in range(steps):
        W, _, _ = standard_delta_transition(W, keys[i], values[i], eta=ETA_W)
    delta_seconds = time.perf_counter() - start
    return {"dim": dim, "steps": steps, "dtype": "float32", "nl_seconds": nl_seconds,
            "nl_updates_per_second": steps / nl_seconds, "delta_seconds": delta_seconds,
            "delta_updates_per_second": steps / delta_seconds,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "persistent_bytes_self_fp32": 4 * (2 * dim * dim) + 16,
            "persistent_bytes_delta_fp32": 4 * dim * dim + 8}


def run_all(seeds: tuple[int, ...] = (3, 11, 29), repeats: int = 16) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        for topology in TOPOLOGIES:
            for order in ORDERS:
                for policy in POLICIES:
                    runs.append(run_one(policy, topology, order, seed, repeats=repeats))
    summary: dict[str, Any] = {}
    for policy in POLICIES:
        selected = [row for row in runs if row["policy"] == policy]
        summary[policy] = {
            "mean_acquisition": sum(sum(float(v) for v in row["acquisition"].values()) / 3 for row in selected) / len(selected),
            "mean_forgetting": sum(sum(float(v) for v in row["forgetting"].values()) / 3 for row in selected) / len(selected),
            "max_forgetting": max(max(float(v) for v in row["forgetting"].values()) for row in selected),
            "persistent_bytes": sorted({row["persistent_bytes"] for row in selected}),
            "all_finite": all(row["finite"] for row in selected),
        }
    return {"schema": "nl_core_01_synthetic_v1", "seeds": list(seeds), "orders": [list(x) for x in ORDERS],
            "topologies": list(TOPOLOGIES), "constants": {"eta_w": ETA_W, "alpha_w": ALPHA_W,
            "eta_g": ETA_G, "alpha_g": ALPHA_G, "slow_period": PERIOD}, "summary": summary,
            "runs": runs, "throughput": throughput()}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_all()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
