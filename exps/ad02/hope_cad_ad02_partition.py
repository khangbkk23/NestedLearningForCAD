# exps/ad02/hope_cad_ad02_partition.py
"""AD-02 partition generators: spatial grids and the balanced random control.

AD-01 varied allocation with a single geometry, the `4x4` grid over the 28x28
patch lattice, so it could not tell *spatial locality* apart from *reduced
per-group competition*. AD-02 adds a control that holds every competition-
relevant quantity fixed and destroys only the spatial structure:

    spatial `g x g`     each group is one contiguous block of the lattice
    random R4           each group holds the same number of positions, but no
                        two positions in a group are lattice neighbours

The random draw is validated and ranked using **position geometry only**. It
never sees a feature, a label, a score or a metric, so the selection cannot tune
the control toward or away from the endpoint.

Why the control is not "renamed bins": the group count, every group's size, every
group's exemplar quota and the number of patches each group receives per image
are identical to the spatial `4x4` arm. The only quantity that differs is whether
the positions that share a group are neighbours on the lattice, which is exactly
the variable H1 claims is responsible for the AD-01 gain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

GRID_SIDE = 28
PATCHES = GRID_SIDE * GRID_SIDE

# Predeclared acceptance bounds for a random draw. Physical meaning only.
#
# An earlier draft also required every group to occupy every spatial cell at
# least once. That is geometrically infeasible: with 49 positions spread over 16
# cells, a group leaves a cell empty with probability about 0.04, and requiring
# 16 such groups to be complete is a coupon-collector event. Measured rejection
# was 200/200 draws. The requirement is therefore replaced by an upper bound on
# per-cell concentration plus a reported empty-cell count, which is what the
# balance claim actually needs.
RANDOM_MAX_CELL_LOAD = 10


def empty_group_cells(assignment: torch.Tensor, cell_grid: int = 4) -> int:
    """Number of (group, spatial cell) pairs with no member. Geometry only."""
    groups = assignment.detach().cpu().numpy()
    cells = spatial_assignment(cell_grid).detach().cpu().numpy()
    total = 0
    for group in range(int(groups.max()) + 1):
        positions = np.nonzero(groups == group)[0]
        total += int((np.bincount(cells[positions], minlength=cell_grid ** 2) == 0).sum())
    return total


def spatial_assignment(
    grid: int, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Bin id of every patch in row-major 28x28 order for a `grid x grid` split.

    Byte-for-byte the AD-01 rule (`bin_assignment`), re-derived here so AD-02
    does not depend on AD-01 internals for its own partition definition.
    """
    if GRID_SIDE % grid:
        raise ValueError(f"grid must divide {GRID_SIDE}, got {grid}")
    stride = GRID_SIDE // grid
    rows = torch.arange(GRID_SIDE) // stride
    cols = torch.arange(GRID_SIDE) // stride
    bin_of = (rows[:, None] * grid + cols[None, :]).reshape(-1)
    return bin_of.to(torch.long).to(device)


def per_group_quota(total: int, n_groups: int) -> list[int]:
    """Split `total` across `n_groups` as evenly as possible, remainder first."""
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    if total < n_groups:
        raise ValueError(f"budget {total} is smaller than {n_groups} groups")
    base, remainder = divmod(total, n_groups)
    return [base + (1 if index < remainder else 0) for index in range(n_groups)]


def group_sizes_for_positions(n_positions: int, n_groups: int) -> list[int]:
    """Position counts per group, as even as possible."""
    base, remainder = divmod(n_positions, n_groups)
    return [base + (1 if index < remainder else 0) for index in range(n_groups)]


def _lattice_xy(positions: np.ndarray) -> np.ndarray:
    return np.stack([positions // GRID_SIDE, positions % GRID_SIDE], axis=1).astype(
        np.float64
    )


def _neighbours(position: int) -> list[int]:
    row, col = divmod(position, GRID_SIDE)
    out = []
    if row > 0:
        out.append(position - GRID_SIDE)
    if row < GRID_SIDE - 1:
        out.append(position + GRID_SIDE)
    if col > 0:
        out.append(position - 1)
    if col < GRID_SIDE - 1:
        out.append(position + 1)
    return out


_NEIGHBOURS: list[list[int]] = [_neighbours(p) for p in range(PATCHES)]


def mean_pairwise_distance(points: np.ndarray) -> float:
    """Mean Euclidean distance over all unordered pairs of `[N,2]` points."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    distance = np.sqrt((diff ** 2).sum(-1))
    upper = np.triu_indices(n, k=1)
    return float(distance[upper].mean())


@dataclass
class PartitionDiagnostics:
    label: str
    n_groups: int
    group_sizes: list[int]
    adjacent_pairs: int
    cell_load_min: int
    cell_load_max: int
    empty_group_cells: int
    intra_distance_mean: float
    intra_distance_min: float
    intra_distance_max: float
    global_distance_mean: float
    imbalance: float
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_groups": self.n_groups,
            "group_sizes": self.group_sizes,
            "adjacent_pairs": self.adjacent_pairs,
            "cell_load_min": self.cell_load_min,
            "cell_load_max": self.cell_load_max,
            "empty_group_cells": self.empty_group_cells,
            "intra_distance_mean": self.intra_distance_mean,
            "intra_distance_min": self.intra_distance_min,
            "intra_distance_max": self.intra_distance_max,
            "global_distance_mean": self.global_distance_mean,
            "imbalance": self.imbalance,
            "fingerprint": self.fingerprint,
        }


def assignment_fingerprint(assignment: torch.Tensor) -> str:
    payload = assignment.detach().to(torch.int64).cpu().numpy().astype("<i8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def partition_diagnostics(
    assignment: torch.Tensor, *, label: str, cell_grid: int = 4
) -> PartitionDiagnostics:
    """Geometry-only description of a partition. Uses no data or metric."""
    groups = assignment.detach().cpu().numpy()
    n_groups = int(groups.max()) + 1
    positions_by_group = [np.nonzero(groups == g)[0] for g in range(n_groups)]

    adjacent_pairs = 0
    for positions in positions_by_group:
        members = set(int(p) for p in positions)
        for position in members:
            adjacent_pairs += sum(1 for n in _NEIGHBOURS[position] if n in members)
    adjacent_pairs //= 2

    cells = spatial_assignment(cell_grid).detach().cpu().numpy()
    cell_loads = [
        np.bincount(cells[positions], minlength=cell_grid * cell_grid)
        for positions in positions_by_group
    ]
    cell_load_min = int(min(int(load.min()) for load in cell_loads))
    cell_load_max = int(max(int(load.max()) for load in cell_loads))

    intra = [
        mean_pairwise_distance(_lattice_xy(positions))
        for positions in positions_by_group
    ]
    global_mean = mean_pairwise_distance(_lattice_xy(np.arange(PATCHES)))
    intra_mean = float(np.mean(intra))
    return PartitionDiagnostics(
        label=label,
        n_groups=n_groups,
        group_sizes=[int(len(p)) for p in positions_by_group],
        adjacent_pairs=int(adjacent_pairs),
        cell_load_min=cell_load_min,
        cell_load_max=cell_load_max,
        empty_group_cells=int(sum(int((load == 0).sum()) for load in cell_loads)),
        intra_distance_mean=intra_mean,
        intra_distance_min=float(np.min(intra)),
        intra_distance_max=float(np.max(intra)),
        global_distance_mean=global_mean,
        imbalance=float(abs(intra_mean - global_mean) / global_mean),
        fingerprint=assignment_fingerprint(assignment),
    )


def _greedy_random_draw(
    rng: np.random.Generator, sizes: Sequence[int]
) -> np.ndarray | None:
    """One attempt at a group-balanced, adjacency-free random partition.

    Positions are taken in random order and placed in the currently smallest
    group that has room and contains none of the position's lattice neighbours.
    Returns `None` if the attempt dead-ends.
    """
    n_groups = len(sizes)
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    members: list[set[int]] = [set() for _ in range(n_groups)]
    for position in rng.permutation(PATCHES):
        position = int(position)
        neighbours = _NEIGHBOURS[position]
        candidates = [
            g
            for g in range(n_groups)
            if len(groups[g]) < sizes[g]
            and not any(n in members[g] for n in neighbours)
        ]
        if not candidates:
            return None
        smallest = min(len(groups[g]) for g in candidates)
        candidates = [g for g in candidates if len(groups[g]) == smallest]
        chosen = int(candidates[rng.integers(len(candidates))])
        groups[chosen].append(position)
        members[chosen].add(position)

    assignment = np.empty(PATCHES, dtype=np.int64)
    for g, positions in enumerate(groups):
        assignment[positions] = g
    return assignment


def balanced_random_assignment(
    n_groups: int,
    *,
    seed: int,
    n_candidates: int = 24,
    max_attempts_per_candidate: int = 40,
    cell_grid: int = 4,
    max_cell_load: int = RANDOM_MAX_CELL_LOAD,
) -> tuple[torch.Tensor, PartitionDiagnostics, dict[str, Any]]:
    """Draw a balanced, non-spatial random partition of the 28x28 lattice.

    Accepts a draw only when every group has the same position count as the
    matched spatial partition, no two members of a group are 4-neighbours, and no
    group concentrates more than `max_cell_load` positions in one spatial cell.
    Among accepted draws the smallest geometry-only spatial-spread imbalance
    wins. No feature, label, score or metric is read anywhere in this function.
    """
    sizes = group_sizes_for_positions(PATCHES, n_groups)
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, PartitionDiagnostics] | None = None
    accepted = 0
    attempts = 0
    for _ in range(n_candidates):
        for _ in range(max_attempts_per_candidate):
            attempts += 1
            draw = _greedy_random_draw(rng, sizes)
            if draw is None:
                continue
            assignment = torch.from_numpy(draw).to(torch.long)
            diagnostics = partition_diagnostics(
                assignment, label=f"random{n_groups}_seed{seed}", cell_grid=cell_grid
            )
            if diagnostics.adjacent_pairs != 0:
                continue
            if diagnostics.cell_load_max > max_cell_load:
                continue
            accepted += 1
            if best is None or diagnostics.imbalance < best[0]:
                best = (diagnostics.imbalance, draw, diagnostics)
            break

    if best is None:
        raise RuntimeError(
            f"no valid balanced random partition for n_groups={n_groups}, seed={seed}"
        )
    assignment = torch.from_numpy(best[1]).to(torch.long)
    diagnostics = partition_diagnostics(
        assignment, label=f"random{n_groups}_seed{seed}", cell_grid=cell_grid
    )
    meta = {
        "n_candidates": n_candidates,
        "attempts": attempts,
        "accepted_draws": accepted,
        "seed": seed,
        "sizes": sizes,
        "constraints": {
            "group_size_fixed": True,
            "no_lattice_adjacency": True,
            "max_cell_load": max_cell_load,
            "cell_grid": cell_grid,
        },
    }
    return assignment, diagnostics, meta


def build_partition(
    kind: str, *, grid: int = 4, seed: int = 0, **kwargs: Any
) -> tuple[torch.Tensor, PartitionDiagnostics, dict[str, Any]]:
    """Partition factory. `kind` is `spatial` or `random`."""
    if kind == "spatial":
        assignment = spatial_assignment(grid)
        diagnostics = partition_diagnostics(assignment, label=f"spatial{grid}x{grid}")
        return assignment, diagnostics, {"seed": None, "kind": kind, "grid": grid}
    if kind == "random":
        return balanced_random_assignment(grid * grid, seed=seed, **kwargs)
    raise ValueError(f"unknown partition kind {kind!r}")
