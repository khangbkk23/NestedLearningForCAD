# exps/ad02/test_hope_cad_ad02_partition.py
"""Focused tests: the random control must match competition and remove locality."""

from __future__ import annotations

import torch

from exps.ad01.hope_cad_ad01_normal_support import bin_assignment, per_bin_quota
from exps.ad02.hope_cad_ad02_partition import (
    PATCHES,
    balanced_random_assignment,
    build_partition,
    partition_diagnostics,
    per_group_quota,
    spatial_assignment,
)


def test_spatial_assignment_matches_the_ad01_rule():
    for grid in (2, 4, 7):
        assert torch.equal(
            spatial_assignment(grid), bin_assignment(grid).cpu()
        ), f"grid {grid} diverges from AD-01"


def test_random_control_matches_group_count_sizes_and_capacities():
    """Competition-relevant quantities must be identical to the spatial 4x4 arm."""
    spatial, _, _ = build_partition("spatial", grid=4)
    random_assignment, diagnostics, _ = balanced_random_assignment(16, seed=10_000)

    spatial_sizes = torch.bincount(spatial, minlength=16).tolist()
    random_sizes = torch.bincount(random_assignment, minlength=16).tolist()
    assert random_sizes == spatial_sizes == [49] * 16
    assert per_group_quota(2500, 16) == per_bin_quota(2500, 16)
    assert diagnostics.n_groups == 16

    # Every position belongs to exactly one group in both designs, so the
    # per-image patch population per group is matched exactly.
    assert int(random_assignment.numel()) == PATCHES
    assert sorted(random_assignment.tolist()) == sorted(spatial.tolist())


def test_random_control_removes_spatial_adjacency():
    spatial, spatial_diagnostics, _ = build_partition("spatial", grid=4)
    random_assignment, diagnostics, _ = balanced_random_assignment(16, seed=10_000)
    assert diagnostics.adjacent_pairs == 0, "random groups must not be lattice neighbours"
    assert spatial_diagnostics.adjacent_pairs > 0, "spatial bins are adjacent by construction"


def test_random_control_is_not_a_renamed_spatial_bin():
    """The control must be spatially diffuse, not a relabelled grid."""
    _, spatial_diagnostics, _ = build_partition("spatial", grid=4)
    _, diagnostics, _ = balanced_random_assignment(16, seed=10_000)

    global_mean = diagnostics.global_distance_mean
    assert abs(diagnostics.intra_distance_mean - global_mean) / global_mean < 0.02
    assert spatial_diagnostics.intra_distance_mean < 0.5 * global_mean
    # A spatial bin sits inside one cell; a random group is spread over many.
    assert spatial_diagnostics.cell_load_max == 49
    assert diagnostics.cell_load_max <= 10
    assert diagnostics.empty_group_cells >= 0


def test_random_draws_are_reproducible_and_distinct():
    first, _, _ = balanced_random_assignment(16, seed=10_000)
    again, _, _ = balanced_random_assignment(16, seed=10_000)
    other, _, _ = balanced_random_assignment(16, seed=10_001)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_diagnostics_share_one_fingerprint_convention():
    assignment, diagnostics, _ = build_partition("spatial", grid=2)
    again = partition_diagnostics(assignment, label=diagnostics.label)
    assert again.fingerprint == diagnostics.fingerprint
    assert again.group_sizes == [196, 196, 196, 196]
