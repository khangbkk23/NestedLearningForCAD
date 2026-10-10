# exps/test_hope_cad_ad01_normal_support.py
"""Focused validation for the AD-01 four-arm normal-support memory.

The load-bearing checks are:

1. `arm A` (global allocation, global scoring) reproduces a plain
   `CADICPatchCoresetV1` exactly, so the four-arm comparison is anchored to the
   frozen CADIC reference rather than to a reimplementation.
2. LOCAL scoring restricts the match set to the query patch's own bin without
   changing the stored support.
3. SPATIAL allocation splits capacity exactly, never exceeding the total budget.
4. Both factors are genuinely independent: changing one leaves the other intact.
"""

from __future__ import annotations

import torch
import pytest

from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    ALLOCATION_SPATIAL,
    GRID_SIDE,
    PATCHES,
    SCORING_GLOBAL,
    SCORING_LOCAL,
    NormalSupportMemory,
    arm_definitions,
    bin_assignment,
    per_bin_quota,
)
from models.cadic_patch_coreset_v1 import CADICPatchCoresetV1, CADICPatchCoresetConfig

DIM = 16
BUDGET = 640


def make_reference(budget: int = BUDGET) -> CADICPatchCoresetV1:
    return CADICPatchCoresetV1(
        CADICPatchCoresetConfig(budget=budget, dim=DIM, image_neighbors=9), device="cpu"
    )


def random_images(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, PATCHES, DIM, generator=generator)


# ------------------------------------------------------------------ structure


def test_bin_assignment_partitions_every_patch_once():
    bins = bin_assignment(4)
    assert bins.shape == (PATCHES,)
    counts = torch.bincount(bins, minlength=16)
    assert counts.tolist() == [49] * 16
    # Row-major: patch 0 is top-left, patch 783 is bottom-right.
    assert int(bins[0]) == 0
    assert int(bins[-1]) == 15
    # The first 7 patches of the first row share a bin.
    assert set(bins[:7].tolist()) == {0}


def test_bin_assignment_rejects_non_divisor_grid():
    with pytest.raises(ValueError):
        bin_assignment(5)


def test_per_bin_quota_sums_exactly():
    assert sum(per_bin_quota(640, 16)) == 640
    assert per_bin_quota(640, 16) == [40] * 16
    quotas = per_bin_quota(641, 16)
    assert sum(quotas) == 641
    assert max(quotas) - min(quotas) <= 1


# ------------------------------------------------- arm A equals CADIC reference


def test_arm_a_reproduces_cadic_reference_bitwise():
    images = random_images(3)
    reference = make_reference()
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM)

    for image in images:
        assert memory.update(image) == reference.update(image)

    assert memory.count == reference.count
    assert torch.equal(memory._global.features, reference.features)

    memory_scores = memory.score(images, scoring=SCORING_GLOBAL)
    reference_scores = reference.score(images)
    assert torch.equal(memory_scores[0], reference_scores[0])
    assert torch.equal(memory_scores[1], reference_scores[1])


def test_single_bin_allocation_matches_global_allocation():
    """With one bin the spatial arm degenerates to the global arm."""
    images = random_images(2, seed=3)
    spatial, _ = NormalSupportMemory(
        budget=BUDGET, grid=1, allocation=ALLOCATION_SPATIAL, dim=DIM
    ), None
    global_memory = NormalSupportMemory(
        budget=BUDGET, grid=1, allocation=ALLOCATION_GLOBAL, dim=DIM
    )
    for image in images:
        spatial.update(image)
        global_memory.update(image)
    assert torch.equal(spatial._banks[0].features, global_memory._global.features)
    assert torch.equal(
        spatial.score(images, scoring=SCORING_GLOBAL)[0],
        global_memory.score(images, scoring=SCORING_GLOBAL)[0],
    )


# ------------------------------------------------------- allocation behaviour


def test_spatial_allocation_respects_total_budget_exactly():
    images = random_images(6, seed=1)
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_SPATIAL, dim=DIM)
    for image in images:
        memory.update(image)
    assert memory.count == BUDGET
    assert memory.feature_bytes == BUDGET * DIM * 4
    occupancy = memory.occupancy()
    assert sum(row["count"] for row in occupancy["bins"]) == BUDGET
    for row in occupancy["bins"]:
        assert row["count"] <= row["capacity"]


def test_spatial_allocation_is_per_bin_and_does_not_exceed_quota():
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_SPATIAL, dim=DIM)
    for image in random_images(10, seed=7):
        memory.update(image)
    for bank in memory._banks:
        assert bank.count <= bank.config.budget


def test_global_allocation_banks_are_a_single_shared_bank():
    """LOCAL scoring under global allocation must not partition capacity."""
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM)
    for image in random_images(4, seed=2):
        memory.update(image)
    banks = memory.bin_banks()
    assert len(banks) == 16
    assert all(bank is memory._global for bank in banks)


# ---------------------------------------------------------- scoring behaviour


def test_local_scoring_restricts_candidates_to_the_patch_bin():
    """LOCAL scoring may only match exemplars admitted at the same bin.

    Under global allocation this must be a mask over the single shared bank: the
    stored support is unchanged, only the permitted candidate set shrinks. This
    is what makes arm B a pure scorer change relative to arm A.
    """
    images = random_images(4, seed=5)
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM)
    for image in images:
        memory.update(image)

    _, local_index, origin = memory._nearest_local(images[0])
    bins = bin_assignment(4)
    bank_bin = memory._bank_bin
    assert bank_bin is not None and bank_bin.shape[0] == memory.count

    for patch in range(PATCHES):
        bin_id, bank_row = origin[patch]
        assert bin_id == int(bins[patch].item())
        # The chosen exemplar must genuinely belong to the query patch's bin.
        assert int(bank_bin[bank_row].item()) == bin_id
        # `local_index` addresses the *masked* candidate set, so it is bounded by
        # the number of exemplars in that bin, not by the whole bank.
        assert 0 <= int(local_index[patch].item()) < int((bank_bin == bin_id).sum())

    # A query patch must never match an exemplar from a different bin.
    _, pixel_local = memory.score(images[:1], scoring=SCORING_LOCAL)
    _, pixel_global = memory.score(images[:1], scoring=SCORING_GLOBAL)
    assert torch.all(pixel_local >= pixel_global - 1e-5)


def test_arm_b_differs_from_arm_a_once_capacity_binds():
    """Scoring locality must have an observable effect, not be a silent no-op."""
    images = random_images(12, seed=23)
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM)
    for image in images:
        memory.update(image)
    assert memory._global.replaced_features > 0
    image_a, pixel_a = memory.score(images, scoring=SCORING_GLOBAL)
    image_b, pixel_b = memory.score(images, scoring=SCORING_LOCAL)
    assert not torch.allclose(pixel_a, pixel_b)
    assert not torch.allclose(image_a, image_b)


def test_local_scoring_never_exceeds_global_distance():
    """Restricting the candidate set can only increase nearest-neighbour distance.

    The bank must be saturated: an unfilled bank still contains every training
    patch, so any query patch finds its own source and both scorers agree
    trivially. Only a bank that has begun evicting can show the difference.
    """
    images = random_images(12, seed=11)
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM)
    for image in images:
        memory.update(image)
    assert memory._global.replaced_features > 0, "test requires an evicting bank"
    _, pixel_global = memory.score(images, scoring=SCORING_GLOBAL)
    _, pixel_local = memory.score(images, scoring=SCORING_LOCAL)
    assert torch.all(pixel_local >= pixel_global - 1e-5)
    # And the two must actually differ once capacity binds.
    assert not torch.allclose(pixel_local, pixel_global)


def test_scoring_does_not_mutate_the_bank():
    images = random_images(3, seed=13)
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_SPATIAL, dim=DIM)
    for image in images:
        memory.update(image)
    before = [bank.features.clone() for bank in memory._banks]
    memory.score(images, scoring=SCORING_GLOBAL)
    memory.score(images, scoring=SCORING_LOCAL)
    for bank, snapshot in zip(memory._banks, before):
        assert torch.equal(bank.features, snapshot)


# -------------------------------------------------------------- reproducibility


def test_update_is_deterministic_across_identical_instances():
    images = random_images(4, seed=17)
    first = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_SPATIAL, dim=DIM)
    second = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_SPATIAL, dim=DIM)
    for image in images:
        first.update(image)
        second.update(image)
    for a, b in zip(first._banks, second._banks):
        assert torch.equal(a.features, b.features)
    assert torch.equal(
        first.score(images, scoring=SCORING_LOCAL)[0],
        second.score(images, scoring=SCORING_LOCAL)[0],
    )


def test_state_round_trip_preserves_scores():
    images = random_images(3, seed=19)
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_SPATIAL, dim=DIM)
    for image in images:
        memory.update(image)
    expected = memory.score(images, scoring=SCORING_LOCAL)

    restored = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_SPATIAL, dim=DIM)
    restored.load_state_dict(memory.state_dict())
    actual = restored.score(images, scoring=SCORING_LOCAL)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_rejects_wrong_patch_count_and_shape():
    memory = NormalSupportMemory(budget=BUDGET, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM)
    with pytest.raises(ValueError):
        memory.update(torch.randn(PATCHES - 1, DIM))
    with pytest.raises(ValueError):
        memory.update(torch.randn(PATCHES, DIM + 1))
    with pytest.raises(ValueError):
        memory.score(torch.randn(1, PATCHES - 1, DIM), scoring=SCORING_GLOBAL)


def test_arm_definitions_cover_the_full_two_by_two_design():
    arms = arm_definitions()
    assert len(arms) == 4
    observed = {(a["allocation"], a["scoring"]) for a in arms}
    assert observed == {
        (ALLOCATION_GLOBAL, SCORING_GLOBAL),
        (ALLOCATION_GLOBAL, SCORING_LOCAL),
        (ALLOCATION_SPATIAL, SCORING_GLOBAL),
        (ALLOCATION_SPATIAL, SCORING_LOCAL),
    }


def test_grid_side_constant_matches_mvtec_patch_lattice():
    assert GRID_SIDE == 28
    assert PATCHES == 784
