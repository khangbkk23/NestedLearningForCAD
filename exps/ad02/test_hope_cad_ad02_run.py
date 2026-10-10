# exps/ad02/test_hope_cad_ad02_run.py
"""Focused tests: instrumented memory, occupancy accounting and coverage."""

from __future__ import annotations

import numpy as np
import torch

from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    ALLOCATION_SPATIAL,
    NormalSupportMemory,
    PATCHES,
)
from exps.ad02.hope_cad_ad02_memory import InstrumentedNormalSupportMemory
from exps.ad02.hope_cad_ad02_partition import (
    balanced_random_assignment,
    spatial_assignment,
)

DIM = 4


def _images(count: int, *, seed: int, scale: float = 1.0, offset: float = 0.0):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(PATCHES, DIM, generator=generator) * scale + offset
        for _ in range(count)
    ]


def _memory(allocation: str, *, budget: int, grid: int = 4, assignment=None):
    return InstrumentedNormalSupportMemory(
        budget=budget,
        grid=grid,
        allocation=allocation,
        dim=DIM,
        image_neighbors=9,
        device="cpu",
        fast=True,
        assignment=assignment,
        partition_kind="spatial" if assignment is None else "random",
    )


def test_global_occupancy_is_disjoint_in_both_implementations():
    """Correction I1: a global bank must never report sixteen copies of itself."""
    images = _images(3, seed=0)
    budget = 32

    ad01 = NormalSupportMemory(
        budget=budget, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM, device="cpu",
        fast=True,
    )
    for image in images:
        ad01.update(image)
    corrected = ad01.occupancy()
    assert len(corrected["bins"]) == 1
    assert sum(row["count"] for row in corrected["bins"]) == corrected["total_count"]
    assert corrected["partitioned"] is False

    ad02 = _memory(ALLOCATION_GLOBAL, budget=budget)
    for index, image in enumerate(images):
        ad02.update(image, task_id=0, image_index=index, global_step=index)
    fixed = ad02.occupancy()
    assert len(fixed["bins"]) == 1
    assert sum(row["count"] for row in fixed["bins"]) == fixed["total_count"]
    assert fixed["capacity_total"] == budget
    assert sum(row["replaced"] for row in fixed["bins"]) == int(
        ad02._banks[0].replaced_features
    )


def test_the_archived_ad01_record_keeps_its_original_double_counted_view():
    """The code fix must not rewrite the archived AD-01 artifact."""
    import json
    from pathlib import Path

    path = Path(
        "results/hope_cad/ad01_phase0/arms_gpu_v5/arms_results.json"
    )
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    globals_seen = 0
    for record in payload["runs"]:
        if record["allocation"] != "global":
            continue
        globals_seen += 1
        occupancy = record["occupancy"]
        assert len(occupancy["bins"]) == 16
        assert sum(row["count"] for row in occupancy["bins"]) == 16 * occupancy["total_count"]
    assert globals_seen > 0


def test_partitioned_occupancy_is_disjoint_and_matches_the_budget():
    assignment, _, _ = balanced_random_assignment(16, seed=10_000)
    memory = _memory(ALLOCATION_SPATIAL, budget=64, grid=4, assignment=assignment)
    for index, image in enumerate(_images(4, seed=1)):
        memory.update(image, task_id=0, image_index=index, global_step=index)
    occupancy = memory.occupancy()
    assert len(occupancy["bins"]) == 16
    assert sum(row["count"] for row in occupancy["bins"]) == occupancy["total_count"]
    assert occupancy["capacity_total"] == 64
    assert occupancy["bins_are_disjoint"] and occupancy["sum_bins_is_total"]


def test_single_group_spatial_matches_global_allocation_bitwise():
    """With one group the partitioned path must equal the global path."""
    images = _images(3, seed=2)
    budget = 40
    global_memory = _memory(ALLOCATION_GLOBAL, budget=budget)
    spatial_memory = _memory(
        ALLOCATION_SPATIAL, budget=budget, grid=1, assignment=spatial_assignment(1)
    )
    for index, image in enumerate(images):
        global_memory.update(image, task_id=0, image_index=index, global_step=index)
        spatial_memory.update(image, task_id=0, image_index=index, global_step=index)
    assert torch.equal(
        global_memory._banks[0].features, spatial_memory._banks[0].features
    )


def test_instrumented_global_scoring_matches_the_ad01_memory():
    """Swapping in a tracked bank must not move a single score."""
    images = _images(3, seed=3)
    queries = _images(2, seed=4)
    budget = 40

    ad01 = NormalSupportMemory(
        budget=budget, grid=4, allocation=ALLOCATION_GLOBAL, dim=DIM, device="cpu",
        fast=True,
    )
    ad02 = _memory(ALLOCATION_GLOBAL, budget=budget)
    for index, image in enumerate(images):
        ad01.update(image)
        ad02.update(image, task_id=0, image_index=index, global_step=index)

    reference_image, reference_pixel = ad01.score(torch.stack(queries), scoring="global")
    tracked_image, tracked_pixel = ad02.score(torch.stack(queries), scoring="global")
    assert torch.equal(reference_image, tracked_image)
    assert torch.equal(reference_pixel, tracked_pixel)


def test_task_origin_and_replacement_events_are_recorded_per_stage():
    budget = 24
    memory = _memory(ALLOCATION_GLOBAL, budget=budget)
    memory.begin_stage(0)
    for index, image in enumerate(_images(2, seed=5)):
        memory.update(image, task_id=0, image_index=index, global_step=index)
    first = memory.end_stage()
    # Insertions count both the deterministic fill and every later replacement,
    # so the invariant is fill + replacements - evictions == stored rows.
    assert first["insertions_total"] - first["evictions_total"] == memory.count
    assert memory.count == budget

    memory.begin_stage(1)
    step = 2
    for index, image in enumerate(_images(6, seed=6, offset=40.0)):
        memory.update(image, task_id=1, image_index=index, global_step=step)
        step += 1
    second = memory.end_stage()
    assert second["evicted_by_origin"].get(0, 0) > 0, "later tasks must evict earlier support"
    slots = memory.slot_counts_by_origin()
    assert slots.get(0, 0) < budget
    assert sum(slots.values()) == memory.count


def test_coverage_separates_own_support_from_any_support():
    budget = 16
    memory = _memory(ALLOCATION_GLOBAL, budget=budget)
    task0 = _images(4, seed=7)
    memory.begin_stage(0)
    for index, image in enumerate(task0):
        memory.update(image, task_id=0, image_index=index, global_step=index)
    memory.end_stage()
    queries = task0[0]
    d_any, d_own, own_slots = memory.coverage_raw(queries, task_id=0)
    assert own_slots == budget
    assert np.allclose(d_any, d_own)

    memory.begin_stage(1)
    step = len(task0)
    for index, image in enumerate(_images(8, seed=8, offset=60.0)):
        memory.update(image, task_id=1, image_index=index, global_step=step)
        step += 1
    memory.end_stage()
    d_any_later, d_own_later, own_slots_later = memory.coverage_raw(queries, task_id=0)
    assert own_slots_later <= budget
    assert np.all(d_own_later >= d_any_later - 1e-6)
    # A task with no surviving exemplar has no own-support distance at all.
    d_any_missing, d_own_missing, own_slots_missing = memory.coverage_raw(
        queries, task_id=99
    )
    assert own_slots_missing == 0
    assert np.all(np.isinf(d_own_missing))
    assert np.isfinite(d_any_missing).all()


def test_boundary_scores_depend_on_the_bank_state():
    budget = 24
    memory = _memory(ALLOCATION_GLOBAL, budget=budget)
    queries = _images(1, seed=9)
    memory.begin_stage(0)
    for index, image in enumerate(_images(2, seed=10)):
        memory.update(image, task_id=0, image_index=index, global_step=index)
    first, _ = memory.score(torch.stack(queries), scoring="global")
    memory.end_stage()
    memory.begin_stage(1)
    for index, image in enumerate(_images(3, seed=11, offset=30.0)):
        memory.update(image, task_id=1, image_index=index, global_step=index)
    second, _ = memory.score(torch.stack(queries), scoring="global")
    memory.end_stage()
    assert not torch.equal(first, second)


def test_transient_bytes_follow_the_partition_fineness():
    coarse = _memory(ALLOCATION_GLOBAL, budget=2500)
    fine = _memory(
        ALLOCATION_SPATIAL, budget=2500, grid=4, assignment=spatial_assignment(4)
    )
    assert coarse.transient_bytes()["closest_pair_matrix_bytes_total"] == 2500 ** 2 * 4
    assert fine.transient_bytes()["closest_pair_matrix_bytes_total"] < 2500 ** 2 * 4
    assert fine.transient_bytes()["n_banks"] == 16
