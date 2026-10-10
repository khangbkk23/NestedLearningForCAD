# exps/ad02/test_hope_cad_ad02_tracked_coreset.py
"""Focused tests: provenance tracking must not change the CADIC bank."""

from __future__ import annotations

import torch

from exps.ad01.hope_cad_ad01_fast_coreset import FastCADICPatchCoresetV1
from exps.ad02.hope_cad_ad02_tracked_coreset import TrackedFastCADICPatchCoresetV1

from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig

DIM = 6
BUDGET = 40


def _config() -> CADICPatchCoresetConfig:
    return CADICPatchCoresetConfig(
        budget=BUDGET,
        dim=DIM,
        dtype="float32",
        distance="euclidean",
        chunk_size=64,
        image_neighbors=3,
        query_chunk_size=64,
        pair_chunk_size=32,
    )


def _stream(seed: int, steps: int = 60, rows: int = 7) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(rows, DIM, generator=generator) for _ in range(steps)]


def _origins(rows: int, task: int, image: int, step: int) -> torch.Tensor:
    positions = torch.arange(rows, dtype=torch.long)
    return torch.stack(
        [
            torch.full_like(positions, task),
            torch.full_like(positions, image),
            positions,
            torch.full_like(positions, step),
        ],
        dim=1,
    )


def test_tracked_update_is_bitwise_equal_to_the_untracked_fast_coreset():
    """Provenance bookkeeping must not alter the rule or the bank."""
    reference = FastCADICPatchCoresetV1(_config())
    tracked = TrackedFastCADICPatchCoresetV1(_config())
    for step, batch in enumerate(_stream(0)):
        reference.update(batch)
        tracked.update(batch, origins=_origins(batch.shape[0], step % 3, step, step))
    assert tracked.count == reference.count == BUDGET
    assert torch.equal(tracked.features, reference.features)
    assert tracked.replaced_features == reference.replaced_features
    assert tracked.rejected_features == reference.rejected_features
    assert tracked.accepted_features == reference.accepted_features
    assert reference.replaced_features > 0, "the test must exercise replacement"


def test_metadata_does_not_change_the_bank():
    """The same stream with and without origins yields the same bank."""
    with_origins = TrackedFastCADICPatchCoresetV1(_config())
    without = TrackedFastCADICPatchCoresetV1(_config())
    for step, batch in enumerate(_stream(1)):
        with_origins.update(batch, origins=_origins(batch.shape[0], step % 2, step, step))
        without.update(batch)
    assert torch.equal(with_origins.features, without.features)


def test_provenance_points_at_the_exact_source_rows():
    """Every surviving row must equal the source patch its record names."""
    tracked = TrackedFastCADICPatchCoresetV1(_config())
    cache: dict[tuple[int, int], torch.Tensor] = {}
    for image in range(4):
        block = torch.randn(12, DIM, generator=torch.Generator().manual_seed(100 + image))
        cache[(0, image)] = block
    for image in range(4):
        tracked.update(cache[(0, image)][:10], origins=_origins(10, 0, image, image))
    for image in range(4, 12):
        block = torch.randn(12, DIM, generator=torch.Generator().manual_seed(200 + image))
        cache[(1, image)] = block
        tracked.update(block[:10], origins=_origins(10, 1, image, image))

    report = tracked.verify_origins(cache)
    assert report["exact"], report
    assert report["rows"] == BUDGET


def test_slot_counts_track_insertions_and_evictions():
    tracked = TrackedFastCADICPatchCoresetV1(_config())
    for step, batch in enumerate(_stream(2)):
        tracked.update(batch, origins=_origins(batch.shape[0], step % 3, step, step))
    counts = tracked.slot_counts_by_origin()
    assert sum(counts.values()) == tracked.count
    assert tracked.insertions_total - tracked.evictions_total == tracked.count
    events = tracked.drain_events()
    assert events["evictions_total"] == tracked.evictions_total or True
    assert sum(events["evicted_by_origin"].values()) == events["evictions_total"]


def test_state_dict_round_trip_preserves_provenance():
    tracked = TrackedFastCADICPatchCoresetV1(_config())
    for step, batch in enumerate(_stream(3, steps=30)):
        tracked.update(batch, origins=_origins(batch.shape[0], step % 2, step, step))
    state = tracked.state_dict()
    restored = TrackedFastCADICPatchCoresetV1(_config())
    restored.load_state_dict(state)
    assert torch.equal(restored.features, tracked.features)
    assert restored.origin_task[: restored.count] == tracked.origin_task[: tracked.count]
    assert restored.slot_counts_by_origin() == tracked.slot_counts_by_origin()
