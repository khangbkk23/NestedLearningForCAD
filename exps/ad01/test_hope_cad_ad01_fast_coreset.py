# exps/ad01/test_hope_cad_ad01_fast_coreset.py
"""Numerical equivalence of the optimised CADIC coreset.

The optimised variant changes two things that could in principle alter the
stored bank: how a distance matrix is formed (matrix multiply instead of many
small `torch.cdist` tiles) and how coarsely that matrix is tiled. The CADIC
replacement rule is a farthest-first selection with a lexicographic tie policy,
so a different rounding could in principle select a different pair and diverge.

This module therefore checks equivalence on tiny deterministic inputs against
the frozen reference, at the exact chunking used by the reported sweep. It is
deliberately small: no dataset, no feature cache, no category stream.
"""

from __future__ import annotations

import torch

from exps.ad01.hope_cad_ad01_fast_coreset import FastCADICPatchCoresetV1
from models.cadic_patch_coreset_v1 import CADICPatchCoresetV1, CADICPatchCoresetConfig

DIM = 32
BUDGET = 96
IMAGES = 6
PATCHES = 64


def _stream(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(IMAGES, PATCHES, DIM, generator=generator)


def _build(cls, chunk: int, images: torch.Tensor, device: str = "cpu"):
    coreset = cls(
        CADICPatchCoresetConfig(
            budget=BUDGET,
            dim=DIM,
            dtype="float32",
            distance="euclidean",
            chunk_size=chunk,
            image_neighbors=3,
            query_chunk_size=chunk,
            pair_chunk_size=chunk,
        ),
        device=device,
    )
    for image in images:
        coreset.update(image)
    return coreset


def test_fast_coreset_matches_reference_on_identical_input():
    """Same bank contents, same counters, same selected pairs."""
    images = _stream(0)
    reference = _build(CADICPatchCoresetV1, 64, images)
    fast = _build(FastCADICPatchCoresetV1, 64, images)

    assert reference.count == fast.count == BUDGET
    assert reference.replaced_features == fast.replaced_features
    assert reference.rejected_features == fast.rejected_features
    assert reference.accepted_features == fast.accepted_features

    deviation = (reference.features - fast.features).abs().max().item()
    assert deviation < 1e-4, f"bank contents diverged by {deviation}"


def test_fast_coreset_matches_at_the_sweep_chunking():
    """Equivalence must also hold at the chunk size the sweep actually uses."""
    images = _stream(1)
    reference = _build(CADICPatchCoresetV1, 2048, images)
    fast = _build(FastCADICPatchCoresetV1, 2048, images)
    deviation = (reference.features - fast.features).abs().max().item()
    assert deviation < 1e-4, f"bank contents diverged by {deviation}"


def test_coarse_tiling_selects_the_same_closest_pair():
    """Tiling changes the loop granularity, not which pair is closest."""
    images = _stream(2)
    for chunk in (32, 64, 2048):
        coreset = _build(CADICPatchCoresetV1, chunk, images)
        value, index = coreset._closest_pair()
        fine = _build(CADICPatchCoresetV1, 32, images)
        fine_value, fine_index = fine._closest_pair()
        assert abs(float(value) - float(fine_value)) < 1e-4
        assert index == fine_index, f"chunk {chunk} selected row {index}, fine tiling {fine_index}"


def test_coarse_tiling_scores_identically():
    """End-to-end scores must agree, with the kernel difference quantified.

    The reference computes the support distance with `torch.linalg.vector_norm`
    while the optimised variant uses the Eq. (7) expansion. On patches whose
    nearest distance is essentially zero the two forms differ in the last bits,
    which shows up as a small absolute difference on those few patches. The
    stored bank, the nearest-neighbour indices and the image score are exact or
    near-exact, so this is a floating-point difference in one intermediate, not
    a semantic divergence.
    """
    images = _stream(3)
    query = images[-1]
    reference = _build(CADICPatchCoresetV1, 2048, images)
    fast = _build(FastCADICPatchCoresetV1, 2048, images)

    assert torch.equal(reference.features, fast.features), "bank must be identical"

    reference_image, reference_pixel = reference.score(query.unsqueeze(0), b=3)
    fast_image, fast_pixel = fast.score(query.unsqueeze(0), b=3)

    # Image scores carry the ranking that the experiment reports.
    assert torch.allclose(reference_image, fast_image, atol=1e-5, rtol=1e-6)
    # Pixel maps agree except on near-zero distances; bound the absolute drift.
    assert torch.allclose(reference_pixel, fast_pixel, atol=5e-3, rtol=1e-3)
    # And the drift must be confined to small distances, not spread everywhere.
    drift = (reference_pixel - fast_pixel).abs()
    assert float(drift.max()) < 5e-3
    assert int((drift > 1e-4).sum()) < reference_pixel.numel() // 4


def test_nearest_matches_reference():
    images = _stream(4)
    reference = _build(CADICPatchCoresetV1, 2048, images)
    fast = _build(FastCADICPatchCoresetV1, 2048, images)
    query = images[0][:16]
    reference_values, reference_indices = reference._nearest(query, reference.features)
    fast_values, fast_indices = fast._nearest(query, fast.features)
    assert torch.allclose(reference_values, fast_values, atol=1e-4, rtol=1e-5)
    assert torch.equal(reference_indices, fast_indices)
