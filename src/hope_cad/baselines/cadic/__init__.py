"""Explicit CADIC bank implementations; no claim of exact paper reproduction."""

from hope_cad.baselines.cadic.bank import (
    CADICPatchCoresetConfig,
    CADICPatchCoresetV1,
    euclidean_distance_mm,
)
from hope_cad.baselines.cadic.adapter import CADICBenchmarkAdapterV1

__all__ = [
    "CADICPatchCoresetConfig",
    "CADICPatchCoresetV1",
    "CADICBenchmarkAdapterV1",
    "euclidean_distance_mm",
]
