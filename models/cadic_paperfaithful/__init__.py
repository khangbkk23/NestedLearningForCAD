"""Paper-faithful CADIC implementation preserved alongside the legacy active path."""

from .cadic_patch_coreset import (
    CADICPatchCoresetConfig,
    CADICPatchCoresetV1,
    euclidean_distance_mm,
)
from .cadic_benchmark_adapter import CADICBenchmarkAdapterV1

__all__ = [
    "CADICPatchCoresetConfig",
    "CADICPatchCoresetV1",
    "CADICBenchmarkAdapterV1",
    "euclidean_distance_mm",
]
