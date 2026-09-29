# models/hope_cad/__init__.py
# models/hope_cad/__init__.py
"""Standalone HOPE representation-core namespace."""

from .continuum_memory import CMSCommitResult, CMSInspection, ContinuumMemorySystem
from .hope_block import HOPECommitResult, HopeBlock
from .self_modifying_titans import SMTProjectionResult, SelfModifyingTitans

__all__ = [
    "CMSCommitResult",
    "CMSInspection",
    "ContinuumMemorySystem",
    "HOPECommitResult",
    "HopeBlock",
    "SMTProjectionResult",
    "SelfModifyingTitans",
]
