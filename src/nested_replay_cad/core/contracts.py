"""Spec3§3 DTOs. Tensor invariants require Phase4B validators, not annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NewType, TypeAlias

RecordID = NewType("RecordID", str)
ArrivalIndex = NewType("ArrivalIndex", int)
NamedTensors: TypeAlias = Mapping[str, "Tensor"]


@dataclass(frozen=True)
class RecordMeta:
    opaque_id: RecordID
    arrival: ArrivalIndex
    raw_sha256: str


@dataclass(frozen=True)
class PhiState:
    parameters: NamedTensors  # W[16,4],b[16],W[1,16],b[1]: 97 scalars


@dataclass(frozen=True)
class AdamState:
    m: NamedTensors
    v: NamedTensors
    step: int


@dataclass(frozen=True)
class PcgState:
    state: str  # canonical decimal uint128, validated by memory.rng
    increment: str
    has_uint32: int
    uinteger: int
    raw_draws: int


@dataclass(frozen=True)
class SamplingRngState:
    """Replay streams only; no torch/model state or global RNG ownership."""

    admission: PcgState
    recent: PcgState
    historical: PcgState
    warmup: PcgState


@dataclass(frozen=True)
class Reservations:
    """Pre-event reserved IDs; active support/slow clock set by event planner."""

    R_reserved: tuple[RecordID, ...]
    Q_recent: tuple[RecordID, ...]
    Q_history: tuple[RecordID, ...]
    eligible: bool
