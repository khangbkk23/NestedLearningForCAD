"""Spec3§5.2 raw-PCG64 sampling over immutable snapshots.

NumPy is loaded only when drawing/seeding; no global NumPy/Torch RNG is used.
"""

from __future__ import annotations

import hashlib
import re

from nested_replay_cad.core.contracts import PcgState, RecordID, SamplingRngState
from nested_replay_cad.core.errors import SchemaError, ScopeBlocked
from nested_replay_cad.core.identity import canonical_json

STREAMS = ("reservoir", "recent_query", "historical_split", "warmup_support")
UINT64 = 1 << 64
UINT128 = 1 << 128


def _integer(value: int, low: int, high: int | None, reason: str) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise SchemaError(reason)
    return value


def validate_seed(seed: int) -> int:
    return _integer(seed, 0, (1 << 63) - 1, "invalid_run_seed")


def _decimal(value: str, reason: str) -> int:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"(0|[1-9][0-9]{0,38})", value, flags=re.ASCII) is None
    ):
        raise SchemaError(reason)
    return _integer(int(value), 0, UINT128 - 1, reason)


def validate_pcg_state(state: PcgState) -> None:
    if not isinstance(state, PcgState):
        raise SchemaError("invalid_pcg_state")
    _decimal(state.state, "invalid_pcg_internal_state")
    if _decimal(state.increment, "invalid_pcg_increment") % 2 != 1:
        raise SchemaError("pcg_increment_must_be_odd")
    _integer(state.has_uint32, 0, 1, "invalid_pcg_has_uint32")
    _integer(state.uinteger, 0, (1 << 32) - 1, "invalid_pcg_uinteger")
    _integer(state.raw_draws, 0, None, "invalid_pcg_raw_draw_count")


def _pcg64_class() -> type[PCG64]:
    import numpy as np

    if np.__version__ != "2.4.5":
        raise ScopeBlocked("rng_requires_reference_numpy_2.4.5")
    return np.random.PCG64


def _snapshot(generator: PCG64, raw_draws: int) -> PcgState:
    value = generator.state
    return PcgState(
        str(value["state"]["state"]),
        str(value["state"]["inc"]),
        int(value["has_uint32"]),
        int(value["uinteger"]),
        raw_draws,
    )


def _restore(state: PcgState) -> PCG64:
    validate_pcg_state(state)
    generator = _pcg64_class()(0)
    generator.state = {
        "bit_generator": "PCG64",
        "state": {"state": int(state.state), "inc": int(state.increment)},
        "has_uint32": state.has_uint32,
        "uinteger": state.uinteger,
    }
    return generator


def seed_stream(run_seed: int, stream_name: str) -> PcgState:
    validate_seed(run_seed)
    if stream_name not in STREAMS:
        raise SchemaError("unknown_replay_rng_stream")
    payload = canonical_json(["A0-R1", run_seed, stream_name]).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return _snapshot(_pcg64_class()(seed), 0)


def seed_sampling(run_seed: int) -> SamplingRngState:
    return SamplingRngState(*(seed_stream(run_seed, name) for name in STREAMS))


def validate_record_id(value: RecordID) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value, flags=re.ASCII) is None
    ):
        raise SchemaError("invalid_record_id")


def randbelow(state: PcgState, n: int) -> tuple[int, PcgState]:
    """Unbiased rejection with ≥1 raw draw, including n=1 and n=2**64."""
    _integer(n, 1, UINT64, "invalid_randbelow_bound")
    generator = _restore(state)
    limit = (UINT64 // n) * n
    draws = state.raw_draws
    while True:
        value = int(generator.random_raw())
        draws += 1
        if value < limit:
            return value % n, _snapshot(generator, draws)


def ordered_sample(
    state: PcgState, candidates: tuple[RecordID, ...], k: int
) -> tuple[tuple[RecordID, ...], PcgState]:
    """Ordered partial Fisher–Yates. No sort; final self-swap consumes a draw."""
    validate_pcg_state(state)
    if not isinstance(candidates, tuple):
        raise SchemaError("sampling_candidates_must_be_tuple")
    for record_id in candidates:
        validate_record_id(record_id)
    if len(set(candidates)) != len(candidates):
        raise SchemaError("duplicate_sampling_candidate")
    _integer(k, 0, len(candidates), "invalid_sample_size")
    values = list(candidates)
    next_state = state
    for j in range(k):
        offset, next_state = randbelow(next_state, len(values) - j)
        values[j], values[j + offset] = values[j + offset], values[j]
    return tuple(values[:k]), next_state
