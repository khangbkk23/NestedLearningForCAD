"""Spec3§5.2 sampling primitives and explicitly clocked MemoryManager wrapper."""

from __future__ import annotations

from dataclasses import replace

from nested_replay_cad.core.contracts import (
    PcgState,
    RecordID,
    Reservations,
    SamplingRngState,
)
from nested_replay_cad.core.errors import SchemaError
from nested_replay_cad.core.memory.rng import (
    ordered_sample,
    randbelow,
    validate_pcg_state,
    validate_record_id,
)


def sample_reservations(
    reservoir: tuple[RecordID, ...], fifo: tuple[RecordID, ...], rng: SamplingRngState
) -> tuple[Reservations, SamplingRngState]:
    """Use pre-event buffers; caller owns arrival legality and atomic commit.

    No arm input: ER/current-only/A0 reserve the same IDs and consume the same
    draws. This function does not grant future access or active replay rights.
    """
    if not isinstance(rng, SamplingRngState):
        raise SchemaError("invalid_sampling_rng_state")
    for pcg in (rng.admission, rng.recent, rng.historical, rng.warmup):
        validate_pcg_state(pcg)
    for ids, capacity in ((reservoir, 24), (fifo, 8)):
        if not isinstance(ids, tuple) or len(ids) > capacity:
            raise SchemaError("invalid_buffer_ids")
        for record_id in ids:
            validate_record_id(record_id)
        if len(set(ids)) != len(ids):
            raise SchemaError("duplicate_buffer_id")
    recent, recent_rng = ordered_sample(rng.recent, fifo, min(2, len(fifo)))
    fifo_ids = set(fifo)
    old_pool = tuple(i for i in reservoir if i not in fifo_ids)
    eligible = len(fifo) >= 2 and len(old_pool) >= 6
    next_rng = replace(rng, recent=recent_rng)
    if eligible:
        six, historical_rng = ordered_sample(rng.historical, old_pool, 6)
        history, replay = six[:2], six[2:]
        next_rng = replace(next_rng, historical=historical_rng)
    else:
        history = ()
        recent_ids = set(recent)
        pool = tuple(i for i in reservoir if i not in recent_ids)
        replay, warmup_rng = ordered_sample(rng.warmup, pool, min(4, len(pool)))
        next_rng = replace(next_rng, warmup=warmup_rng)
    return Reservations(replay, recent, history, eligible), next_rng


def admission_slot(state: PcgState, arrival: int) -> tuple[int | None, PcgState]:
    """Algorithm R proposal after event computation; no buffer/tensor mutation.

    Returns a zero-based slot (append for arrival≤24), or None for rejection.
    Caller must verify arrival==committed_count+1 before using this proposal.
    """
    validate_pcg_state(state)
    if type(arrival) is not int or not 1 <= arrival <= 1 << 64:
        raise SchemaError("invalid_admission_arrival")
    if arrival <= 24:
        return arrival - 1, state
    index, next_rng = randbelow(state, arrival)
    return (index if index < 24 else None), next_rng
