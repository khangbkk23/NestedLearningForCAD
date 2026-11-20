"""B324 helpers: lossless uint8 input replay, auxiliary RNG streams, update schedules.

Storage: a replay sample is the uint8 RGB image after the encoder's own resize
(3 x 392 x 392, 460,992 bytes) instead of its frozen grouped feature (4,847,616
bytes). ``normalize_uint8`` rebuilds exactly the tensor ``encoder.preprocess``
returns, so a stored sample re-encodes to the feature the frozen encoder gives.

Auxiliary streams use their own seed derivation, so the four registered replay
streams (admission, recent, historical, warmup) are never touched.
"""
import hashlib
import json

from nested_replay_cad.core.identity import canonical_json
from nested_replay_cad.core.memory.rng import _pcg64_class, _snapshot, ordered_sample, validate_seed

# Copies of the encoder constants (model/encoder.py imports torch at module load);
# a torch test asserts they equal encoder.MEAN, encoder.STD and encoder.SIZE.
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIZE = 392

FORMAT = "uint8_chw_392"
BYTES_PER_SAMPLE = 3 * SIZE * SIZE
AUX_STREAMS = ("consolidation", "stream_order")


def load_uint8(path):
    """The image as the encoder sees it before normalisation: RGB, bilinear 392 x 392, uint8 CHW."""
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    array = np.array(im, dtype=np.uint8, copy=True)
    if array.shape != (SIZE, SIZE, 3):
        raise ValueError("Unexpected image shape after resize")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def normalize_uint8(value):
    """Same operations, in the same order, as ``encoder.preprocess`` after the resize."""
    import numpy as np
    import torch

    if value.dtype != torch.uint8 or tuple(value.shape) != (3, SIZE, SIZE):
        raise ValueError("Stored replay sample must be uint8 (3, 392, 392)")
    array = value.permute(1, 2, 0).numpy()
    x = torch.from_numpy(np.asarray(array, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


def aux_stream(run_seed, name):
    validate_seed(run_seed)
    if name not in AUX_STREAMS:
        raise ValueError("Unknown B324 auxiliary stream")
    payload = canonical_json(["B324-AUX", run_seed, name]).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return _snapshot(_pcg64_class()(seed), 0)


def uniform_sample(state, pool, k):
    """Uniform draw without replacement from ``pool`` (a tuple of record IDs)."""
    return ordered_sample(state, tuple(pool), min(k, len(pool)))


def shuffled_records(records, run_seed):
    """A deterministic permutation of the stream, renumbered 1..n (arm ``iid``)."""
    ids = tuple(row["opaque_id"] for row in records)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate stream IDs")
    order, _ = ordered_sample(aux_stream(run_seed, "stream_order"), ids, len(ids))
    by_id = {row["opaque_id"]: row for row in records}
    return [dict(by_id[identity], arrival=index) for index, identity in enumerate(order, 1)]


def validate_schedule(schedule):
    mode = schedule.get("mode")
    if mode == "single":
        return schedule
    if mode == "flat" and schedule.get("extra_per_arrival", 0) >= 1 and schedule.get("samples", 0) >= 1:
        return schedule
    if mode == "nested" and all(schedule.get(key, 0) >= 1 for key in ("every", "updates", "samples", "final_arrival")) \
            and schedule.get("tail", -1) >= 0 and schedule["final_arrival"] % schedule["every"] != 0:
        return schedule
    raise ValueError("Invalid B324 update schedule")


def extra_updates(schedule, arrival):
    """Optimizer steps added after the fast step of ``arrival``."""
    mode = schedule["mode"]
    if mode == "single":
        return 0
    if mode == "flat":
        return schedule["extra_per_arrival"]
    if arrival == schedule["final_arrival"]:
        return schedule["tail"]
    return schedule["updates"] if arrival % schedule["every"] == 0 else 0


def expected_updates(schedule, committed):
    """Total optimizer steps after ``committed`` arrivals (one fast step each plus extras)."""
    return committed + sum(extra_updates(schedule, arrival) for arrival in range(1, committed + 1))


def memory_digest(reservoir, fifo):
    return hashlib.sha256(json.dumps([list(reservoir), list(fifo)]).encode()).hexdigest()


def ids_digest(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def store_digest(store):
    value = hashlib.sha256()
    for key in sorted(store):
        value.update(key.encode())
        value.update(store[key].contiguous().numpy().tobytes())
    return value.hexdigest()
