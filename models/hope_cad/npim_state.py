# models/hope_cad/npim_state.py
"""Deterministic state hashing, byte accounting and atomic persistence for NPIM-CAD.

The helpers are intentionally generic over a nested ``state`` mapping whose
tensor leaves are the persistent detector state.  They extend, but do not
modify, :mod:`models.hope_cad.state`.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from models.hope_cad.state import persistent_tensor_bytes, validate_finite_state


def _walk(value: Any, path: str, tensor_bytes: Dict[str, int], scalars: Dict[str, Any]) -> None:
    if isinstance(value, torch.Tensor):
        tensor_bytes[path] = int(value.numel() * value.element_size())
        return
    if isinstance(value, Mapping):
        for key in value:
            _walk(value[key], f"{path}.{key}" if path else str(key), tensor_bytes, scalars)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]", tensor_bytes, scalars)
        return
    scalars[path] = value


def tensor_byte_breakdown(state: Mapping[str, Any]) -> Dict[str, int]:
    tensor_bytes: Dict[str, int] = {}
    scalars: Dict[str, Any] = {}
    _walk(state, "", tensor_bytes, scalars)
    return tensor_bytes


def non_tensor_metadata_bytes(state: Mapping[str, Any]) -> int:
    tensor_bytes: Dict[str, int] = {}
    scalars: Dict[str, Any] = {}
    _walk(state, "", tensor_bytes, scalars)
    payload = json.dumps({k: _jsonable(v) for k, v in scalars.items()}, sort_keys=True, separators=(",", ":"))
    return len(payload.encode("utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def deterministic_state_hash(state: Mapping[str, Any]) -> str:
    """SHA-256 over sorted tensor bytes and scalar metadata (structure-stable)."""

    tensor_bytes: Dict[str, int] = {}
    scalars: Dict[str, Any] = {}
    _walk(state, "", tensor_bytes, scalars)
    digest = hashlib.sha256()
    for path in sorted(tensor_bytes):
        digest.update(path.encode("utf-8"))
        digest.update(b"|")
        value = _lookup(state, path)
        payload = value.detach().to("cpu", copy=True).contiguous()
        digest.update(f"{tuple(payload.shape)}|{payload.dtype}|".encode("utf-8"))
        digest.update(payload.numpy().tobytes())
        digest.update(b";")
    digest.update(b"scalars|")
    digest.update(
        json.dumps({k: _jsonable(v) for k, v in scalars.items()}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _lookup(state: Mapping[str, Any], path: str) -> Any:
    """Resolve the ``a.b[0].c`` path emitted by :func:`_walk`."""

    current: Any = state
    for token in path.replace("[", ".").replace("]", "").split("."):
        if token == "":
            continue
        if isinstance(current, Mapping):
            current = current[token]
        elif isinstance(current, (list, tuple)):
            current = current[int(token)]
        else:  # pragma: no cover - defensive
            raise KeyError(f"cannot resolve {path}")
    return current


def ledger(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Exact detector byte ledger.

    ``tensor_subtotal`` counts every persistent tensor (weights, pending
    sufficient statistics, projection, calibration buffers).  ``metadata_bytes``
    counts the UTF-8 JSON of all scalar counters/configuration.  Serialized
    file size is measured separately by :func:`save_state_atomic` because
    pickle/framing overhead is not logical state.
    """

    validate_finite_state(state)
    breakdown = tensor_byte_breakdown(state)
    subtotal = persistent_tensor_bytes(state)
    metadata = non_tensor_metadata_bytes(state)
    grouped: Dict[str, int] = {}
    for path, size in breakdown.items():
        group = path.split(".")[0].split("[")[0] if path else "<root>"
        grouped[group] = grouped.get(group, 0) + int(size)
    return {
        "tensor_bytes_by_path": breakdown,
        "tensor_bytes_by_group": grouped,
        "tensor_subtotal_bytes": int(subtotal),
        "metadata_bytes": int(metadata),
        "total_persistent_detector_bytes": int(subtotal + metadata),
        "shared_frozen_encoder_checkpoint_bytes": 343_282_995,
        "note": "shared encoder checkpoint is reported separately and excluded from detector totals",
    }


def save_state_atomic(state: Mapping[str, Any], path: str | Path) -> Dict[str, Any]:
    """Write a new immutable state file via temporary-file promotion."""

    validate_finite_state(state)
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable state: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"incomplete state write exists: {temporary}")
    torch.save(dict(state), temporary)
    os.replace(temporary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "path": str(target),
        "bytes": int(target.stat().st_size),
        "sha256": digest,
        "state_hash": deterministic_state_hash(state),
        "ledger": ledger(state),
    }


def load_state_file(path: str | Path) -> Dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("state file does not contain a mapping")
    validate_finite_state(state)
    return state


def serialized_state_bytes(state: Mapping[str, Any]) -> int:
    """Serialized size of an in-memory state (for accounting before writing)."""

    buffer = io.BytesIO()
    torch.save(dict(state), buffer)
    return int(buffer.tell())


def rng_fingerprint() -> str:
    """Hash of torch CPU RNG plus NumPy RNG state (read-only check helper)."""

    torch_state = torch.get_rng_state().numpy().tobytes()
    numpy_state = np.random.get_state()
    numpy_payload = (
        numpy_state[0].encode("utf-8")
        + numpy_state[1].tobytes()
        + int(numpy_state[2]).to_bytes(8, "little", signed=False)
        + int(numpy_state[3]).to_bytes(8, "little", signed=False)
        + float(numpy_state[4]).hex().encode("utf-8")
    )
    return hashlib.sha256(torch_state + numpy_payload).hexdigest()
