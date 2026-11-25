"""Continual State: union exemplar keys (fp16), cumulative residual moment S (fp64, canonical per D6), count n.

Serialization/checkpoint publication (§10.2 of the task spec) is NOT part of M1 and is not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ContinualState:
    dim: int
    keys: tuple = ()                    # per-task fp16 CPU tensors, task-major append order
    selected: tuple = ()                # per-task selected FIT row indices (provenance)
    S: torch.Tensor | None = None       # fp64 [d, d], summed in task order
    n: int = 0                          # residual count = sum(|X_t| - k_t)
    tasks: tuple = field(default_factory=tuple)

    @property
    def stage(self):
        return len(self.keys) - 1

    def union_keys(self):
        return torch.cat(self.keys)

    def ledger(self):
        """Bytes (owner amendment 2). Serialized checkpoint bytes are measured only once P5 exists."""
        K, d = sum(len(k) for k in self.keys), self.dim
        keys = K * 2 * d
        return {'n_keys': K,
                'historical_registered_ledger': keys + d * d * 4 + 8,
                'actual_persistent_tensor_bytes': keys + d * d * 8 + 8,
                'inference_only_state': keys + d * d * 4,
                'runtime_cache_keys32_mw32_norms': K * d * 8 + K * 4,
                'provenance_selected_indices': 8 * K,
                'serialized_checkpoint_bytes': None}


# ============================================================ P5: canonical checkpoint publication (spec §10.2)
import hashlib as _hashlib
import json as _json
import os as _os
import re as _re
import uuid as _uuid
from pathlib import Path as _Path

SCHEMA = 'nanm.ckpt/1'
ENCODER_IDENTITY = {'arch': 'dinov2_vitb14_reg4 (exps/dinomaly/encoder.py, verbatim in nanm/encoder.py)',
                    'checkpoint_sha256': '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71',
                    'blocks': [6, 7, 8, 9], 'image_size': 392, 'dtype': 'float32', 'batch': 8}
_GEN = _re.compile(r'^gen_\d{4}_[0-9a-f]{12}$')
PAYLOAD = 'state.safetensors'


class CheckpointError(RuntimeError):
    pass


def _fsync_dir(p):
    fd = _os.open(p, _os.O_RDONLY)
    try:
        _os.fsync(fd)
    finally:
        _os.close(fd)


def _write(p, data):
    with open(p, 'wb') as f:
        f.write(data); f.flush(); _os.fsync(f.fileno())


def _sha(b):
    return _hashlib.sha256(b).hexdigest()


def publish(root, state, meta, _fault=None):
    """Write a new immutable generation. Commit point = atomic rename of the complete, fsynced temp directory
    (same filesystem); LATEST is then replaced atomically. Readers never see a temp directory."""
    from safetensors.torch import save as st_save
    root = _Path(root); root.mkdir(parents=True, exist_ok=True)
    gen = f'gen_{state.stage + 1:04d}_{_uuid.uuid4().hex[:12]}'
    tmp = root / f'.tmp_{gen}'
    tmp.mkdir()
    off = [0]
    for k in state.keys:
        off.append(off[-1] + len(k))
    tensors = {'keys': state.union_keys().contiguous(), 'offsets': torch.tensor(off, dtype=torch.int64),
               'S': state.S.contiguous(), 'n': torch.tensor([state.n], dtype=torch.int64)}
    tensors.update({f'selected_{t:04d}': s.to(torch.int64).contiguous() for t, s in enumerate(state.selected)})
    blob = st_save(tensors)
    _write(tmp / PAYLOAD, blob)
    if _fault == 'after_payload':
        raise CheckpointError('injected fault after payload write')
    prev = (root / 'LATEST').read_text().strip() if (root / 'LATEST').exists() else None
    man = dict(meta, schema=SCHEMA, generation=gen, parent=prev, completed_tasks=len(state.keys),
               tasks=list(state.tasks), n=state.n, files={PAYLOAD: {'sha256': _sha(blob), 'bytes': len(blob)}})
    _write(tmp / 'manifest.json', (_json.dumps(man, indent=1, sort_keys=True, default=str) + '\n').encode())
    if _fault == 'after_manifest':
        raise CheckpointError('injected fault after manifest write')
    _fsync_dir(tmp)
    _os.rename(tmp, root / gen)                                       # commit point
    _fsync_dir(root)
    if _fault == 'after_rename':
        raise CheckpointError('injected fault after rename, before LATEST')
    _write(root / '.LATEST.tmp', gen.encode())
    _os.replace(root / '.LATEST.tmp', root / 'LATEST')
    _fsync_dir(root)
    return gen


def list_generations(root):
    """Committed generations only (temp directories are never listed)."""
    return sorted(p.name for p in _Path(root).iterdir() if p.is_dir() and _GEN.match(p.name))


def read(root, generation=None, contract_sha256=None, encoder_sha256=None):
    from safetensors.torch import load as st_load
    root = _Path(root)
    if generation is None:
        if not (root / 'LATEST').exists():
            raise CheckpointError('no LATEST pointer')
        generation = (root / 'LATEST').read_text().strip()
    if not _GEN.match(generation or ''):
        raise CheckpointError(f'not a committed generation name: {generation!r}')
    d = root / generation
    if not d.is_dir():
        raise CheckpointError(f'missing generation {generation}')
    try:
        man = _json.loads((d / 'manifest.json').read_text())
    except FileNotFoundError:
        raise CheckpointError('missing manifest') from None
    except ValueError as e:
        raise CheckpointError(f'unreadable manifest: {e}') from None
    if man.get('schema') != SCHEMA:
        raise CheckpointError(f'unsupported schema {man.get("schema")!r}')
    if man.get('generation') != generation:
        raise CheckpointError('manifest generation mismatch')
    present = {p.name for p in d.iterdir()}
    if present != {'manifest.json'} | set(man['files']):
        raise CheckpointError(f'unexpected file set {sorted(present)}')
    blobs = {}
    for name, v in man['files'].items():
        b = (d / name).read_bytes()
        if len(b) != v['bytes'] or _sha(b) != v['sha256']:
            raise CheckpointError(f'checksum/size mismatch: {name}')
        blobs[name] = b
    if contract_sha256 is not None and man['contract']['sha256'] != contract_sha256:
        raise CheckpointError('scientific contract mismatch')
    if encoder_sha256 is not None and man['encoder']['checkpoint_sha256'] != encoder_sha256:
        raise CheckpointError('encoder checkpoint identity mismatch')
    try:
        t = st_load(blobs[PAYLOAD])
    except Exception as e:                                          # noqa: BLE001
        raise CheckpointError(f'unreadable payload: {e}') from None
    T = man['completed_tasks']
    keys, off, S, n = t['keys'], t['offsets'].tolist(), t['S'], int(t['n'][0])
    ok = (keys.dtype == torch.float16 and keys.ndim == 2 and keys.shape[1] == man['contract']['fields']['dim']
          and S.dtype == torch.float64 and S.shape == (keys.shape[1],) * 2 and len(off) == T + 1 and off[0] == 0
          and off[-1] == len(keys) and all(a < b for a, b in zip(off, off[1:])) and n == man['n'] and T >= 1
          and len(man['tasks']) == T and all(f'selected_{i:04d}' in t for i in range(T))
          and bool(torch.isfinite(keys).all()) and bool(torch.isfinite(S).all()))
    if not ok:
        raise CheckpointError('invalid state tensors')
    state = ContinualState(dim=keys.shape[1], keys=tuple(keys[off[i]:off[i + 1]].clone() for i in range(T)),
                           selected=tuple(t[f'selected_{i:04d}'] for i in range(T)), S=S, n=n,
                           tasks=tuple(man['tasks']))
    return state, man
