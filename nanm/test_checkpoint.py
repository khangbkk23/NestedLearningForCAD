"""nanm.checkpoint tests (NANM-RECOVERY-01 Phase 4), run per filesystem root.

Roots: NANM_EXT4_TESTDIR (local Linux ext4) and NANM_DRVFS_TESTDIR (WSL DrvFS / NTFS); each skips when unset.
Separated properties: serialization, round trip, checksum validation, publication atomicity (observational).
Payloads are a few KB; nothing large is written.
"""
import os
import threading
import uuid
from pathlib import Path

import pytest
import torch

from nanm import checkpoint as C

ROOTS = {'ext4': os.environ.get('NANM_EXT4_TESTDIR'), 'drvfs': os.environ.get('NANM_DRVFS_TESTDIR')}


@pytest.fixture(params=sorted(ROOTS))
def root(request):
    base = ROOTS[request.param]
    if not base:
        pytest.skip(f'NANM_{request.param.upper()}_TESTDIR not set')
    d = Path(base) / f'nanm_ckpt_{uuid.uuid4().hex[:8]}'
    d.mkdir(parents=True)
    yield d
    for p in d.iterdir():                       # only the files this test created; no recursive delete
        p.unlink()
    d.rmdir()


def payload(v):
    g = torch.Generator().manual_seed(v)
    return {'keys': torch.randn(16, 768, generator=g).half(), 'S': torch.randn(8, 8, generator=g).double(),
            'kc': torch.randperm(50, generator=g), 'mask': torch.rand(5, generator=g) > 0.5,
            'meta': {'version': v, 'tasks': ['a', 'b'], 'n': 123}}


def same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k])
        else:
            assert a[k] == b[k]


def test_serialization_correctness(root):
    p = root / 'state.pt'
    rec = C.publish(payload(1), p)
    same(C.load(p), payload(1))
    assert isinstance(rec.dir_fsynced, bool)


def test_round_trip_overwrite_and_sidecar(root):
    p = root / 'state.pt'
    r1 = C.publish(payload(1), p)
    r2 = C.publish(payload(2), p)
    same(C.load(p), payload(2))
    assert r1.sha256 != r2.sha256
    assert C.expected_sha256(p) == r2.sha256 == C.sha256_file(p)
    assert r2.nbytes == p.stat().st_size
    assert sorted(x.name for x in root.iterdir()) == ['state.pt', 'state.pt.sha256']   # no temp leftovers


def test_checksum_validation(root):
    p = root / 'state.pt'
    C.publish(payload(3), p)
    b = bytearray(p.read_bytes()); b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        C.load(p)
    (root / 'state.pt.sha256').write_text('deadbeef  state.pt\n')
    with pytest.raises(RuntimeError, match='malformed'):
        C.load(p)
    (root / 'state.pt.sha256').unlink()
    with pytest.raises(FileNotFoundError):
        C.load(p)


def test_failed_replace_leaves_previous_version(root, monkeypatch):
    p = root / 'state.pt'
    C.publish(payload(4), p)
    before = p.read_bytes()

    def boom(*a, **k):
        raise OSError('injected failure before rename')
    monkeypatch.setattr(C.os, 'replace', boom)
    with pytest.raises(OSError, match='injected'):
        C.publish(payload(5), p)
    monkeypatch.undo()
    assert p.read_bytes() == before
    same(C.load(p), payload(4))
    assert sorted(x.name for x in root.iterdir()) == ['state.pt', 'state.pt.sha256']


def test_concurrent_reader_never_sees_partial_payload(root):
    """Observational: a reader looping during 25 publishes sees only complete published payloads."""
    p = root / 'state.pt'
    digests = {C.publish(payload(0), p).sha256}
    seen, errors, stop = [], [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                seen.append(C.sha256_bytes(p.read_bytes()))
            except FileNotFoundError as e:       # the target must never disappear
                errors.append(e)

    t = threading.Thread(target=reader); t.start()
    try:
        for v in range(1, 26):
            digests.add(C.publish(payload(v), p).sha256)
    finally:
        stop.set(); t.join()
    assert not errors
    assert seen and set(seen) <= digests


def test_storage_guard_refuses_before_writing(root):
    p = root / 'state.pt'
    with pytest.raises(C.StorageGuardError):
        C.publish(payload(6), p, reserve=10 ** 18)
    assert list(root.iterdir()) == []
