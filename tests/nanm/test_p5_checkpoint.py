"""P5 CPU tests: checkpoint publication, verification, failure injection, fresh-process continuation (bitwise).

    python -B -m pytest -q -p no:cacheprovider tests/nanm/test_p5_checkpoint.py
Synthetic normal tokens only (no data). Torch threads pinned so CPU kernels are run-to-run identical.
"""
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from nanm import G3Config, NANMSystem
from nanm import state as NS

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]


def task_tokens(t, n_img=6):
    g = torch.Generator().manual_seed(1000 + t)
    return torch.randn(n_img, 784, 768, generator=g) + 0.3 * t


def queries():
    return torch.randn(3, 784, 768, generator=torch.Generator().manual_seed(7))


def fingerprint(s):
    h = hashlib.sha256()
    for k in s.state.keys:
        h.update(k.numpy().tobytes())
    h.update(s.state.S.numpy().tobytes()); h.update(str(s.state.n).encode())
    h.update(str(len(s.state.tasks)).encode())
    if s._cache is not None:
        h.update(s._cache[1].numpy().tobytes()); h.update(s._cache[4].numpy().tobytes())
    return h.hexdigest()


def fitted(n_tasks, root=None):
    s = NANMSystem.create(G3Config(), 'cpu')
    for t in range(n_tasks):
        s.fit_task(task_tokens(t), task_label=f't{t}')
        if root is not None:
            s.save_state(root)
    return s


def same(a, b):
    pa, pb = a.predict(queries()), b.predict(queries())
    return (fingerprint(a) == fingerprint(b) and all(torch.equal(x, y) for x, y in zip(a.state.selected, b.state.selected))
            and torch.equal(a._cache[0], b._cache[0]) and all(torch.equal(pa[k], pb[k]) for k in pa))


def test_roundtrip_bitwise(tmp_path):
    a = fitted(2, tmp_path)
    b = NANMSystem.load_state(tmp_path, device='cpu')
    assert same(a, b) and b.load_report['cond_rebuilt_equals_saved']
    assert len(NS.list_generations(tmp_path)) == 2


def test_fresh_process_continuation(tmp_path):
    a = fitted(4)                                                 # uninterrupted
    fitted(2, tmp_path)                                           # stop at boundary 2, save, unload
    code = textwrap.dedent(f"""
        import sys, torch; sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
        torch.set_num_threads(2)
        from test_p5_checkpoint import task_tokens
        from nanm import NANMSystem
        s = NANMSystem.load_state({str(tmp_path)!r}, device='cpu')
        for t in (2, 3):
            s.fit_task(task_tokens(t), task_label=f't{{t}}')
        s.save_state({str(tmp_path)!r})
    """)
    subprocess.run([sys.executable, '-B', '-c', code], check=True)
    b = NANMSystem.load_state(tmp_path, device='cpu')
    assert same(a, b)


def _prev_ok(root, gen):
    s = NANMSystem.load_state(root, device='cpu')
    return s.load_report['generation'] == gen


@pytest.mark.parametrize('fault', ['after_payload', 'after_manifest'])
def test_interrupted_write_not_published(tmp_path, fault):
    s = fitted(1, tmp_path); gen = NS.list_generations(tmp_path)[0]
    s.fit_task(task_tokens(1), task_label='t1')
    with pytest.raises(NS.CheckpointError):
        s.save_state(tmp_path, _fault=fault)
    assert NS.list_generations(tmp_path) == [gen] and _prev_ok(tmp_path, gen)
    assert any(p.name.startswith('.tmp_') for p in tmp_path.iterdir())          # left behind, never listed


def test_real_crash_before_commit(tmp_path):
    fitted(1, tmp_path); gen = NS.list_generations(tmp_path)[0]
    code = textwrap.dedent(f"""
        import os, sys, torch; sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
        torch.set_num_threads(2)
        from test_p5_checkpoint import task_tokens
        from nanm import NANMSystem
        import nanm.state as NS
        s = NANMSystem.load_state({str(tmp_path)!r}, device='cpu')
        s.fit_task(task_tokens(1), task_label='t1')
        NS._os.rename = lambda *a: os._exit(3)                     # hard process death at the commit point
        s.save_state({str(tmp_path)!r})
    """)
    r = subprocess.run([sys.executable, '-B', '-c', code])
    assert r.returncode == 3
    assert NS.list_generations(tmp_path) == [gen] and _prev_ok(tmp_path, gen)


def test_interrupted_after_commit_before_latest(tmp_path):
    s = fitted(1, tmp_path); gen = NS.list_generations(tmp_path)[0]
    s.fit_task(task_tokens(1), task_label='t1')
    with pytest.raises(NS.CheckpointError):
        s.save_state(tmp_path, _fault='after_rename')
    gens = NS.list_generations(tmp_path)
    assert len(gens) == 2 and _prev_ok(tmp_path, gen)              # LATEST still the previous valid generation
    new = [g for g in gens if g != gen][0]
    assert same(s, NANMSystem.load_state(tmp_path, device='cpu', generation=new))   # committed and complete


def _committed(tmp_path):
    fitted(1, tmp_path)
    s = NANMSystem.create(G3Config(), 'cpu')
    for t in range(2):
        s.fit_task(task_tokens(t), task_label=f't{t}')
    gen = s.save_state(tmp_path)
    return tmp_path / gen, NS.list_generations(tmp_path)[0]


@pytest.mark.parametrize('damage', ['missing_manifest', 'corrupt_payload', 'truncated_payload', 'bad_checksum',
                                    'wrong_encoder', 'wrong_contract', 'bad_schema', 'extra_file'])
def test_invalid_checkpoint_rejected(tmp_path, damage):
    d, first = _committed(tmp_path)
    man = json.loads((d / 'manifest.json').read_text())
    if damage == 'missing_manifest':
        (d / 'manifest.json').unlink()
    elif damage == 'corrupt_payload':
        b = bytearray((d / NS.PAYLOAD).read_bytes()); b[len(b) // 2] ^= 0xFF; (d / NS.PAYLOAD).write_bytes(bytes(b))
    elif damage == 'truncated_payload':
        b = (d / NS.PAYLOAD).read_bytes(); (d / NS.PAYLOAD).write_bytes(b[:len(b) // 3])
    elif damage == 'extra_file':
        (d / 'stray.bin').write_bytes(b'x')
    else:
        if damage == 'bad_checksum':
            man['files'][NS.PAYLOAD]['sha256'] = '0' * 64
        elif damage == 'wrong_encoder':
            man['encoder']['checkpoint_sha256'] = 'f' * 64
        elif damage == 'wrong_contract':
            man['contract']['sha256'] = 'e' * 64
        elif damage == 'bad_schema':
            man['schema'] = 'nanm.ckpt/999'
        (d / 'manifest.json').write_text(json.dumps(man))
    with pytest.raises(NS.CheckpointError):
        NANMSystem.load_state(tmp_path, device='cpu')
    assert NANMSystem.load_state(tmp_path, device='cpu', generation=first).state.stage == 0   # earlier one still valid


def test_wrong_contract_requested(tmp_path):
    fitted(1, tmp_path)
    with pytest.raises(NS.CheckpointError):
        NANMSystem.load_state(tmp_path, cfg=G3Config(seed=2712), device='cpu')
    with pytest.raises(NS.CheckpointError):
        NANMSystem.load_state(tmp_path, device='cpu', encoder_sha256='a' * 64)


@pytest.mark.parametrize('where', ['writer', 'geometry'])
def test_failed_fit_task_leaves_state_unchanged(monkeypatch, where):
    s = fitted(1); before = fingerprint(s)

    def boom(*a, **k):
        raise RuntimeError('injected')
    if where == 'writer':
        monkeypatch.setattr(s.writer, 'write', boom)
    else:
        monkeypatch.setattr(s.geometry, 'geometry', boom)
    with pytest.raises(RuntimeError):
        s.fit_task(task_tokens(1), task_label='t1')
    assert fingerprint(s) == before and len(s.state.tasks) == 1


def test_non_pd_geometry_rejected():
    from nanm.geometry import inv_sqrt_psd
    with pytest.raises(RuntimeError):
        inv_sqrt_psd(torch.zeros(4, 4, dtype=torch.float64), 1e-12)
