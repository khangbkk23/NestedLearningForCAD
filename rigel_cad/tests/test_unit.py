"""RIGEL-CAD unit tests (CPU; no dataset or checkpoint needed)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from rigel_cad import data, metrics, ops
from rigel_cad.config import KEY_BYTES, SLOW_BYTES, MiB, RigelConfig
from rigel_cad.model import BudgetViolation, FastKeyMemory, RigelCAD


def _tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, 768, generator=g), dim=1).half()


def test_capacities_and_bytes():
    assert RigelConfig(budget_mib=8).key_capacity == 3925
    assert RigelConfig(budget_mib=4).key_capacity == 1194
    assert RigelConfig(budget_mib=16).key_capacity == 9386
    assert RigelConfig(budget_mib=32).key_capacity == 20309
    assert RigelConfig(budget_mib=8, slow_level=False).key_capacity == 5461
    assert RigelConfig(budget_mib=None).key_capacity is None
    assert SLOW_BYTES == 2_359_304 and KEY_BYTES == 1536
    for mib in (4, 8, 16, 32):
        c = RigelConfig(budget_mib=mib)
        assert c.key_capacity * KEY_BYTES + SLOW_BYTES <= mib * MiB < (c.key_capacity + 1) * KEY_BYTES + SLOW_BYTES


def test_kcenter_deterministic_and_prefix():
    x = _tokens(500, 0)
    a = ops.greedy_kcenter(x, 40, seed=2711)
    b = ops.greedy_kcenter(x, 40, seed=2711)
    assert torch.equal(a, b) and len(set(a.tolist())) == 40
    assert torch.equal(ops.greedy_kcenter(x, 10, seed=2711), a[:10])
    assert not torch.equal(ops.greedy_kcenter(x, 40, seed=2712), a)


def test_whitener_is_inverse_sqrt():
    x = torch.randn(4000, 768, dtype=torch.float64)
    S, n = x.T @ x, len(x)
    W, cond = ops.whitener(S, n)
    C = ops.shrink(S / n)
    assert torch.allclose(W, W.T, atol=1e-10)
    assert torch.allclose(W @ C @ W, torch.eye(768, dtype=torch.float64), atol=1e-8)
    assert cond >= 1


def test_fast_memory_rules():
    K = _tokens(30, 1)
    m = FastKeyMemory(capacity=50, selection='kcenter', seed=2711, device='cpu')
    m.write(K[:20]); m.write(K[20:])
    assert torch.equal(m.current(), K)                       # under capacity: union, insertion order
    m.write(_tokens(40, 2))
    assert len(m) == 50
    f = FastKeyMemory(capacity=25, selection='fifo', seed=2711, device='cpu')
    f.write(K[:20]); f.write(K[20:])
    assert torch.equal(f.current(), K[5:])                   # oldest five evicted
    u = FastKeyMemory(capacity=None, selection='kcenter', seed=2711, device='cpu')
    u.write(K[:20]); u.write(K[20:])
    assert torch.equal(u.current(), K)


def _stream(model, n_tasks=3, n_img=12):
    for t in range(n_tasks):
        model.fit_task(_tokens(784 * n_img, 10 + t), f't{t}')
    return model


def test_model_budget_and_unbounded_equivalence():
    big = _stream(RigelCAD(RigelConfig(budget_mib=1000, device='cpu')))
    unb = _stream(RigelCAD(RigelConfig(budget_mib=None, device='cpu')))
    q = _tokens(784 * 2, 99)
    assert torch.equal(big.state().keys, unb.state().keys)    # budget never binds -> identical to unbounded
    assert torch.equal(big.patch_energy(q), unb.patch_energy(q))
    small = _stream(RigelCAD(RigelConfig(budget_mib=2.5, device='cpu')))
    st = small.state()
    assert st.bytes <= 2.5 * MiB and len(st.keys) == RigelConfig(budget_mib=2.5).key_capacity
    with pytest.raises(BudgetViolation):
        small.fast.keys = torch.cat([small.fast.keys, small.fast.keys[:1]])
        small.state()


def test_fast_only_and_grid_shapes():
    m = _stream(RigelCAD(RigelConfig(budget_mib=8, slow_level=False, device='cpu')), n_tasks=2)
    st = m.state()
    assert st.W is None and st.bytes == len(st.keys) * KEY_BYTES
    g = m.patch_energy_grid(_tokens(784 * 3, 5).reshape(3, 784, 768))
    assert g.shape == (3, 28, 28) and bool((g >= 0).all())


def test_save_load_roundtrip(tmp_path):
    for cfg in (RigelConfig(budget_mib=2.5, device='cpu'), RigelConfig(budget_mib=None, device='cpu')):
        m = _stream(RigelCAD(cfg))
        p = m.save(tmp_path / 'm.rigel')
        r = RigelCAD.load(p)
        q = _tokens(784, 7)
        assert torch.equal(m.patch_energy(q), r.patch_energy(q)) and r.persistent_bytes() == m.persistent_bytes()


def test_orders():
    o = data.stream_orders('mpdd')
    assert o['a'] == data.MPDD_CATEGORIES and o['b'] == tuple(reversed(o['a'])) and sorted(o['c']) == sorted(o['a'])


def test_binned_ap_and_aupro_small():
    pos = np.array([0, 0, 1, 2], np.float32); tot = np.array([5, 3, 2, 2], np.float32)
    # descending bins: (2 pos / 2), (1 / 2), (0 / 3), (0 / 5): AP = 2/3 * 1 + 1/3 * 3/4
    assert abs(metrics.binned_ap(pos, tot) - (2 / 3 + 1 / 4)) < 1e-6
    acc = metrics.ProAccumulator()
    m = np.zeros((8, 8), np.float32); m[2:4, 2:4] = 1.0
    acc.add(m.astype(np.float16), m > 0)                         # perfect map
    assert abs(acc.au_pro(0.05) - 1.0) < 1e-9
    f1 = metrics.SegF1(); f1.add(m > 0.5, m > 0)
    assert f1.value() == 1.0
    assert abs(metrics.trapezoid([0, 0.5, 1], [0, 1, 1], x_max=0.25) - 0.0625) < 1e-12


def test_ad2_mask_lookup_uses_directory_not_stem(tmp_path):
    """MVTec AD 2 good and bad test_public images share file names; only bad/ images may get a mask."""
    gt = tmp_path / 'can' / 'test_public' / 'ground_truth' / 'bad'
    gt.mkdir(parents=True)
    (gt / '000_regular_mask.png').write_bytes(b'x')
    for d in ('good', 'bad'):
        (tmp_path / 'can' / 'test_public' / d).mkdir(parents=True, exist_ok=True)
    assert data.ad2_mask_of(tmp_path, 'can/test_public/good/000_regular.png') is None
    assert data.ad2_mask_of(tmp_path, 'can/test_public/bad/000_regular.png') == gt / '000_regular_mask.png'
    with pytest.raises(FileNotFoundError):
        data.ad2_mask_of(tmp_path, 'can/test_public/bad/001_regular.png')
    with pytest.raises(ValueError):
        data.ad2_mask_of(tmp_path, 'can/validation/good/000_regular.png')


AD2_ROOT = Path('/mnt/k/mvtec_ad_2')


@pytest.mark.skipif(not AD2_ROOT.exists(), reason='needs MVTec AD 2')
def test_ad2_masks_on_real_dataset():
    """Every bad test_public image has a non-empty mask; every good one gets None (all-normal)."""
    for s in data.ad2_scenarios(AD2_ROOT):
        for r in data.ad2_files(AD2_ROOT, s, 'test_public'):
            m = data.ad2_mask_of(AD2_ROOT, r)
            assert (m is None) == ('/good/' in r), r


# ---------------------------------------------------------------------------------------------- compact storage
from rigel_cad import storage as STO  # noqa: E402
from rigel_cad.config import KEY4_BYTES, SLOW_TRI_BYTES  # noqa: E402
from rigel_cad.model import FORMAT, FORMAT_INT4_TRI  # noqa: E402

Q4 = 'int4_triangular'


def test_int4_capacity_bytes_and_config():
    c = RigelConfig(budget_mib=8, storage=Q4)
    assert (KEY4_BYTES, SLOW_TRI_BYTES, STO.N_TRI) == (386, 1_181_192, 295_296)
    assert (c.key_bytes, c.slow_bytes, c.key_capacity) == (386, 1_181_192, 18_672)
    assert c.key_capacity * 386 + SLOW_TRI_BYTES <= 8 * MiB < (c.key_capacity + 1) * 386 + SLOW_TRI_BYTES
    assert RigelConfig(budget_mib=8).storage == 'fp16' and RigelConfig(budget_mib=8).key_capacity == 3925
    for bad in (dict(budget_mib=None), dict(slow_level=False), dict(selection='fifo')):
        with pytest.raises(ValueError):
            RigelConfig(storage=Q4, **bad)
    with pytest.raises(ValueError):
        RigelConfig(storage='int8')
    assert 'storage' not in RigelConfig().to_dict()            # v1 config layout unchanged
    assert RigelConfig.from_dict(c.to_dict()) == c


def test_int4_quantiser_pack_roundtrip():
    x = torch.cat([_tokens(300, 3), torch.zeros(1, 768, dtype=torch.float16)])   # include an all-zero row
    q, s = STO.quantize4(x)
    assert q.dtype == torch.int8 and s.dtype == torch.float16 and int(q.abs().max()) <= 7
    assert int(q[:-1].abs().amax(1).min()) == 7                # absmax element maps to +-7
    k = STO.dequantize(q, s)
    q2, s2 = STO.quantize4(k)
    assert torch.equal(q2, q) and torch.equal(s2, s)           # idempotent: stored codes are a fixed point
    p = STO.pack4(q)
    assert p.dtype == torch.uint8 and p.shape == (len(x), 384)
    assert torch.equal(STO.unpack4(p), q)
    assert p[0].numel() * p.element_size() + s[0].numel() * s.element_size() == KEY4_BYTES
    assert float((k[:-1] - x[:-1].float()).abs().max()) <= float(s.float().max()) / 2 + 1e-7


def test_triangular_statistic_exact():
    x = torch.randn(3000, 768, dtype=torch.float64)
    S = x.T @ x
    t = STO.tri_pack(S)
    assert t.dtype == torch.float32 and t.numel() == STO.N_TRI
    assert torch.equal(STO.tri_unpack(t), S.float().double())
    W_tri, _ = ops.whitener(STO.tri_unpack(t), 3000)
    W_full, _ = ops.whitener(S.float().double(), 3000)
    assert torch.equal(W_tri, W_full)                          # triangle == full fp32 matrix, bitwise


def _q4_cfg():
    return RigelConfig(budget_mib=1.2, storage=Q4, device='cpu')   # 199 keys: consolidation binds after 7 tasks


def test_int4_memory_consolidates_dequantised_codes():
    m = RigelCAD(_q4_cfg(), keep_history=True)
    _stream(m, n_tasks=8, n_img=4)                              # 32 keys per task
    assert len(m.fast) == m.config.key_capacity == 199
    st = m.state()
    assert st.keys.dtype == torch.float32 and torch.equal(st.keys, STO.dequantize(STO.unpack4(st.q4), st.scale))
    assert st.bytes == 199 * 386 + SLOW_TRI_BYTES <= 1.2 * MiB
    for l in range(8):                                          # compact history reproduces the live state
        h = m.state_at(l)
        assert len(h.keys) == min(32 * (l + 1), 199) and h.keys.dtype == torch.float32
    assert torch.equal(m.state_at(7).keys, st.keys) and torch.equal(m.state_at(7).W, st.W)
    # under capacity the int4 memory is the quantised union, in insertion order
    u = RigelCAD(RigelConfig(budget_mib=8, storage=Q4, device='cpu'))
    _stream(u, n_tasks=2, n_img=4)
    assert len(u.fast) == 64


def test_int4_save_load_resume(tmp_path):
    full = _stream(RigelCAD(_q4_cfg()), n_tasks=8, n_img=4)
    part = _stream(RigelCAD(_q4_cfg()), n_tasks=4, n_img=4)
    p = part.save(tmp_path / 'q4.rigel')
    d = torch.load(p, weights_only=True)
    assert d['format'] == FORMAT_INT4_TRI and d['storage'] == Q4 and d['q4'].dtype == torch.uint8
    tensor_bytes = sum(d[k].numel() * d[k].element_size() for k in ('q4', 'scale', 'S_tri')) + 8
    assert tensor_bytes == part.persistent_bytes() == 128 * 386 + SLOW_TRI_BYTES
    r = RigelCAD.load(p)
    assert r.config == part.config and r.n_tasks == 4
    for t in range(4, 8):                                       # continue learning after restoration
        r.fit_task(_tokens(784 * 4, 10 + t), f't{t}')
    assert torch.equal(r.fast.q, full.fast.q) and torch.equal(r.fast.scale, full.fast.scale)
    assert torch.equal(r.slow.S_tri, full.slow.S_tri) and r.slow.n == full.slow.n
    q = _tokens(784, 7)
    assert torch.equal(r.patch_energy(q), full.patch_energy(q))


def test_v1_checkpoint_format_unchanged(tmp_path):
    m = _stream(RigelCAD(RigelConfig(budget_mib=2.5, device='cpu')))
    d = torch.load(m.save(tmp_path / 'v1.rigel'), weights_only=True)
    assert d['format'] == FORMAT == 'rigel-cad-state/1' and 'storage' not in d and 'storage' not in d['config']
    assert set(d) == {'format', 'config', 'n_tasks', 'task_names', 'keys', 'blocks', 'S', 'n', 'persistent_bytes'}
    assert d['keys'].dtype == torch.float16 and d['S'].dtype == torch.float64
    d['format'] = FORMAT_INT4_TRI                               # format / storage mismatch is refused
    torch.save(d, tmp_path / 'bad.rigel')
    with pytest.raises(ValueError):
        RigelCAD.load(tmp_path / 'bad.rigel')


V1_STATE = Path(__file__).resolve().parents[2] / 'results/rigel_cad/states/mpdd_a2711_dinov2_vitb14_reg4_B8.rigel'


@pytest.mark.skipif(not V1_STATE.exists(), reason='needs the RIGEL-CAD 1.0 release state file')
def test_loads_rigel_cad_1_0_checkpoint():
    m = RigelCAD.load(V1_STATE, device='cpu')
    assert m.config.storage == 'fp16' and m.n_tasks == 6
    assert m.persistent_bytes() == torch.load(V1_STATE, weights_only=True)['persistent_bytes']
    st = m.state()
    assert st.keys.dtype == torch.float16 and st.W.shape == (768, 768)


def _fake_ad2(root):
    """Two-scenario MVTec AD 2 folder layout with tiny images (API check only; no real data, no evaluation claim)."""
    from PIL import Image
    for s in ('aaa', 'bbb'):
        for rel in ('train/good/000.png', 'train/good/001.png', 'validation/good/000.png',
                    'test_public/good/000_regular.png', 'test_public/bad/000_regular.png'):
            p = root / s / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (40, 32), (128, 128, 128)).save(p)
        m = root / s / 'test_public/ground_truth/bad/000_regular_mask.png'
        m.parent.mkdir(parents=True, exist_ok=True)
        a = np.zeros((32, 40), dtype=np.uint8)
        a[8:16, 10:20] = 255
        Image.fromarray(a).save(m)


def test_ad2_pipeline_accepts_int4_storage_and_keeps_v1_files(tmp_path, monkeypatch):
    from rigel_cad import pipeline

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def get(self, key, root, rels):
            return _tokens(784 * len(list(rels)), sum(map(ord, key))).reshape(-1, 784, 768)

    monkeypatch.setattr(pipeline, 'TokenStore', FakeStore)
    _fake_ad2(tmp_path / 'ad2')
    out = tmp_path / 'out'
    r4 = pipeline.run_ad2(tmp_path / 'ad2', RigelConfig(budget_mib=1.2, storage=Q4, device='cpu'), out,
                          per_stage=True, write_submission=False)
    rv = pipeline.run_ad2(tmp_path / 'ad2', RigelConfig(budget_mib=8, device='cpu'), out, per_stage=True,
                          write_submission=False)
    assert r4['storage'] == Q4 and rv['storage'] == 'fp16'
    assert (out / 'RESULTS_ad2_int4tri.json').exists() and (out / 'RESULTS_ad2.json').exists()
    assert (out / 'states' / 'ad2_int4tri.rigel').exists() and (out / 'states' / 'ad2.rigel').exists()
    assert RigelCAD.load(out / 'states' / 'ad2_int4tri.rigel').config.storage == Q4
    assert r4['bytes'] == RigelCAD.load(out / 'states' / 'ad2_int4tri.rigel').persistent_bytes()
    assert pipeline.storage_suffix(RigelConfig()) == '' and pipeline.storage_suffix(RigelConfig(storage=Q4)) == '_int4tri'


def test_stream_pipeline_int4_unbounded_and_v1_file_names(tmp_path, monkeypatch):
    """run_stream on a fake MPDD layout (API check): int4 storage and the unbounded reference run end to end, and the
    int4 run writes *_int4tri files next to (never over) the fp16 ones."""
    from PIL import Image
    from rigel_cad import pipeline

    class FakeStore:
        def __init__(self, *a, **k):
            pass

        def get(self, key, root, rels):
            return _tokens(784 * len(list(rels)), sum(map(ord, key))).reshape(-1, 784, 768)

    monkeypatch.setattr(pipeline, 'TokenStore', FakeStore)
    root = tmp_path / 'mpdd'
    for c in data.MPDD_CATEGORIES:
        for rel in ('train/good/000.png', 'train/good/001.png', 'test/good/000.png', 'test/hole/000.png'):
            p = root / c / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (40, 32), (128, 128, 128)).save(p)
        a = np.zeros((32, 40), dtype=np.uint8)
        a[8:16, 10:20] = 255
        (root / c / 'ground_truth/hole').mkdir(parents=True)
        Image.fromarray(a).save(root / c / 'ground_truth/hole/000_mask.png')
    out = tmp_path / 'out'
    runs = {'q4': RigelConfig(budget_mib=2.5, storage=Q4, device='cpu'),
            'v1': RigelConfig(budget_mib=2.5, device='cpu'), 'u': RigelConfig(budget_mib=None, device='cpu')}
    res = {k: pipeline.run_stream('mpdd', root, c, 'a', out, bootstrap=False) for k, c in runs.items()}
    assert res['q4']['tag'].endswith('_int4tri') and res['q4']['storage'] == Q4
    assert res['q4']['tag'][:-len('_int4tri')] == res['v1']['tag']
    for r in res.values():
        assert (out / f"RESULTS_{r['tag']}.json").exists() and (out / 'states' / f"{r['tag']}.rigel").exists()
    assert res['q4']['bytes_max'] == 2 * 6 * 8 * 386 + SLOW_TRI_BYTES      # 8 keys per task, never consolidated
