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
