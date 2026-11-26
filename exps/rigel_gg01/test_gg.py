"""GG-01 focused tests. CPU tests use synthetic unit tokens; the parity test (CUDA + locked caches) checks that the
GG-01 learner's V1/F/U arms reproduce the locked RIGEL-CAD v1 grids bitwise (MPDD, order a, seed 2711).

    python -m pytest -q -p no:cacheprovider exps/rigel_gg01/test_gg.py
"""
from pathlib import Path

import pytest
import torch

from exps.rigel_gg01 import gg
from rigel_cad import RigelCAD, RigelConfig, ops


def _tokens(n, seed=0, d=768):
    g = torch.Generator().manual_seed(seed)
    return ops.unit(torch.randn(n, d, generator=g)).half()


def test_kcenter_w_semantics():
    x = _tokens(600)
    W = torch.eye(768)
    a = gg.kcenter_w(x, 50, 2711, W)
    b = gg.kcenter_w(x, 50, 2711, W)
    assert torch.equal(a, b)                                     # deterministic
    assert int(a[0]) == int(ops.greedy_kcenter(x, 50, 2711)[0])  # same seeded start index as v1
    assert len(set(a.tolist())) == 50


def test_kcenter_w_is_farthest_point_in_whitened_space():
    x = _tokens(300, 1)
    A = torch.randn(768, 768, generator=torch.Generator().manual_seed(3)) / 30
    W = A @ A.T + torch.eye(768)
    idx = gg.kcenter_w(x, 20, 7, W)
    z = x.double() @ W.double()
    for i in range(1, 20):                                       # brute-force check of every greedy step
        d = torch.cdist(z, z[idx[:i]]).min(1).values ** 2
        assert int(d.argmax()) == int(idx[i]) or abs(float(d.max() - d[idx[i]])) < 1e-3 * float(d.max())


def test_gg_equals_v1_when_no_history_and_differs_after():
    s = gg.Stream(['V1', 'GG01'], 2711, device='cpu')
    for m in s.arms.values():
        m.cap = 60                                               # small cap so that consolidation binds on CPU
    for l in range(3):
        x = _tokens(8000, 10 + l)
        s.task(l, x, gg.fit_task(x, 2711))
    assert s.arms['GG01'].log[0]['metric'] == 'P'               # task 1 (80 > 60 keys): v1 rule
    assert torch.equal(s.snap['V1'][0], s.snap['GG01'][0])
    assert [r['metric'] for r in s.arms['GG01'].log[1:]] == ['W_prev', 'W_prev']
    assert not torch.equal(s.snap['V1'][2], s.snap['GG01'][2])  # the metric matters
    assert s.arms['GG01'].bytes() == s.arms['V1'].bytes()


def test_v1_arm_equals_rigel_cad_package():
    s = gg.Stream(['V1', 'F', 'U'], 2711, device='cpu')
    ref = {'V1': RigelCAD(RigelConfig(budget_mib=2.3, device='cpu')),
           'F': RigelCAD(RigelConfig(budget_mib=0.1, slow_level=False, device='cpu')),
           'U': RigelCAD(RigelConfig(budget_mib=None, device='cpu'))}
    s.arms['V1'].cap = ref['V1'].config.key_capacity
    s.arms['F'].cap = ref['F'].config.key_capacity
    q = _tokens(784 * 2, 99)
    for l in range(3):
        x = _tokens(5000, 20 + l)
        s.task(l, x, gg.fit_task(x, 2711))
        for m in ref.values():
            m.fit_task(x)
    for a, m in ref.items():
        assert torch.equal(s.state(a, 2)[0], m.state().keys), a
        assert torch.equal(s.energy(a, 2, q), m.patch_energy(q)), a


def test_cc_expand_base_set():
    x = _tokens(5000, 5)
    base = _tokens(40, 6)
    e = gg.cc_expand(x, 30, 2711, base)
    P = torch.randn(768, 128, generator=torch.Generator().manual_seed(2711)) / 128 ** 0.5
    z, zb = x.float() @ P, base.float() @ P
    d0 = torch.cdist(z, zb).min(1).values
    assert int(e[0]) == int(d0.argmax())                         # first point: farthest from the base set
    assert len(set(e.tolist())) == 30


def test_byte_accounting():
    p = gg.physical_bytes(gg.CAP_SLOW, True)
    assert p['nominal_registered'] == 8_388_104 and p['total'] == 3925 * 1536 + 768 * 768 * 8 + 8
    assert gg.physical_bytes(gg.CAP_FAST, False)['total'] == 5461 * 1536


MRB = Path('/mnt/k/NLCAD/cache/nl03_multirate_01/bench/mpdd_a2711')
G3E = Path('/mnt/k/NLCAD/cache/g3e_paper_01/cells')


@pytest.mark.skipif(not (torch.cuda.is_available() and MRB.exists()), reason='needs CUDA and locked caches')
def test_parity_with_locked_v1_mpdd():
    from exps.nl03_multirate_01 import final as FN
    from exps.nl03_multirate_01 import run as RN
    from exps.nl03_pressure_00 import core as PC
    G = PC.bind()
    PC.use(G, 'mpdd')
    order = FN.bench_order('mpdd', 'a')
    s = gg.Stream(['V1', 'GG01', 'F', 'U'], 2711)
    for l, t in enumerate(order):
        x16, _ = RN.fit_tokens(G, t, order=FN.train_rows_sorted(G, t))
        s.task(l, x16, gg.fit_task(x16, 2711))
        del x16
    ex = PC.extract_rec(G, 'mpdd')
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = G.load_test('mpdd', c, ex)
        for arm, src in (('V1', MRB / 'E'), ('U', MRB / 'U'), ('F', G3E / 'A4_dinov2_mpdd_a2711' / 'F')):
            ref = torch.load(src / f'{c}.pt', weights_only=True)
            assert ref['rows'] == rows
            mine = torch.stack([s.energy(arm, l, q) for l in ref['stages']]).reshape(len(ref['stages']), len(rows), 28, 28)
            assert torch.equal(mine, ref['grid']), (arm, c)
