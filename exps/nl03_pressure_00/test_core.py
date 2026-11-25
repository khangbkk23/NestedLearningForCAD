"""NL03-MEMORY-PRESSURE-00 tests T1-T6 (registration §7). T1 uses the GPU on MPDD (canonical parity); no anomaly label
or mask is read by any test (T5 uses a synthetic category)."""
import numpy as np
import pytest
import torch

from exps.nl03_pressure_00 import core as C

MiB = C.MiB


@pytest.fixture(scope='module')
def G():
    torch.backends.cuda.matmul.allow_tf32 = False
    return C.bind()


@pytest.fixture(scope='module')
def mp(G):
    order = C.orders('mpdd')['a']
    mat = C.load_material(G, order, 2711)
    return order, mat, C.Pool(order, mat)


# ------------------------------------------------------------------ T1 parity oracle (B = infinity == canonical G3)
def test_T1_parity_oracle_mpdd(G, mp):
    order, mat, pool = mp
    traj, logs = C.run_arm('E', pool, 2711, None)
    assert all(len(x['evicted_ids']) == 0 for x in logs)                       # B = inf: no eviction
    Ws = C.whiteners(order, mat)
    C.use(G, 'mpdd')
    rec = C.nrec(G, 'mpdd')
    stage = G.load_stage('mpdd', 2711, rec)
    for l in range(len(order)):
        canon = torch.cat([G.load_norm('mpdd', 2711, t.split('/')[1], rec)['keys'][:mat[t]['k1']] for t in order[:l + 1]])
        got = pool.keys[traj[l].cuda()].cpu()
        assert torch.equal(got, canon)                                          # same tensor -> same multiset
        assert torch.equal(sum(mat[t]['S1'] for t in order[:l + 1]),
                           sum(G.load_norm('mpdd', 2711, t.split('/')[1], rec)['S1'] for t in order[:l + 1]))
        assert torch.equal(Ws['acc'][l], stage[l]['G3']), f'W stage {l}'
    ex = C.extract_rec(G, 'mpdd')
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = G.load_test('mpdd', c, ex)
        q = q.cuda()
        g = torch.stack([C.energy(G, q, pool.keys[traj[l].cuda()], Ws['acc'][l]) for l in range(j, len(order))])
        g = g.reshape(len(order) - j, len(rows), 28, 28)
        ref = torch.load(G.SCORE / f'mpdd_{c}.pt', weights_only=True)['grids']['G3|s2711']
        assert torch.equal(g, ref), f'grid parity {c}'


# ------------------------------------------------------------------ T2 byte accounting
def test_T2_byte_accounting(mp):
    order, mat, pool = mp
    for arm in C.ARMS[1:]:
        for b in C.BUDGETS_MIB:
            cap = C.cap_keys(b * MiB, C.ARM_SLOW[arm])
            assert C.bytes_of(cap, C.ARM_SLOW[arm]) <= b * MiB < C.bytes_of(cap + 1, C.ARM_SLOW[arm])
    assert C.bytes_of(10, 'g3') == 10 * 1536 + 2_359_304 and C.bytes_of(10, 'none') == 15360
    assert C.bytes_of(10, 'g1') == 10 * 1536 + 2_362_376
    caps = {f'{a}@{b}': C.cap_keys(b * MiB, C.ARM_SLOW[a]) for a in C.ARMS[1:] for b in C.BUDGETS_MIB}
    assert caps['E@8'] == 3925 and caps['F@8'] == 5461 and caps['G@8'] == 3923 and caps['E@4'] == 1194
    assert caps['E@16'] == 9386 and caps['F@4'] == 2730 and caps['G@16'] == 9384
    with pytest.raises(C.BudgetViolation):                                      # over-budget fixture
        C.assert_budget('E', 0, caps['E@4'] + 1, 'g3', 4 * MiB)
    with pytest.raises(C.BudgetViolation):                                      # a mis-sized stream fails loudly
        C.run_arm('E', pool, 2711, 4 * MiB, cap_override=caps['E@4'] + 5)
    for arm in C.ARMS[1:]:                                                      # every arm's logged bytes = bytes_of
        traj, logs = C.run_arm(arm, pool, 2711, 4 * MiB)
        assert all(x['bytes'] == C.bytes_of(x['n_keys'], C.ARM_SLOW[arm]) <= 4 * MiB for x in logs), arm


# ------------------------------------------------------------------ T3 eviction log invariants
def _fake_pool(T=4, k=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    order = [f'fake/t{i}' for i in range(T)]
    mat = {t: {'keys': torch.nn.functional.normalize(torch.randn(k, C.D, generator=g), dim=1).half(), 'k1': k}
           for t in order}
    return order, C.Pool(order, mat)


def test_T3_eviction_invariants_synthetic_and_real(mp):
    order, pool = _fake_pool()
    for arm in ('E', 'F', 'G', 'FIFO', 'Q'):
        traj, logs = C.run_arm(arm, pool, 2711, None, cap_override=70)
        ins, ev = 0, 0
        for l, x in enumerate(logs):
            assert x['invariant_ok'], (arm, l)
            assert len(traj[l]) == x['n_keys'] == min(70, 50 * (l + 1)) == int(x['survivors_per_task'].sum())
            ins += x['inserted']
            if arm != 'Q':
                ev += len(x['evicted_ids'])
                assert len(traj[l]) + ev == ins
    traj, _ = C.run_arm('FIFO', pool, 2711, None, cap_override=70)
    assert torch.equal(traj[1], torch.cat([torch.arange(30, 50), torch.arange(50, 100)]))
    traj, _ = C.run_arm('FIFO', pool, 2711, None, cap_override=40)              # new coreset alone > cap: prefix
    assert torch.equal(traj[2], torch.arange(100, 140))
    order, mat, rp = mp
    for arm in C.ARMS[1:]:
        traj, logs = C.run_arm(arm, rp, 2711, 4 * MiB)
        assert all(x['invariant_ok'] for x in logs), arm
        assert sum(len(x['evicted_ids']) for x in logs) > 0 or arm == 'Q', arm


# ------------------------------------------------------------------ T4 Q_B
def test_T4_hindsight_q(mp):
    order, mat, pool = mp
    traj, logs = C.run_arm('Q', pool, 2711, 4 * MiB)
    cap = (4 * MiB - C.SLOW_B['g3']) // 1536
    for l, K in enumerate(traj):
        assert len(set(K.tolist())) == len(K) and int(K.max()) < int(pool.off[l + 1])     # subset of seen union
        assert len(K) == min(cap, int(pool.off[l + 1]))
    assert len(traj[-1]) == cap == 1194


# ------------------------------------------------------------------ T5 lite evaluator == EVAL-01 eval_grid (synthetic)
def test_T5_eval_lite_equals_eval_grid(G):
    from exps.bench_aud_00 import run as AUD
    from exps.nanm_eval_01 import metrics as M
    from exps.nl03_pressure_00 import evaluate as EV
    rng = np.random.default_rng(0)
    n = 8
    hws = [(96, 128)] * n
    y = np.array([0, 1] * 4)
    masks = []
    for i in range(n):
        m = np.zeros(hws[i], bool)
        if y[i]:
            r, c = rng.integers(0, 60), rng.integers(0, 90)
            m[r:r + 20, c:c + 30] = True
        masks.append(m)
    cat = {'y': y, 'hws': hws, 'masks': masks, 'm224': [AUD.mask224(m) for m in masks]}
    g = torch.rand(3, n, 28, 28, generator=torch.Generator().manual_seed(1))
    g[:, 1::2, 8:12, 10:14] += 0.5
    bw = M.boot_weights(n, 0, B=50)
    ref, rb = M.eval_grid(G, cat, g, 0, 3, True, bw)
    got, gb = EV.eval_lite(cat, g, [0, 1, 2], bw, True)
    for l in range(3):
        assert got[l]['ap'] == ref[l]['ap_native_binned'] and got[l]['i_auroc'] == ref[l]['i_auroc']
    assert got[2]['ap_exact'] == ref[2]['ap_native']
    assert np.array_equal(gb['ap'], rb['ap']) and np.array_equal(gb['auroc'], rb['auroc'], equal_nan=True)


# ------------------------------------------------------------------ T6 orders, stages, exclusion
def test_T6_orders_stages_exclusion(G):
    from exps.nanm_eval_01 import common as U
    from exps.calm02 import data as CD
    s = C.orders('s33')
    assert len(s['a']) == 33 and all(sorted(v) == sorted(s['a']) for v in s.values()) and s['b'] == s['a'][::-1]
    assert s['a'][0] == 'mvtec/bottle' and s['a'][15] == 'visa/candle' and s['a'][-1] == 'mpdd/tubes'
    v = C.orders('visa')
    assert {k: [t.split('/')[1] for t in x] for k, x in v.items()} == {k: list(x) for k, x in U.orders(CD.VISA_ORDER).items()}
    assert C.logged_stages('s33', 33) == list(range(2, 33, 3)) and C.logged_stages('visa', 12) == list(range(12))
    for t in C.NL02_SELECTED:
        ex = C.nl02_excluded(G, t)
        assert len(ex) == 32
        cr = C.credit_rows(G, t)
        assert len(cr) == C.COVER_IMGS and not set(cr) & ex
    assert C.nl02_excluded(G, 'visa/candle') == set()
    ds, c = 'mpdd', 'tubes'
    rows = G.train_rows(ds, c)
    fit = {rows[i]['relative_path'] for i in G.roles(ds, rows)['FIT']}
    assert not set(C.credit_rows(G, 'mpdd/tubes')) & fit                       # CREDIT is disjoint from FIT
