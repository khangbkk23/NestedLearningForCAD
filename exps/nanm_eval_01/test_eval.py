"""Focused tests for NANM-EVAL-01. Synthetic + REPRO-01 artifact parity (grids and pre-existing exposed metrics only;
no new anomaly metric is computed here). GPU required for the parity tests."""
import math

import numpy as np
import pytest
import torch

from exps.nanm_eval_01 import arms as AR
from exps.nanm_eval_01 import common as U
from exps.nanm_eval_01 import metrics as M

torch.backends.cuda.matmul.allow_tf32 = False


@pytest.fixture(scope='module')
def G():
    return U.bind()


# ---------------------------------------------------------------- sizing / plan (pure)
def test_cap_ratio_reproduces_bench_g3_c_sizes(G):
    rec = G.nrec()['visa']['sizes']
    r, sz = U.cap_ratio(rec['n_fit'], G.C_BYTES, U.S_B)
    assert sz == rec['g3c'] and abs(r - rec['g3c_ratio']) < 1e-12
    r0, sz0 = U.cap_ratio(rec['n_fit'], G.C_BYTES, 0)
    assert sz0 == rec['a0c30']


def test_a0b_like_sizes_reproduces_canonical(G):
    for ds in ('mpdd', 'visa'):
        rec = G.nrec()[ds]['sizes']
        assert U.a0b_like_sizes(rec['n_fit'], 0.01)[1] == rec['a0b']
        assert U.ratio_sizes(rec['n_fit'], 0.01) == rec['g3']


def test_cap_infeasible_rules():
    assert U.cap_ratio([1000, 1000], 10 ** 12, 0) == (None, None)            # more than every token
    assert U.cap_ratio([1000, 1000], 100, U.S_B) == (None, None)            # below the statistic itself


def test_orders_are_permutations():
    base = tuple('abcdefghijkl')
    od = U.orders(base)
    assert od['a'] == base and od['b'] == base[::-1] and sorted(od['c']) == sorted(base) and od['c'] not in (od['a'], od['b'])


def test_refresh_stage_and_catchup():
    assert [AR.refresh_stage(l, 2, 6) for l in range(6)] == [0, 1, 1, 3, 3, 5]
    assert [AR.refresh_stage(l, 4, 6) for l in range(6)] == [0, 0, 0, 3, 3, 5]
    for T in (6, 12):
        for P in (2, 4):
            assert AR.refresh_stage(T - 1, P, T) == T - 1                    # final catch-up


def test_binned_ap_rows_matches_oracle():
    from exps.nanm_r0 import audit as A
    rng = np.random.default_rng(0)
    P = rng.integers(0, 5, (7, 64)).astype(np.float32); Tt = P + rng.integers(0, 20, (7, 64)).astype(np.float32)
    v = M.binned_ap_rows(P, Tt)
    for i in range(7):
        assert v[i] == A.binned_ap(P[i], Tt[i])


def test_boot_weights_deterministic():
    a, b = M.boot_weights(50, 3, B=20), M.boot_weights(50, 3, B=20)
    assert np.array_equal(a, b) and a.sum(1).tolist() == [50.0] * 20 and not np.array_equal(a, M.boot_weights(50, 4, B=20))


def test_bytes_ledger():
    assert U.bytes_of('g3', 4221) == 8_842_760                                # BENCH-G3 MPDD G3 bytes
    assert U.bytes_of('g3', 40780) == 64_997_384                              # BENCH-G3 VisA G3 bytes
    assert U.bytes_of('euclid', 5758) == 8_844_288 and U.bytes_of('g1', 4221) == 8_845_832


# ---------------------------------------------------------------- canonical parity on REPRO-01 artifacts (GPU)
@pytest.fixture(scope='module')
def mpdd(G):
    from exps.nanm_eval_01 import run as RN
    base, si = RN.sizes_info(G, 'mpdd')
    banks = RN.load_banks(G, 'mpdd', 2711)
    ex = U.rj(G.OUT / 'EXTRACT_COMPLETE.json')
    Q = {c: G.load_test('mpdd', c, ex) for c in base}
    return base, si, banks, Q


def test_stage_W_bitwise_equals_stored(G, mpdd):
    base, si, banks, _ = mpdd
    stage = torch.load(G.NORM / 'mpdd_s2711_stage.pt', weights_only=True)
    for l in range(len(base)):
        W, _ = AR.stage_specs('N0', base, l, banks, si['k1'], {'k1': si['k1']})[2:4]
        W1 = AR.stage_specs('A2', base, l, banks, si['k1'], {'k1': si['k1']})[2]
        assert torch.equal(W, stage[l]['G3']) and torch.equal(W1, stage[l]['G1b']), l


def test_score_parity_and_stage0_identities(G, mpdd):
    base, si, banks, Q = mpdd
    cfg = {'k1': si['k1']}
    arms = {a: (a, si['a0b'] if a == 'A1' else si['k1'], cfg) for a in ('N0', 'A1', 'A2', 'N1', 'N2', 'N3', 'N4', 'B5')}
    grids, _ = AR.score_cell(G, base, banks, Q, arms)
    for c in base:
        ref = torch.load(G.SCORE / f'mpdd_{c}.pt', weights_only=True)['grids']
        assert torch.equal(grids['N0'][c], ref['G3|s2711']), c
        assert torch.equal(grids['A1'][c], ref['A0b|s2711']), c
        assert torch.equal(grids['A2'][c], ref['G1b|s2711']), c
    c0 = base[0]
    for a in ('N1', 'N2', 'N3', 'N4'):                                        # at stage 0 every factorial arm == N0
        assert torch.equal(grids[a][c0][0], grids['N0'][c0][0]), a
    assert torch.allclose(grids['B5'][c0][0], grids['N0'][c0][0], rtol=1e-5, atol=1e-6)
    for c in base:                                                            # N3 never changes after acquisition
        assert all(torch.equal(grids['N3'][c][i], grids['N3'][c][0]) for i in range(len(grids['N3'][c])))


def test_kcentre_prefix_determinism(G):
    from exps.nanm_r0 import audit as A
    rec = G.nrec()
    base = G.load_norm('mpdd', 2711, 'metal_plate', rec)
    fit = G.fit_tokens('mpdd', 'metal_plate')
    kc = A.greedy_coreset(fit, len(base['kc']) + 64, seed=2711)
    assert torch.equal(kc[:len(base['kc'])], base['kc'])


def test_metrics_parity_on_stored_grid(G):
    """Pre-existing exposed artifact only: the evaluator wrapper reproduces REPRO-01 CELLS for stored G3|s2711."""
    cells = U.rj(U.REPRO / 'bench_g3/CELLS_mpdd.json')
    d = torch.load(G.SCORE / 'mpdd_tubes.pt', weights_only=True)
    cat = M.load_category(G, 'mpdd', 'tubes', d['rows'])
    res, _ = M.eval_grid(G, cat, d['grids']['G3|s2711'], 5, 6, True, None)
    ref = cells['mpdd|G3|s2711|5|tubes']
    for met, v in ref.items():
        assert res[5][met] == v, met
