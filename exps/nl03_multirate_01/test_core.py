"""NL03-MULTIRATE-01 Phase F tests (§12 F). GPU; normal-only data (MPDD FIT / synthetic fixtures); no test token.

    pytest -q -s -p no:cacheprovider exps/nl03_multirate_01/test_core.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from exps.nl03_multirate_01 import core as CO
from exps.nl03_multirate_01 import spec as SP

TAU = 6.0


def _tokens(n_img, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_img * 784, SP.D, generator=g)
    return torch.nn.functional.normalize(x, dim=-1).half().cuda()


def _drive(arms, budget, n_steps=20, seed=2711, tasks=(0,) * 64):
    fasts, slows, bind = CO.plan_arms(arms, budget, TAU, seed)
    hist = {a: [] for a in arms}
    seen_mass = 0
    for s in range(n_steps):
        xb = _tokens(SP.MB_IMAGES, 1000 + s)
        keys, S_b, n_b, cnt = CO.write_candidates(xb, seed, any(f.counts for f in fasts.values()))
        seen_mass += len(xb)
        for k, f in fasts.items():
            ev = f.step(keys, cnt, tasks[s], s)
            for a, kk in bind.items():
                if kk == k:
                    slows[a].after_fast(f, ev, S_b, n_b, s)
                    hist[a].append((s, CO.sha_t(slows[a].S) if slows[a].S is not None else None,
                                    len(f.K), slows[a].n_pending()))
    return fasts, slows, bind, hist, seen_mass


SMALL = 2 * SP.S_B + 1000 * 1540           # P: 1,000 stored keys (496 fast + 504 reserve) -> eviction from step 8


def test_schedule_and_hash_timing():
    """C=1 writes every step; C>1 writes exactly when (#fast steps) % C == 0; S changes only at slow writes."""
    fasts, slows, bind, hist, _ = _drive(('C2', 'C4', 'C3', 'P', 'P4'), SMALL, n_steps=17)
    for a in ('C2', 'C4', 'C3', 'P', 'P4'):
        C = SP.ARMS[a]['C']
        assert slows[a].writes == 17 // C, (a, slows[a].writes)
        wsteps = [r['step'] for r in slows[a].log]
        assert wsteps == [s for s in range(17) if (s + 1) % C == 0], (a, wsteps)
        prev = None
        for s, h, nk, npend in hist[a]:
            if h is not None and prev is not None and h != prev:
                assert (s + 1) % C == 0, f'{a}: S changed off-schedule at step {s}'
            prev = h if h is not None else prev
        for r in slows[a].log:                                   # window provenance: inputs strictly from the window
            assert r['in_steps'][1] == r['step'] and r['n_in_steps'] == C and r['in_steps'][0] == r['step'] - C + 1


def test_pending_and_budget_invariants():
    fasts, slows, bind, hist, seen = _drive(('P', 'C2', 'C3', 'C1'), SMALL, n_steps=19)
    for a in ('P', 'C2', 'C3', 'C1'):
        ntot = SP.total_cap(a, SMALL)
        for s, h, nk, npend in hist[a]:
            assert nk + npend <= ntot
        assert slows[a].max_bytes <= SMALL
    f = fasts[bind['P']]
    assert len(set(f.ids.tolist())) == len(f.ids)
    # mass conservation (counts arm P): kept + pending + released-by-slow-writes == tokens seen
    released = sum(r['n_evict_mass'] for r in slows['P'].log)
    pend = float(sum(c.sum() for c in slows['P'].P['cnt'])) if slows['P'].P['cnt'] else 0.0
    assert int(f.cnt.sum()) + pend + released == seen
    # C1/C3: no pending at any time
    assert all(npend == 0 for *_, npend in hist['C1']) and all(npend == 0 for *_, npend in hist['C3'])


def test_budget_overflow_fixture():
    s = CO.Slow('P', 8 * SP.MiB, TAU)
    ntot = SP.total_cap('P', 8 * SP.MiB)
    s.check(ntot, 'ok')
    with pytest.raises(SP.BudgetViolation):
        s.check(ntot + 1, 'overflow')
    s.P['ids'].append(torch.zeros(5, dtype=torch.long))
    with pytest.raises(SP.BudgetViolation):
        s.check(ntot - 4, 'pending-overflow')                   # pending keys are charged


def test_eviction_event_order_and_transfer_provenance():
    fasts, slows, bind, hist, _ = _drive(('P',), SMALL, n_steps=24)
    log = slows['P'].log
    assert any(r['n_evict_ids'] > 0 for r in log), 'no eviction transferred'
    for r in log:
        assert 0.0 <= r['w_e'] < 1.0
        assert r['released_keys'] == r['n_evict_ids']


def test_not_commutative():
    """Schedule effect exists: same inputs, C=1 vs C=8 (matched beta) give different S (no commutative reduction)."""
    fasts, slows, bind, hist, _ = _drive(('C4', 'C3'), 64 * SP.MiB, n_steps=16)
    assert not torch.allclose(slows['C4'].S, slows['C3'].S, rtol=1e-4, atol=1e-7)


def test_no_future_and_resume_equivalence(tmp_path):
    from exps.nl03_multirate_01 import run as RN
    torch.manual_seed(0)
    a = RN.Cell('mpdd', 'a', 2711, 8, ('C0', 'C1', 'P', 'C5'), 'test_resume_a', ntasks=2)
    G = RN.bind()
    a.run_task(G); a.run_task(G)
    b = RN.Cell('mpdd', 'a', 2711, 8, ('C0', 'C1', 'P', 'C5'), 'test_resume_b', ntasks=2)
    b.run_task(G)
    p = tmp_path / 'ck.pt'; torch.save(b, p); b = torch.load(p, weights_only=False)
    b.run_task(G)
    for arm in ('C0', 'C1', 'P', 'C5'):
        ka, Wa, *_ = a.state(arm); kb, Wb, *_ = b.state(arm)
        assert torch.equal(ka, kb)
        assert (Wa is None and Wb is None) or torch.equal(Wa, Wb)
        assert a.res[arm][1]['tasks'][0]['auroc'] == b.res[arm][1]['tasks'][0]['auroc']
    steps = [s for s, l, _ in a.causal]
    assert steps == list(range(len(steps)))
    stage = [l for _, l, _ in a.causal]
    assert stage == sorted(stage) and set(stage) == {0, 1}    # each fast step only sees its own task's tokens


def test_parity_canonical_g3_mpdd():
    """C1 with unbounded budget and task-sized minibatches == canonical G3 keys / S (MPDD order a, SELECTED excluded)."""
    from exps.bench_g3 import run as G0
    from exps.nanm_eval_01 import common as U
    from exps.nl03_pressure_00 import core as PC
    G = PC.bind(); PC.use(G, 'mpdd')
    rec = PC.nrec(G, 'mpdd')
    f = CO.Fast(None, False, 2711)
    s = CO.Slow('C1', None, TAU)
    S_ref = 0
    checked = []
    for step, c in enumerate(G0.MPDD_ORDER):
        if f'mpdd/{c}' in SP.NL02_SELECTED:
            continue
        x16 = G.fit_tokens('mpdd', c)
        keys, S_b, n_b, _ = CO.write_candidates(x16, 2711, False)
        ev = f.step(keys, None, step, step)
        s.after_fast(f, ev, S_b, n_b, step)
        st = G.load_norm('mpdd', 2711, c, rec)
        k = len(keys)
        assert torch.equal(keys.cpu(), st['keys'][:k]), c
        assert torch.equal(S_b, st['S1']) and n_b == st['n1'], c
        S_ref = S_ref + st['S1']
        checked.append(c)
        del x16
    assert len(ev['ids']) == 0
    assert torch.allclose(s.S.double().cpu(), S_ref.float().double(), rtol=1e-5, atol=1e-6)
    assert len(checked) == 5
