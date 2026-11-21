"""Synthetic CPU parity / invariant tests for nanm.g3 (NANM-RECOVERY-01 Phase 3B).

Two kinds of evidence, kept separate:
  * ORACLE parity  — nanm output vs the surviving oracle code path (exps/bench_g3/run.py) executed on CPU
                     (torch.Tensor.cuda patched to identity). Expected bit-identical (torch.equal).
  * INDEPENDENT    — fp64 brute-force recomputation from the definitions (no oracle code), with tolerances.
No real data, no labels, no caches. Synthetic tokens only.
"""
import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exps.bench_g3 import run as BG
from exps.nanm_r0 import audit as A
from exps.nanm_r2 import run as R2
from exps.nanm_r2b import run as B
from nanm import g3

D = g3.D
SEEDS = (2711, 2712, 2713)


def tokens(n, task, seed=0):
    """Clustered unit-norm fp16 tokens; each task has its own centres (structured, not isotropic)."""
    g = torch.Generator().manual_seed(1000 * task + seed)
    centres = torch.randn(8, D, generator=g) * 3
    x = centres[torch.randint(8, (n,), generator=g)] + torch.randn(n, D, generator=g)
    return F.normalize(x, dim=-1).half()


@pytest.fixture(scope='module')
def tasks():
    return [tokens(n, t) for t, n in enumerate((700, 520, 610))]


@pytest.fixture
def cpu_cuda(monkeypatch):
    """Run the oracle's `.cuda()` code paths on CPU."""
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)


# ---------------------------------------------------------------- constants
def test_byte_constants_match_bench_g3():
    assert g3.KEY_BYTES == BG.KEY == 1536
    assert g3.S_BYTES == BG.EXTRA_G3 == 2_359_304
    assert g3.SHRINK == R2.SHRINK == B.ALPHA == 0.5
    assert g3.QC == BG.QC
    assert g3.task_size(70_000) == 700 and g3.task_size(70_001) == 701


# ---------------------------------------------------------------- ORACLE parity
@pytest.mark.parametrize('seed', SEEDS)
def test_fit_task_equals_bench_g3_normal_composition(tasks, seed, cpu_cuda):
    fit = tasks[0]
    k, n_select = 9, 14                                   # g3 prefix of a larger k-centre run, as in _normal_ds
    kc = A.greedy_coreset(fit, n_select, seed=seed)       # oracle lines, verbatim order
    S1, n1 = B.residual_stats(fit.float(), kc[:k].cuda())
    keys = fit[kc.cuda()].cpu()[:k]
    st = g3.fit_task(fit, k, seed, n_select=n_select)
    assert torch.equal(st.kc, kc) and torch.equal(st.keys, keys)
    assert torch.equal(st.S, S1) and st.n == n1 and st.S.dtype == torch.float64


def test_stage_whitener_equals_bench_g3(tasks):
    states = [g3.fit_task(x, 12, 2711) for x in tasks]
    for l in range(len(states)):
        seen = states[:l + 1]
        W3, c3 = B.whiten(sum(x.S for x in seen), sum(x.n for x in seen))
        W, c = g3.stage_whitener(seen)
        assert torch.equal(W, W3.float()) and c == c3


def test_score_equals_bench_g3_whitened(tasks, cpu_cuda):
    states = [g3.fit_task(x, 12, 2712) for x in tasks]
    W, _ = g3.stage_whitener(states)
    keys = g3.union_keys(states)
    q = torch.cat([tokens(700, 9), tokens(300, 10)])     # > QC rows: exercises chunk boundaries
    assert torch.equal(g3.score(q, keys, W), BG.whitened(q, keys, W))


# ---------------------------------------------------------------- INDEPENDENT checks (fp64, from definitions)
def test_coreset_invariants(tasks):
    x = tasks[1]
    a, b = A.greedy_coreset(x, 40, seed=2711), A.greedy_coreset(x, 40, seed=2711)
    assert torch.equal(a, b)                                            # deterministic
    assert len(set(a.tolist())) == 40                                   # distinct centres
    assert not torch.equal(a, A.greedy_coreset(x, 40, seed=2712))      # seed matters
    # k-centre: covering radius (in the seeded projection) is non-increasing along the prefix
    g = torch.Generator(device='cpu').manual_seed(2711)
    P = torch.randn(D, A.PROJ_DIM, generator=g) / np.sqrt(A.PROJ_DIM)
    z = x.float() @ P
    rad = [float(torch.cdist(z, z[a[:m]]).min(1).values.max()) for m in range(1, 41)]
    assert all(r1 >= r2 - 1e-5 for r1, r2 in zip(rad, rad[1:]))


def test_residual_stats_fp64_bruteforce(tasks):
    x = tasks[2]
    st = g3.fit_task(x, 17, 2713)
    xd = x.double()
    member = torch.zeros(len(x), dtype=torch.bool); member[st.kc] = True
    q, m = xd[~member], xd[st.kc]
    r = q - m[torch.cdist(q, m).argmin(1)]
    S_ref = r.T @ r
    assert st.n == len(x) - 17
    assert torch.allclose(st.S, S_ref, rtol=1e-4, atol=1e-6)


def test_whitener_fp64_definition(tasks):
    states = [g3.fit_task(x, 12, 2711) for x in tasks]
    S = sum(s.S for s in states); n = sum(s.n for s in states)
    C = S / n
    C = 0.5 * C + 0.5 * (torch.trace(C) / D) * torch.eye(D, dtype=torch.float64)
    W64, cond = B.whiten(S, n)
    assert torch.allclose(W64, W64.T, atol=1e-10)
    assert torch.allclose(W64 @ C @ W64, torch.eye(D, dtype=torch.float64), atol=1e-8)
    ev = np.linalg.eigvalsh(C.numpy())
    assert cond == pytest.approx(ev.max() / ev.min(), rel=1e-8)


def test_score_fp64_bruteforce(tasks):
    states = [g3.fit_task(x, 12, 2711) for x in tasks]
    W, _ = g3.stage_whitener(states)
    keys = g3.union_keys(states)
    q = tokens(600, 11)
    Wd = W.double()
    ref = 0.5 * torch.cdist(q.double() @ Wd, keys.double() @ Wd).pow(2).min(1).values
    got = g3.score(q, keys, W)
    assert torch.allclose(got.double(), ref, rtol=1e-3, atol=1e-3 * float(ref.mean()))
    assert bool(torch.isfinite(got).all()) and float(got.min()) >= 0


def test_final_whitener_is_task_order_invariant(tasks):
    """Roadmap §3.3: the slow statistic is a per-prefix sum, so the final state is order-invariant."""
    states = [g3.fit_task(x, 12, 2711) for x in tasks]
    Ws = [g3.stage_whitener([states[i] for i in p])[0] for p in itertools.permutations(range(3))]
    for W in Ws[1:]:
        assert torch.allclose(W, Ws[0], rtol=1e-5, atol=1e-6)


def test_persistent_bytes_and_union_memory(tasks):
    ks = (7, 12, 5)
    states = [g3.fit_task(x, k, 2711) for x, k in zip(tasks, ks)]
    assert g3.union_keys(states).shape == (sum(ks), D)
    assert g3.persistent_bytes(states) == sum(ks) * 1536 + 2_359_304


def test_fit_task_rejects_bad_sizes(tasks):
    with pytest.raises(ValueError):
        g3.fit_task(tasks[0], 0, 2711)
    with pytest.raises(ValueError):
        g3.fit_task(tasks[0], 10, 2711, n_select=5)
