"""NANM-SYS-01 Phase B: bounded numerical-sensitivity probe of the UNCHANGED k-centre writer (label-free).

Uses only the existing R1 normal FIT tokens (regenerated cache) and the sealed oracle functions. Perturbations are
applied to COPIES of the tokens; R1 artifacts are read-only. No test image, label, mask or metric.

    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_sys.forensics probe
"""
from __future__ import annotations

import json
import math
import time

import torch

from exps.nanm_sys import regen as RG

TASKS = (('visa', 'pcb1'), ('visa', 'cashew'), ('mpdd', 'bracket_black'))   # divergent-in-R2, stable-in-R2, MPDD
SEED = 2711
PERTURB = (('single_element', None), ('frac_1e-5', 1e-5), ('frac_1e-4', 1e-4), ('frac_1e-3', 1e-3))
TIE_REL = (2 ** -20, 2 ** -16, 2 ** -10)


@torch.no_grad()
def kcenter_trace(x, n, seed=SEED, dim=128):
    """Statement-for-statement copy of A.greedy_coreset plus read-only margin logging (verified equal by caller)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = (torch.randn(x.shape[1], dim, generator=g) / math.sqrt(dim)).to(x.device, torch.float32)
    z = x.float() @ P
    idx = torch.empty(n, dtype=torch.long)
    cur = int(torch.randint(len(z), (1,), generator=g))
    first = cur
    mind = torch.full((len(z),), float('inf'), device=x.device)
    top = torch.empty(n, 2, dtype=torch.float64)
    for i in range(n):
        idx[i] = cur
        mind = torch.minimum(mind, ((z - z[cur]) ** 2).sum(1))
        top[i] = mind.topk(2).values.double().cpu()
        cur = int(mind.argmax())
    zn = z.norm(dim=1)
    return idx, top, first, {'min': float(zn.min()), 'median': float(zn.median()), 'max': float(zn.max())}


def flip_ulp(x16, frac, seed):
    """Copy of fp16 tokens with a random subset of elements moved by exactly one fp16 ulp (away from zero or toward)."""
    y = x16.clone()
    flat = y.view(-1)
    g = torch.Generator(device='cpu').manual_seed(seed)
    m = 1 if frac is None else max(1, int(round(frac * flat.numel())))
    pos = torch.randint(flat.numel(), (m,), generator=g).to(flat.device)
    step = (torch.randint(2, (m,), generator=g) * 2 - 1).to(torch.int16).to(flat.device)
    bits = flat.view(torch.int16)                  # +/-1 on the bit pattern = exactly one fp16 ulp (finite, nonzero)
    zero = flat[pos] == 0
    bits[pos] = torch.where(zero, torch.ones_like(step), bits[pos] + step)   # +-0 -> smallest positive subnormal
    if not bool(torch.isfinite(y).all()):
        raise RuntimeError('perturbation produced a non-finite token')
    return y, m


def compare_idx(a, b):
    neq = (a != b).nonzero()
    k = len(a)
    return {'first_divergent_step': int(neq[0]) if len(neq) else None,
            'shared_fraction': len(set(a.tolist()) & set(b.tolist())) / k,
            'shared_fraction_first_half': len(set(a[:k // 2].tolist()) & set(b[:k // 2].tolist())) / (k // 2)}


@torch.no_grad()
def propagate(G, ds, c, fit16, idx_a, idx_b):
    """Bank A vs bank B (same original tokens): residual statistic, single-task geometry, EVAL-normal energies."""
    f32 = fit16.float()
    out = {}
    SA, nA = G.B.residual_stats(f32, idx_a.cuda())
    SB, nB = G.B.residual_stats(f32, idx_b.cuda())
    WA, cA = G.B.whiten(SA, nA)
    WB, cB = G.B.whiten(SB, nB)
    out['S_rel_frobenius'] = float((SA - SB).norm() / SA.norm())
    out['n'] = [nA, nB]
    out['cond'] = [cA, cB]
    out['W_rel_frobenius'] = float((WA - WB).norm() / WA.norm())
    data = G.load_train(ds, c)
    ev = torch.as_tensor(G.roles(ds, data['rows'])['EVAL'])
    q16 = G.A.unit(data['features'][ev].reshape(-1, G.D)).half()          # CPU unit + fp16, as G.load_test
    eA = G.whitened(q16, fit16[idx_a.cuda()].float(), WA.float().cuda())
    eB = G.whitened(q16, fit16[idx_b.cuda()].float(), WB.float().cuda())
    rel = (eA - eB).abs() / eA.abs().clamp_min(1e-30)
    rk = lambda v: torch.empty_like(v).scatter_(0, v.argsort(), torch.arange(len(v), dtype=v.dtype))
    out['eval_images'] = len(ev)
    out['energy'] = {'max_rel': float(rel.max()), 'median_rel': float(rel.median()),
                     'frac_rel_gt_1e-3': float((rel > 1e-3).float().mean()),
                     'frac_bitwise_equal': float((eA == eB).float().mean()),
                     'spearman': float(torch.corrcoef(torch.stack([rk(eA.double()), rk(eB.double())]))[0, 1])}
    return out


def probe(dev):
    RG.numerics()
    G, out = RG.setup()
    t_all = time.monotonic()
    res = {'device': dev, 'tasks': {}}
    for ds, c in TASKS:
        t0 = time.monotonic()
        rec = G.A.rj(out / 'r1_runA' / f'normal_{ds}.json')
        k = rec['sizes']['g3'][G.order(ds).index(c)]
        fit = G.fit_tokens(ds, c)                                        # unchanged oracle loader path (fp16, CUDA)
        idx = G.A.greedy_coreset(fit, k, seed=SEED)                      # unchanged oracle writer
        sealed_here = torch.load(RG.CACHE / 'normal_runA' / f'{ds}_s{SEED}_{c}.pt', weights_only=True)['kc'][:k]
        tr_idx, top, first, zn = kcenter_trace(fit, k)
        G.K.require(torch.equal(idx, tr_idx), 'instrumented trace != oracle greedy_coreset')
        rel_margin = (top[:, 0] - top[:, 1]) / top[:, 0]
        r = {'n_tokens': len(fit), 'k': k, 'first_centre': first, 'equals_normal_runA_prefix': bool(torch.equal(idx, sealed_here)),
             'trace_equals_oracle': True, 'projected_norm': zn,
             'margin_rel': {'min': float(rel_margin[:-1].min()), 'median': float(rel_margin[:-1].median()),
                            'step_of_min': int(rel_margin[:-1].argmin())},
             'exact_ties': int((rel_margin[:-1] == 0).sum()),
             'near_ties': {f'{t:.1e}': int((rel_margin[:-1] < t).sum()) for t in TIE_REL},
             'perturbations': {}}
        for name, frac in PERTURB:
            y, m = flip_ulp(fit, frac, seed=20261120 + len(name))
            j = G.A.greedy_coreset(y, k, seed=SEED)
            d = compare_idx(idx, j)
            mx = max(float((y[i:i + 65536].float() - fit[i:i + 65536].float()).abs().max())
                     for i in range(0, len(fit), 65536))
            d.update({'elements_flipped': m, 'element_fraction': m / fit.numel(), 'max_abs_token_change': mx})
            del y
            torch.cuda.empty_cache()
            if name == 'frac_1e-4':
                d['propagation'] = propagate(G, ds, c, fit, idx, j)
            r['perturbations'][name] = d
        # kernel-path perturbation: identical tokens, CPU (MKL) GEMM/reductions instead of CUDA
        torch.set_num_threads(32)
        jc = G.A.greedy_coreset(fit.cpu(), k, seed=SEED)
        torch.set_num_threads(2)
        r['perturbations']['cpu_kernel_same_tokens'] = compare_idx(idx, jc)
        r['seconds'] = time.monotonic() - t0
        res['tasks'][f'{ds}/{c}'] = r
        G.A.log('PROBE', ds, c, json.dumps({n: (v['first_divergent_step'], round(v['shared_fraction'], 4))
                                           for n, v in r['perturbations'].items()}))
        del fit
        torch.cuda.empty_cache()
    res['seconds'] = time.monotonic() - t_all
    G.A.wj(out / 'forensics' / 'PROBE.json', res)
    RG.stamp(out / 'forensics', 'probe', {'seconds': res['seconds'], 'device': dev})


if __name__ == '__main__':
    probe(RG.guard_env(True))
