"""G3-FINAL-ROADMAP Task B: does an ordinary per-prefix (batch, refit-from-scratch) estimator reproduce G3 exactly?

Normal-only: R0 FIT tokens, R1 banks, R2 stage metrics, R0 EVAL normal tokens (train/good). No test data.

    python -B -m exps.g3_roadmap.per_prefix_identity

For stages s in {0, 7, 14}:
  (a) incremental G3 reader: R2 stage_metrics[s]['G3'] (running sum S <- S + sum r r^T, task by task);
  (b) batch per-prefix reader: residuals of ALL tasks <= s recomputed from scratch against the stored banks
      (same assignment rule and chunking as R2: Euclidean NN, members excluded, chunk 8192), stacked, one second
      moment, same shrinkage and inverse square root;
  (c) task-free energies of EVAL normal tokens of 3 categories under (a) and (b) against the union bank.
Writes results/hope_cad/g3_roadmap/PER_PREFIX_IDENTITY.json.
"""
from __future__ import annotations

import time

import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1
from exps.nanm_r2 import run as R2
from exps.nanm_r2b import run as B

OUT = A.ROOT / 'results/hope_cad/g3_roadmap'
STAGES = (0, 7, 14)
CATS = ('bottle', 'grid', 'zipper')


@torch.no_grad()
def main():
    t0 = time.monotonic()
    nr = A.rj(A.OUT / 'NORMAL.json')
    st = torch.load(R2.CACHE / 'stage_metrics.pt', weights_only=True)
    banks = []
    for c in A.ORDER:
        fit = A.load_norm(c, nr)['fit']
        kc = torch.load(R1.CACHE / f'mv_{c}.pt', weights_only=True)['kc_idx'][:R1.n_of(0.01, len(fit))]
        banks.append(fit[kc].clone()); del fit

    def refit(s):
        """Batch per-prefix estimator: recompute every residual of tasks <= s from scratch (fp32, R2 chunking)."""
        C = torch.zeros(768, 768, dtype=torch.float64, device='cuda'); n = 0
        for c in A.ORDER[:s + 1]:
            fit = A.load_norm(c, nr)['fit'].cuda().float()
            kc = torch.load(R1.CACHE / f'mv_{c}.pt', weights_only=True)['kc_idx'][:R1.n_of(0.01, len(fit))].cuda()
            member = torch.zeros(len(fit), dtype=torch.bool, device='cuda'); member[kc] = True
            mem = fit[kc]
            for i in range(0, len(fit), 8192):
                x = fit[i:i + 8192][~member[i:i + 8192]]
                r = x - mem[R2.sqdist(x, mem).argmin(1)]
                C += (r.T @ r).double(); n += len(r)
            del fit, mem
        return C.cpu(), n

    out = {}
    for s in STAGES:
        C, n_res = refit(s)
        Wb, _ = B.whiten(C, n_res)
        Wi = st[s]['G3'].double()
        keys = torch.cat(banks[:s + 1]).cuda().float()
        e_dev = []
        for c in CATS[: (1 if s == 0 else 3)]:
            q = A.load_norm(c, nr, 'eval')['eval_unit'].cuda().float()
            ei = 0.5 * R2.sqdist(q @ Wi.float().cuda(), keys @ Wi.float().cuda()).min(1).values
            eb = 0.5 * R2.sqdist(q @ Wb.float().cuda(), keys @ Wb.float().cuda()).min(1).values
            e_dev.append([float(((ei - eb).abs() / ei.abs().clamp_min(1e-12)).max()),
                          float((ei - eb).abs().max() / ei.abs().max()), bool(torch.equal(ei, eb))])
        out[s] = {'n_residuals': n_res,
                  'W_rel_frobenius': float((Wb - Wi).norm() / Wi.norm()),
                  'energy_max_patchwise_rel_diff': max(e[0] for e in e_dev),
                  'energy_max_abs_diff_over_max_energy': max(e[1] for e in e_dev),
                  'energies_bitwise_equal': [e[2] for e in e_dev],
                  'eval_categories': list(CATS[: (1 if s == 0 else 3)]),
                  'note': 'incremental W stored fp32 by R2; batch W fp64 cast to fp32 for scoring'}
        A.log('STAGE', s, out[s])
    A.wj(OUT / 'PER_PREFIX_IDENTITY.json', {'stages': out, 'seconds': time.monotonic() - t0,
                                             'inputs': {'r2_stage_metrics': A.sha(R2.CACHE / 'stage_metrics.pt'),
                                                        'r1_coresets': A.sha(R1.OUT / 'CORESETS.json')}})


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
