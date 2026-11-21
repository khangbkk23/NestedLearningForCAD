"""NANM-R0 Candidate B degeneracy probe (normal-only). Pre-registration: agents/tasks/task_nanm_r0_probe_b.md.

    python -B -m exps.nanm_r0.probe_b
"""
from __future__ import annotations

import json

import numpy as np
import torch

from exps.nanm_r0 import audit as A

TAUS, K, PRIMARY_TAU, THRESH, FRAC = (0.02, 0.05, 0.1), 8, 0.05, 0.98, 0.90


@torch.no_grad()
def run():
    nr = A.rj(A.OUT / 'NORMAL.json')
    mems, owner = [], []
    for t, c in enumerate(A.ORDER):
        m = A.load_norm(c, nr)['fit']
        mems.append(m); owner.append(torch.full((len(m),), t, dtype=torch.int16))
    mem = torch.cat(mems); owner = torch.cat(owner)
    del mems
    per, allv = {}, {f'tau{t}': [] for t in TAUS}
    allv['top1'] = []; cross = []
    chunk = 400_000
    for ti, c in enumerate(A.ORDER):
        q = A.load_norm(c, nr, 'eval')['eval_unit'].cuda()
        best_v = torch.full((len(q), K), -2., device='cuda', dtype=torch.float32)
        best_i = torch.zeros((len(q), K), dtype=torch.long, device='cuda')
        for s in range(0, len(mem), chunk):
            mg = mem[s:s + chunk].cuda()
            for i in range(0, len(q), 2048):
                sim = (q[i:i + 2048] @ mg.T).float()
                v, j = sim.topk(K, dim=1)
                cv = torch.cat([best_v[i:i + 2048], v], 1); ci = torch.cat([best_i[i:i + 2048], j + s], 1)
                vv, jj = cv.topk(K, dim=1)
                best_v[i:i + 2048] = vv; best_i[i:i + 2048] = ci.gather(1, jj)
            del mg
        rec = {'top1_cos': best_v[:, 0].cpu()}
        cross.append((owner[best_i[:, 0].cpu()] != ti).float().mean().item())
        keys = mem[best_i.cpu().flatten()].float().reshape(len(q), K, 768).cuda()
        for tau in TAUS:
            w = torch.softmax(best_v / tau, 1)
            proj = (w[..., None] * keys).sum(1)
            rec[f'tau{tau}'] = torch.nn.functional.cosine_similarity(q.float(), proj, dim=1).cpu()
        per[c] = {k: {'median': float(v.median()), 'p10': float(v.quantile(.1)), 'p90': float(v.quantile(.9)),
                      'frac_ge_098': float((v >= THRESH).float().mean())} for k, v in
                  (('top1', rec['top1_cos']),) + tuple((f'tau{t}', rec[f'tau{t}']) for t in TAUS)}
        per[c]['cross_task_top1_frac'] = cross[-1]
        allv['top1'].append(rec['top1_cos'])
        for t in TAUS:
            allv[f'tau{t}'].append(rec[f'tau{t}'])
        A.log('PROBE', c, json.dumps(per[c]['tau0.05']), 'cross', round(cross[-1], 4))
        del q, keys; torch.cuda.empty_cache()
    pooled = {k: {'median': float(torch.cat(v).median()), 'frac_ge_098': float((torch.cat(v) >= THRESH).float().mean()),
                  'p10': float(torch.cat(v).quantile(.1))} for k, v in allv.items()}
    frac = pooled[f'tau{PRIMARY_TAU}']['frac_ge_098']
    out = {'per_task': per, 'pooled': pooled, 'decision': 'B_TARGET_DEGENERATE' if frac >= FRAC else
           'B_TARGET_NONTRIVIAL', 'primary_frac_ge_098': frac, 'k': K, 'memory_patches': len(mem),
           'prereg_sha256': A.sha(A.ROOT / 'agents/tasks/task_nanm_r0_probe_b.md')}
    A.wj(A.OUT / 'PROBE_B.json', out)
    A.log('PROBE_B', out['decision'], frac)


if __name__ == '__main__':
    run()
