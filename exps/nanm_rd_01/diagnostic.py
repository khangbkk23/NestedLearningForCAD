"""NANM-RD-01A: normal-only image-readout diagnostic (registration agents/tasks/task_nanm_rd_01.md §2-3). No labels.

    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_rd_01.diagnostic maps       # GPU: cache + hash G3 (nanm) and A0b (oracle) maps, once
    python -B -m exps.nanm_rd_01.diagnostic analyze    # CPU: registered readouts R0-R3 and measures
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from exps.nanm_sys import final as F
from exps.nanm_sys import regen as RG

SEED, DSETS = 2711, ('mpdd', 'visa')
K1, K5 = int(0.01 * 392 * 392), int(0.05 * 392 * 392)
BOOT_B, BOOT_SEED = 2000, 20261120


def out():
    o = F.out().parent.parent / 'nanm_rd_01' / 'rd01a'
    o.mkdir(parents=True, exist_ok=True)
    return o


@torch.no_grad()
def maps():
    from nanm import G3Config, NANMSystem
    dev = F.guard(); RG.numerics(); G, _ = RG.setup()
    man, t0 = {}, time.monotonic()
    for ds in DSETS:
        rec = {ds: G.A.rj(G.ROOT / 'results/hope_cad/nanm_sys/regen/r1_runA' / f'normal_{ds}.json')}
        sz, ords = rec[ds]['sizes'], G.order(ds)
        nm = NANMSystem.load_state(F.CK / 'A' / f'{ds}_G3_s{SEED}', cfg=G3Config(seed=SEED), device='cuda')
        G.K.require(len(nm.state.keys) == len(ords), 'final G3 state')
        a0b = [G.load_norm(ds, SEED, c, rec)['keys'][:sz['a0b'][t]].cuda().float() for t, c in enumerate(ords)]
        for c in ords:
            data = G.load_train(ds, c)
            roles = G.roles(ds, data['rows'])
            for role in ('CREDIT', 'EVAL'):
                idx = roles[role]
                tok = data['features'][torch.as_tensor(idx)]
                g3 = nm.predict(tok)['map392']
                q16 = G.A.unit(tok.reshape(-1, G.D)).half()
                e = torch.stack([G.euclid(q16, k) for k in a0b]).amin(0).reshape(len(idx), 28, 28)
                a0 = G.R2.maps_of(e)
                p = out() / 'maps' / f'{ds}_{c}_{role}.pt'
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'rows': [data['rows'][i]['relative_path'] for i in idx], 'G3': g3, 'A0b': a0}, p)
                man[p.name] = {'sha256': G.A.sha(p), 'n': len(idx)}
            G.A.log('MAPS', ds, c)
    G.A.wj(out() / 'MAPS_MANIFEST.json', {'files': man, 'seconds': time.monotonic() - t0, 'device': dev,
                                          'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


def readouts(m):
    """Registered R0-R3 on [N,392,392] maps (identical tensor for all readouts)."""
    f = m.flatten(1)
    r2 = torch.cat([torch.quantile(f[i:i + 32], 0.99, dim=1) for i in range(0, len(f), 32)])
    conc = f.topk(K1, dim=1).values.sum(1) / f.sum(1)
    return {'R0': f.topk(K1, dim=1).values.mean(1), 'R1': f.amax(1), 'R2': r2,
            'R3': f.topk(K5, dim=1).values.mean(1)}, conc


def _stats(x):
    x = np.asarray(x, np.float64)
    med = np.median(x)
    q = np.quantile(x, [.25, .75, .9])
    return {'robust_cv': float(np.median(np.abs(x - med)) / med), 'iqr_over_median': float((q[1] - q[0]) / med),
            'q90_over_median': float(q[2] / med), 'max_over_median': float(x.max() / med), 'n': len(x)}


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze():
    G, _ = RG.setup()
    man = G.A.rj(out() / 'MAPS_MANIFEST.json')['files']
    res = {}
    rng = np.random.default_rng(BOOT_SEED)
    for ds in DSETS:
        ords = G.order(ds)
        S = {}                                          # S[method][readout][task][role] -> scores
        conc = {}
        for c in ords:
            for role in ('CREDIT', 'EVAL'):
                p = out() / 'maps' / f'{ds}_{c}_{role}.pt'
                G.K.require(G.A.sha(p) == man[p.name]['sha256'], 'map cache hash')
                d = torch.load(p, weights_only=True)
                for meth in ('G3', 'A0b'):
                    r, cc = readouts(d[meth])
                    for k, v in r.items():
                        S.setdefault(meth, {}).setdefault(k, {}).setdefault(c, {})[role] = v.double().numpy()
                    if role == 'EVAL':
                        conc.setdefault(meth, {})[c] = float(cc.mean())
        R = {}
        for meth in ('G3', 'A0b'):
            for k in ('R0', 'R1', 'R2', 'R3'):
                per = {}
                tau_pool = np.quantile(np.concatenate([S[meth][k][c]['CREDIT'] for c in ords]), .95)
                for c in ords:
                    ev, cr = S[meth][k][c]['EVAL'], S[meth][k][c]['CREDIT']
                    per[c] = dict(_stats(ev), fpr_task_tau=float((ev > np.quantile(cr, .95)).mean()),
                                  fpr_pooled_tau=float((ev > tau_pool).mean()),
                                  spearman_vs_R0=_spearman(S[meth]['R0'][c]['EVAL'], ev))
                mac = {m: float(np.mean([per[c][m] for c in ords])) for m in per[ords[0]] if m != 'n'}
                mac['pooled_fpr_range'] = float(max(per[c]['fpr_pooled_tau'] for c in ords) -
                                                min(per[c]['fpr_pooled_tau'] for c in ords))
                R[f'{meth}|{k}'] = {'macro': mac, 'per_task': per}
        # task-stratified image bootstrap: macro robust CV difference G3-A0b per readout and interaction vs R0
        boot = {k: [] for k in ('R0', 'R1', 'R2', 'R3')}
        for _ in range(BOOT_B):
            pick = {c: rng.integers(0, len(S['G3']['R0'][c]['EVAL']), len(S['G3']['R0'][c]['EVAL'])) for c in ords}
            for k in boot:
                cv = lambda meth: np.mean([_stats(S[meth][k][c]['EVAL'][pick[c]])['robust_cv'] for c in ords])
                boot[k].append(cv('G3') - cv('A0b'))
        ci = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boot.items()}
        inter = {k: [float(np.quantile(np.asarray(boot[k]) - np.asarray(boot['R0']), q)) for q in (.025, .975)]
                 for k in ('R1', 'R2', 'R3')}
        res[ds] = {'readouts': R, 'concentration_top1pct_mass': {m: {'macro': float(np.mean(list(v.values()))),
                                                                     'per_task': v} for m, v in conc.items()},
                   'robust_cv_G3_minus_A0b_ci95': ci, 'interaction_robust_cv_ci95': inter}
    G.A.wj(out() / 'RD01A_RESULT.json', res)
    for ds in DSETS:
        print(ds)
        for key, v in res[ds]['readouts'].items():
            m = v['macro']
            print(f"  {key:7s} rCV {m['robust_cv']:.4f} IQR/med {m['iqr_over_median']:.4f} q90/med {m['q90_over_median']:.4f} "
                  f"max/med {m['max_over_median']:.4f} FPRtask {m['fpr_task_tau']:.3f} FPRpool {m['fpr_pooled_tau']:.3f} "
                  f"range {m['pooled_fpr_range']:.3f} rho {m['spearman_vs_R0']:.4f}")
        print('  conc', {m: round(v['macro'], 4) for m, v in res[ds]['concentration_top1pct_mass'].items()})
        print('  dCV CI', res[ds]['robust_cv_G3_minus_A0b_ci95'], 'inter', res[ds]['interaction_robust_cv_ci95'])


if __name__ == '__main__':
    {'maps': maps, 'analyze': analyze}[sys.argv[1]]()
