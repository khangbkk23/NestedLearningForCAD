"""Task 1 v2 (protocol Amendment 1): privileged in-sample rank-16 representational reference on DESIGN HOLDOUT.
Imports exps/nanm_nl_02_forensics/run.py unchanged. Writes TASK1_V2.json (create-exclusive)."""
from __future__ import annotations

import json
import math
import resource
import sys
import time

import torch

from exps.nanm_nl_02 import core as C
from exps.nanm_nl_02 import run as R
from exps.nanm_nl_02_forensics import run as F

STEPS, LR = 400, 3e-3


def fit(cx, tri, tei, seed, track=False):
    P, Q = F.leaf(*C.init_phi(seed))
    opt = torch.optim.Adam([P, Q], lr=LR)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    I0 = torch.zeros(F.D, C.RANK, dtype=F.F64)
    c0_tr = float(cx.hold(I0, I0, tri)); c0_te = float(cx.hold(I0, I0, tei)) if tei is not None else None
    tr, path = [], []
    for s in range(STEPS + 1):
        if track and s % 10 == 0:
            with torch.no_grad():
                path.append((s, c0_te - float(cx.hold(P, Q, tei))))
        if s == STEPS:
            break
        L = cx.hold(P, Q, tri)
        opt.zero_grad(); L.backward()
        if s % 10 == 0 or s == STEPS - 1:
            tr.append((s, c0_tr - float(L.detach())))
        opt.step(); sch.step()
    P, Q = P.detach(), Q.detach()
    with torch.no_grad():
        g_in = c0_tr - float(cx.hold(P, Q, tri))
    last = [v for s, v in tr if s >= int(0.9 * STEPS)]
    out = {'seed': seed, 'gain_in': g_in, 'last10pct_gain_change': last[-1] - last[0], 'curve': tr[::4], **F.phi_stats(P, Q)}
    if track:
        b = max(path, key=lambda x: x[1])
        out.update({'heldout_path': path, 'heldout_best': b[1], 'heldout_best_step': b[0], 'heldout_final': path[-1][1]})
    return out


def main():
    F.OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    proto_sha = R.fsha(F.PROTO); lock0 = R.fsha(R.RUN / 'PREREG_LOCK.json'); st0 = R.storage(); t0 = time.time()
    au, U, streams, ctxs = F.setup()
    allidx = None
    cells, rob = {}, {}
    for (n, s, t), cx in ctxs.items():
        allidx = torch.arange(len(cx.himg))
        r = fit(cx, allidx, None, s); r['task'] = cx.task
        cells[f'{n}|{s}|{t}'] = r
        F.log('task1v2', n, s, t, cx.task, 'R_in %.2f  last10%% %.3f  cond %.1f' % (r['gain_in'], r['last10pct_gain_change'], r['cond_A']))
    for t in range(4):
        cx = ctxs[('a', R.SEEDS[0], t)]; allidx = torch.arange(len(cx.himg))
        ids = sorted(set(cx.himg.tolist()))
        folds = []
        for f in (0, 1):
            m = torch.isin(cx.himg, torch.tensor(ids[:4] if f == 0 else ids[4:]))
            folds.append(fit(cx, m.nonzero().squeeze(1), (~m).nonzero().squeeze(1), R.SEEDS[0], track=True))
        rob[t] = {'seeds': [fit(cx, allidx, None, R.SEEDS[0] + o)['gain_in'] for o in (101, 202)], 'crossfit_oracle_stop': folds}
        F.log('task1v2 robustness t', t, 'seeds', [round(x, 2) for x in rob[t]['seeds']],
              'cf best', [(round(f_['heldout_best'], 2), f_['heldout_best_step']) for f_ in folds],
              'cf final', [round(f_['heldout_final'], 2) for f_ in folds])
    files, ok = F.opened_ok(au); assert ok
    meta = {'phase': 'task1_v2', 'protocol_sha256': proto_sha, 'prereg_lock_sha256': lock0,
            'prereg_lock_unchanged': R.fsha(R.RUN / 'PREREG_LOCK.json') == lock0,
            'code_sha256': R.fsha(R.ROOT / 'exps/nanm_nl_02_forensics/task1_v2.py'),
            'forensics_run_sha256': R.fsha(R.ROOT / 'exps/nanm_nl_02_forensics/run.py'),
            'opened_files': files, 'selected_accessed': False, 'U': U, 'gate': 0.1 * U, 'wall_s': time.time() - t0,
            'storage_before': st0, 'storage_after': R.storage(),
            'peak_rss_GB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            'c0_c0q': {f'{n}|{s}|{t}': (cx.c0, cx.c0q, cx.task) for (n, s, t), cx in ctxs.items()}}
    R.wjson(F.OUT / 'TASK1_V2.json', {'meta': meta, 'result': {'cells': cells, 'robustness_a2711': rob}}, exclusive=True)
    F.log('done task1_v2 %.0fs' % meta['wall_s'])


if __name__ == '__main__':
    main()
