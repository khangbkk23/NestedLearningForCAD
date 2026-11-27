"""RIGEL-Q8-01 S1 (PREREG.md): int8 key error, label-free. GO/NO-GO for S3.

    python -B -m exps.rigel_q8.s1_int8
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.rigel_gg01 import dev as GD
from exps.rigel_q8 import q8
from exps.rigel_q8.common import OUT, PREREG, log, prereg_sha, wj
from rigel_cad import ops


@torch.no_grad()
def nearest(q16, keys, W, chunk=512):
    mw = keys @ W
    e, idx = [], []
    for i in range(0, len(q16), chunk):
        d = ops.sqdist(q16[i:i + chunk].float() @ W, mw)
        v, j = d.min(1)
        e.append(0.5 * v)
        idx.append(j)
    return torch.cat(e), torch.cat(idx)


@torch.no_grad()
def main():
    sha0 = prereg_sha()
    GD.guard()
    d = torch.load(GD.CACHE / 'dev' / 'states' / 'V1.pt', weights_only=True)
    K16 = d['keys'].cuda()
    W = ops.whitener(d['S'], d['n'])[0].float().cuda()
    q, s = q8.quantize(K16)
    Kh = q8.dequantize(q, s)
    K = K16.float()
    q2, s2 = q8.quantize(Kh)
    idem = bool(torch.equal(q2, q) and torch.equal(s2, s))
    cos = torch.nn.functional.cosine_similarity(K, Kh, dim=1)
    rel_l2 = (K - Kh).norm(dim=1) / K.norm(dim=1)
    werr = ((K - Kh) @ W).norm(dim=1)
    # median nearest-other-key whitened distance among the stored keys
    mw = K @ W
    dd = ops.sqdist(mw, mw)
    dd.fill_diagonal_(float('inf'))
    nn_w = dd.min(1).values.sqrt()
    per, all_rel, all_chg = {}, [], []
    for l, sc in enumerate(GD.scenarios()):
        from rigel_cad import data as DATA
        x = GD.tokens(f'ad2/{sc}_train', DATA.ad2_files(GD.AD2_ROOT, sc, 'train')).reshape(-1, q8.D)
        g = torch.Generator().manual_seed(20261126 + l)
        smp = x[torch.randperm(len(x), generator=g)[:4096]].cuda()
        del x
        e0, i0 = nearest(smp, K, W)
        e1, i1 = nearest(smp, Kh, W)
        rel = ((e1 - e0).abs() / e0.clamp_min(1e-12)).cpu().numpy()
        chg = (i0 != i1).float().cpu().numpy()
        per[sc] = {'median_rel_dE': float(np.median(rel)), 'p99_rel_dE': float(np.percentile(rel, 99)),
                   'nearest_changed': float(chg.mean()), 'mean_E_fp16': float(e0.mean()), 'mean_E_int8': float(e1.mean())}
        all_rel.append(rel)
        all_chg.append(chg)
        log('S1', sc, per[sc])
    rel = np.concatenate(all_rel)
    res = {
        'keys': len(K), 'idempotent': idem,
        'cos_min': float(cos.min()), 'cos_median': float(cos.median()),
        'rel_l2_median': float(rel_l2.median()), 'rel_l2_max': float(rel_l2.max()),
        'whitened_err_median': float(werr.median()), 'nn_whitened_dist_median': float(nn_w.median()),
        'whitened_err_over_nn_median': float((werr / nn_w).median()),
        'median_rel_dE': float(np.median(rel)), 'p99_rel_dE': float(np.percentile(rel, 99)),
        'nearest_changed': float(np.concatenate(all_chg).mean()), 'per_scenario': per,
    }
    go = res['median_rel_dE'] <= 0.01 and res['p99_rel_dE'] <= 0.05 and idem
    res['verdict'] = 'Q8_GO' if go else 'Q8_NO_GO'
    res['prereg_sha256'] = sha0
    res['time'] = time.strftime('%FT%T')
    assert prereg_sha() == sha0
    wj(OUT / 'S1_INT8.json', res)
    log('S1 VERDICT', res['verdict'], {k: res[k] for k in ('median_rel_dE', 'p99_rel_dE', 'nearest_changed', 'cos_min')})


if __name__ == '__main__':
    main()
