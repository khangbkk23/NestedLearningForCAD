"""RIGEL-Q8-01 S2 (PREREG.md): read-out smoothing sigma and image score. DEV selection on the GG-01 V1 MVTec AD 2
test_public grids (corrected masks), then a one-shot FULL confirmation on the locked E@8 grids (order a, seed 2711).

    python -B -m exps.rigel_q8.s2_readout dev
    python -B -m exps.rigel_q8.s2_readout full      # only after RO_DEV_PASS
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exps.rigel_q8.common import CACHE, OUT, log, prereg_sha, wj

SIGMAS = [2, 3, 4, 6, 8]
SIGMAS_A1 = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]          # PREREG_AMENDMENT_1.md
SCORES = ['top0.1', 'top0.5', 'top1', 'top2', 'top5', 'max']
V1_SIGMA, V1_SCORE = 4, 'top1'


def maps_sigma(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """rigel_cad.readout.anomaly_maps with a variable sigma (kernel 2*int(4*sigma+.5)+1); identical at sigma 4."""
    from rigel_cad.config import INPUT_SIZE
    ks = 2 * int(4 * sigma + .5) + 1
    ax = torch.arange(ks, dtype=torch.float32) - ks // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    up = F.interpolate(grid[:, None].float(), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
    up = F.conv2d(up, g.view(1, 1, ks, 1), padding=(ks // 2, 0))
    m = F.conv2d(up, g.view(1, 1, 1, ks), padding=(0, ks // 2))[:, 0]
    return m - min(0., float(m.min()))


def image_score(maps: torch.Tensor, kind: str) -> np.ndarray:
    f = maps.flatten(1)
    if kind == 'max':
        return f.max(1).values.numpy()
    frac = float(kind[3:]) / 100
    k = max(1, int(frac * f.shape[1]))
    return f.topk(k, dim=1).values.mean(1).numpy()


def eval_readout(grid_last, hi, cat, bw, sigmas=SIGMAS):
    """{sigma: {'ap', 'auroc': {score}, 'boot_ap', 'boot_auroc': {score}}} for one category (final stage)."""
    from sklearn.metrics import roc_auc_score
    from rigel_cad import metrics as M
    y = cat.y
    out = {}
    for s in sigmas:
        mp_ = maps_sigma(grid_last, s)
        h = M.image_hists(mp_, cat.hws, cat.masks, 0., hi)
        P, T = bw @ h[0], bw @ h[1]
        r = {'ap': M.binned_ap(h[0].sum(0), h[1].sum(0)),
             'boot_ap': np.array([M.binned_ap(P[i], T[i]) for i in range(len(P))], np.float32),
             'auroc': {}, 'boot_auroc': {}}
        for k in SCORES:
            sc = image_score(mp_, k)
            r['auroc'][k] = float(roc_auc_score(y, sc))
            r['boot_auroc'][k] = np.array([roc_auc_score(y, sc, sample_weight=w) if (w * y).sum() > 0 and
                                           (w * (1 - y)).sum() > 0 else np.nan for w in bw], np.float32)
        out[s] = r
    return out


# ---------------------------------------------------------------------------------------------- DEV (MVTec AD 2)
def _dev_task(args):
    s, j, sigmas = args
    from exps.rigel_gg01 import dev as GD
    from rigel_cad import metrics as M
    rows = GD.test_rows(s)
    mm = GD.mask_map(s)

    class Cat:
        pass
    cat = Cat()
    cat.hws = [M.native_hw(GD.AD2_ROOT / r) for r in rows]
    cat.masks = [M.native_mask(mm[Path(r).stem] if Path(r).parts[2] == 'bad' else None, hw)
                 for r, hw in zip(rows, cat.hws)]
    assert all((Path(r).parts[2] == 'bad') == bool(m.any()) for r, m in zip(rows, cat.masks))
    cat.y = np.array([int(m.any()) for m in cat.masks])
    d = torch.load(GD.CACHE / 'dev' / 'V1' / f'{s}.pt', weights_only=True)
    assert d['rows'] == rows
    t0 = time.time()
    r = eval_readout(d['grid'][-1], float(d['grid'].max()) + 1e-12, cat, M.boot_weights(len(rows), j), sigmas)
    return s, r, time.time() - t0


def macro(res, cats, sig, key, score=None):
    if key == 'ap':
        pt = np.mean([res[c][sig]['ap'] for c in cats])
        bt = np.mean([res[c][sig]['boot_ap'] for c in cats], axis=0)
    else:
        pt = np.mean([res[c][sig]['auroc'][score] for c in cats])
        bt = np.nanmean([res[c][sig]['boot_auroc'][score] for c in cats], axis=0)
    return float(pt), bt


def delta(res, cats, a, b, key):
    """a, b = (sigma, score); paired delta and CI95 of macro metric."""
    pa, ba = macro(res, cats, a[0], key, a[1])
    pb, bb = macro(res, cats, b[0], key, b[1])
    d = ba - bb
    return {'delta': pa - pb, 'ci95': [float(v) for v in np.nanpercentile(d, [2.5, 97.5])], 'a': pa, 'b': pb}


def dev(sigmas=SIGMAS, out_name='S2_DEV.json'):
    sha0 = prereg_sha()
    from exps.rigel_gg01 import dev as GD
    scens = GD.scenarios()
    res, secs = {}, {}
    with mp.get_context('spawn').Pool(2) as pool:
        for s, r, dt in pool.imap_unordered(_dev_task, [(s, j, sigmas) for j, s in enumerate(scens)]):
            res[s], secs[s] = r, dt
            log('S2 DEV', s, {sg: round(r[sg]['ap'], 4) for sg in sigmas}, 'auroc@4', {k: round(v, 3) for k, v in r[4]['auroc'].items()}, '%.0fs' % dt)
    # parity: sigma 4 / top 1 % must reproduce the GG-01 V1 METRICS (final stage) exactly
    import json
    gm = json.loads((Path(__file__).resolve().parents[2] / 'results/hope_cad/rigel_gg01/dev/METRICS.json').read_text())
    T = len(scens)
    parity = {s: (res[s][4]['ap'] == gm['cells'][s]['V1'][str(T - 1)]['ap'],
                  abs(res[s][4]['auroc']['top1'] - gm['cells'][s]['V1'][str(T - 1)]['i_auroc']) < 1e-12) for s in scens}
    assert all(a and b for a, b in parity.values()), parity
    ap = {sg: macro(res, scens, sg, 'ap')[0] for sg in sigmas}
    sig_best = max(sigmas, key=lambda sg: ap[sg])
    da = delta(res, scens, (sig_best, None), (V1_SIGMA, None), 'ap')
    sig_star = sig_best if sig_best != V1_SIGMA and da['delta'] >= 0.003 and da['ci95'][0] > 0 else V1_SIGMA
    au = {k: macro(res, scens, sig_star, 'auroc', k)[0] for k in SCORES}
    sc_best = max(SCORES, key=lambda k: au[k])
    db = delta(res, scens, (sig_star, sc_best), (sig_star, V1_SCORE), 'auroc')
    sc_star = sc_best if sc_best != V1_SCORE and db['delta'] >= 0.005 and db['ci95'][0] > 0 else V1_SCORE
    verdict = 'RO_DEV_PASS' if (sig_star, sc_star) != (V1_SIGMA, V1_SCORE) else 'RO_NO_CHANGE'
    out = {'verdict': verdict, 'sigma_star': sig_star, 'score_star': sc_star,
           'step_a': {'macro_ap_by_sigma': ap, 'best': sig_best, 'delta_vs_v1': da},
           'step_b': {'macro_auroc_by_score_at_sigma_star': au, 'best': sc_best, 'delta_vs_v1': db},
           'all_macro_auroc': {sg: {k: macro(res, scens, sg, 'auroc', k)[0] for k in SCORES} for sg in sigmas},
           'v1_combined_delta': {'ap': delta(res, scens, (sig_star, sc_star), (V1_SIGMA, V1_SCORE), 'ap'),
                                 'auroc': delta(res, scens, (sig_star, sc_star), (V1_SIGMA, V1_SCORE), 'auroc')},
           'per_scenario': {s: {sg: {'ap': res[s][sg]['ap'], 'auroc': res[s][sg]['auroc']} for sg in sigmas} for s in scens},
           'sigmas': sigmas,
           'parity_with_gg01_v1': {s: list(v) for s, v in parity.items()}, 'seconds': secs,
           'prereg_sha256': sha0, 'time': time.strftime('%FT%T')}
    assert prereg_sha() == sha0
    wj(OUT / out_name, out)
    log('S2 VERDICT', verdict, 'sigma*', sig_star, 'score*', sc_star, 'step a', da, 'step b', db)


# ---------------------------------------------------------------------------------------------- FULL (locked E@8 grids)
BENCH = Path('/mnt/k/NLCAD/cache/nl03_multirate_01/bench')
DATA = {'mvtec': 'data/mvtec', 'visa': 'data/visa', 'mpdd': 'data/mpdd'}


def _full_task(args):
    ds, c, j, sig, sc = args
    from rigel_cad import data as D
    from rigel_cad import metrics as M
    from exps.rigel_q8.common import ROOT
    root = ROOT / DATA[ds]
    _, test = D.category_split(ds, root, c)
    cat = M.Category(root, test)
    d = torch.load(BENCH / f'{ds}_a2711' / 'E' / f'{c}.pt', weights_only=False)
    assert list(d['rows']) == cat.rows, (ds, c)
    bw = M.boot_weights(len(cat.rows), j)
    r = eval_readout(d['grid'][-1], float(d['grid'].max()) + 1e-12, cat, bw, sorted({V1_SIGMA, sig}))
    keep = {s: r[s] for s in {V1_SIGMA, sig}}
    for s in keep:
        keep[s] = {'ap': keep[s]['ap'], 'boot_ap': keep[s]['boot_ap'],
                   'auroc': {k: keep[s]['auroc'][k] for k in {V1_SCORE, sc}},
                   'boot_auroc': {k: keep[s]['boot_auroc'][k] for k in {V1_SCORE, sc}}}
    return ds, c, keep


def full():
    import json
    sha0 = prereg_sha()
    a1 = OUT / 'S2_DEV_A1.json'                       # amendment 1 supersedes the original selection
    dv = json.loads((a1 if a1.exists() else OUT / 'S2_DEV.json').read_text())
    assert dv['verdict'] == 'RO_DEV_PASS', 'FULL runs only after RO_DEV_PASS'
    sig, sc = dv['sigma_star'], dv['score_star']
    from rigel_cad import data as D
    jobs = [(ds, c, j, sig, sc) for ds in DATA for j, c in enumerate(D.stream_orders(ds)['a'])]
    res = {ds: {} for ds in DATA}
    with mp.get_context('spawn').Pool(2) as pool:
        for ds, c, r in pool.imap_unordered(_full_task, jobs):
            res[ds][c] = r
            log('S2 FULL', ds, c, {s: round(v['ap'], 4) for s, v in r.items()})
    adopted = 'ap' if sig != V1_SIGMA else 'auroc'
    other = 'auroc' if adopted == 'ap' else 'ap'
    per = {}
    for ds in DATA:
        cats = list(D.stream_orders(ds)['a'])
        per[ds] = {'ap': delta(res[ds], cats, (sig, sc), (V1_SIGMA, V1_SCORE), 'ap'),
                   'auroc': delta(res[ds], cats, (sig, sc), (V1_SIGMA, V1_SCORE), 'auroc')}
    ok1 = all(per[ds][adopted]['delta'] >= 0 for ds in DATA) and sum(per[ds][adopted]['ci95'][0] > 0 for ds in DATA) >= 2
    ok2 = all(per[ds][other]['delta'] >= -0.005 for ds in DATA)
    verdict = 'RO_CONFIRMED' if ok1 and ok2 else 'RO_NOT_CONFIRMED'
    out = {'verdict': verdict, 'sigma_star': sig, 'score_star': sc, 'adopted_metric': adopted, 'per_dataset': per,
           'per_category': {ds: {c: {s: {'ap': v['ap'], 'auroc': v['auroc']} for s, v in r.items()}
                                 for c, r in res[ds].items()} for ds in DATA},
           'prereg_sha256': sha0, 'time': time.strftime('%FT%T')}
    wj(OUT / 'S2_FULL.json', out)
    log('S2 FULL VERDICT', verdict, per)


if __name__ == '__main__':
    {'dev': dev, 'dev_a1': lambda: dev(SIGMAS_A1, 'S2_DEV_A1.json'), 'full': full}[sys.argv[1]]()
