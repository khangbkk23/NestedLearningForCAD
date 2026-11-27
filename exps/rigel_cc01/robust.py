"""RIGEL-CC-01 §7 conditional robustness: selected candidate vs V1 over orders {a,b,c} x seeds {2711,2712,2713}.

    python -B -m exps.rigel_cc01.robust            # aggregate -> ROBUST.json + outcome classification
"""
from __future__ import annotations

import json
import time

import numpy as np

from exps.rigel_cc01.run import DSETS, OUT, ROOT, sha, spec_ok, summary, tag_of, wj, log

CELLS = [(o, s) for o in 'abc' for s in (2711, 2712, 2713)]
G3E = ROOT / 'results/hope_cad/g3e_paper_01/cells'
MR = ROOT / 'results/hope_cad/nl03_multirate_01/bench'


def locked_v1(ds, o, s):
    """(path, arm key) of the locked V1 (= E@8) METRICS for one cell (SPEC §7)."""
    if s == 2711 and (MR / f'METRICS_{ds}_{o}2711.json').exists():
        return MR / f'METRICS_{ds}_{o}2711.json', 'E'
    if (ds, o, s) == ('mpdd', 'c', 2711):
        return G3E / 'METRICS_A1_dinov2_mpdd_c2711.json', 'E@8'
    if o == 'a':
        return G3E / f'METRICS_A3_dinov2_{ds}_a{s}.json', 'E@8'
    return G3E / f'METRICS_A2_dinov2_{ds}_{o}{s}.json', 'E@8'


def main():
    sha0 = spec_ok()
    gate = json.loads((OUT / 'GATE.json').read_text())
    cand = gate['selected']
    assert cand, 'no candidate selected'
    per, parity, negcls = {}, {}, {}
    for ds in DSETS:
        per[ds] = {}
        for o, s in CELLS:
            tag = tag_of(ds, o, s)
            st = json.loads((OUT / f'STREAM_{tag}.json').read_text())
            me = json.loads((OUT / f'METRICS_{tag}.json').read_text())
            order, T = st['order'], len(st['order'])
            Sc, Sv = summary(order, me['cells'], cand), summary(order, me['cells'], 'V1')
            lp, key = locked_v1(ds, o, s)
            lm = json.loads(lp.read_text())['cells']
            lock_ap = float(np.mean([lm[t][key][str(T - 1)]['ap'] for t in order]))
            parity[tag] = {'file': str(lp.relative_to(ROOT)), 'sha256': sha(lp), 'locked_final_ap': lock_ap,
                           'recomputed_final_ap': Sv['pixel_ap'], 'equal': lock_ap == Sv['pixel_ap']}
            per[ds][f'{o}{s}'] = {'dAP': Sc['pixel_ap'] - Sv['pixel_ap'], 'dAUROC': Sc['image_auroc'] - Sv['image_auroc'],
                                  'dFM': Sc['forgetting_ap'] - Sv['forgetting_ap'], 'dACQ': Sc['acquisition_ap'] - Sv['acquisition_ap'],
                                  'v1_ap': Sv['pixel_ap'], 'cand_ap': Sc['pixel_ap'], 'v1_auroc': Sv['image_auroc'],
                                  'cand_auroc': Sc['image_auroc'], 'v1_fm': Sv['forgetting_ap'], 'cand_fm': Sc['forgetting_ap']}
            for t in order:
                d = me['cells'][t][cand][str(T - 1)]['ap'] - me['cells'][t]['V1'][str(T - 1)]['ap']
                if d < 0:
                    negcls.setdefault(ds, {}).setdefault(t, []).append(round(d, 4))
    summ, rule = {}, {}
    for ds in DSETS:
        v = per[ds].values()
        arr = lambda k: np.array([x[k] for x in v])
        summ[ds] = {k: {'mean': float(arr(k).mean()), 'std': float(arr(k).std(ddof=1)), 'min': float(arr(k).min()),
                        'max': float(arr(k).max())} for k in ('dAP', 'dAUROC', 'dFM', 'dACQ')}
        summ[ds]['n_dAP_pos'] = int((arr('dAP') > 0).sum())
    rule['mean_dAP_gt_0_all'] = all(summ[ds]['dAP']['mean'] > 0 for ds in DSETS)
    rule['ge2_datasets_7of9'] = sum(summ[ds]['n_dAP_pos'] >= 7 for ds in DSETS) >= 2
    rule['mean_dAUROC_ge_-0.005_all'] = all(summ[ds]['dAUROC']['mean'] >= -0.005 for ds in DSETS)
    rule['mean_dFM_le_0.005_all'] = all(summ[ds]['dFM']['mean'] <= 0.005 for ds in DSETS)
    par_ok = all(p['equal'] for p in parity.values())
    outcome = ('COMPRESSION_BLOCKED' if not par_ok else
               'COMPRESSION_BENCHMARK_SUPPORTED' if all(rule.values()) else 'COMPRESSION_INCONCLUSIVE')
    out = {'candidate': cand, 'outcome': outcome, 'rule': rule, 'summary': summ, 'per_cell': per,
           'v1_parity_locked': parity, 'negative_classes_final_ap': negcls, 'spec_sha256': sha0,
           'time': time.strftime('%FT%T')}
    wj(OUT / 'ROBUST.json', out)
    log('ROBUST', outcome, rule, {ds: (round(summ[ds]['dAP']['mean'], 4), summ[ds]['n_dAP_pos']) for ds in DSETS})


if __name__ == '__main__':
    main()
