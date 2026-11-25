"""NL03-MULTIRATE-01 report helper (read-only over locked artifacts): BENCH_RESULTS.json + markdown tables.

    python -B -m exps.nl03_multirate_01.report_tables
"""
from __future__ import annotations

import json
import time

import numpy as np

from exps.nl03_multirate_01 import final as FN
from exps.nl03_multirate_01 import run as RN


def main():
    FN.verify_final()
    out = {'bench': {}, 'not_run': {}}
    for ds in ('mvtec', 'visa', 'mpdd'):
        for ok in 'abc':
            tag = f'{ds}_{ok}2711'
            if not (FN.BENCH_OUT / f'METRICS_{tag}.json').exists():
                out['not_run'][tag] = 'see report (engineering edge case)'
                continue
            out['bench'][tag] = FN.bench_table(ds, ok)
    # order-mean (a, b, c available) per dataset and arm
    agg = {}
    for ds in ('mvtec', 'visa', 'mpdd'):
        tags = [t for t in out['bench'] if t.startswith(ds + '_')]
        arms = [a for a in out['bench'][tags[0]] if all(a in out['bench'][t] for t in tags)]
        agg[ds] = {a: {m: float(np.mean([out['bench'][t][a][m] for t in tags])) for m in
                       ('pixel_ap', 'image_auroc', 'fm', 'prequential_ap', 'acq_ap')} | {'orders': [t[-5] for t in tags]}
                   for a in arms}
        agg[ds]['P_minus'] = {a: [out['bench'][t]['P']['pixel_ap'] - out['bench'][t][a]['pixel_ap'] for t in tags]
                              for a in arms if a != 'P'}
    out['order_mean'] = agg
    out['ad2_test_public'] = RN.rj(FN.AD2_OUT / 'AD2_TEST_PUBLIC.json')['table']
    out['time'] = time.strftime('%FT%T')
    RN.wj(FN.OUT / 'BENCH_RESULTS.json', out)
    for t, tab in out['bench'].items():
        print(f'## {t}')
        for a, r in tab.items():
            ci = r.get('pixel_ap_ci95')
            print(f"{a:3s} AP {r['pixel_ap']:.4f}{' [%.4f, %.4f]' % tuple(ci) if ci else ''} AUROC {r['image_auroc']:.4f} "
                  f"FM {r['fm']:.4f} preq {r['prequential_ap']:.4f} acq {r['acq_ap']:.4f} MiBmax {r['bytes_max'] / 2**20:.2f} "
                  f"upd {r['update_s'] if not isinstance(r['update_s'], (int, float)) else round(r['update_s'], 1)} "
                  f"lat {r['latency_ms_per_img']}")
        for k, v in tab.get('P', {}).items():
            if k.startswith('ap_minus'):
                print('   ', k, round(v['delta'], 4), [round(x, 4) for x in v['ci95']])
    print('## order mean')
    for ds, d in agg.items():
        for a, r in d.items():
            if a == 'P_minus':
                print(ds, 'P_minus', {k: [round(x, 4) for x in v] for k, v in r.items()})
            else:
                print(ds, a, {m: round(r[m], 4) for m in ('pixel_ap', 'image_auroc', 'fm', 'prequential_ap')}, r['orders'])


if __name__ == '__main__':
    main()
