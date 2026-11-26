"""Export the paper tables under RIGEL-CAD names from the locked research results (read-only).

    python -m rigel_cad.paper_tables [--src results/hope_cad/g3e_paper_01/RESULTS.json] [--out results/rigel_cad/paper]

Writes RIGEL_CAD_RESULTS.json (same numbers, renamed methods) and RIGEL_CAD_TABLES.md. The numbers are not
recomputed; provenance (source file and its sha256) is recorded in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NAMES = {
    'E@8': 'RIGEL-CAD (8 MiB)', 'E@4': 'RIGEL-CAD (4 MiB)', 'E@16': 'RIGEL-CAD (16 MiB)', 'E@32': 'RIGEL-CAD (32 MiB)',
    'E': 'RIGEL-CAD (8 MiB)', 'U': 'RIGEL-CAD (unbounded)',
    'F': 'Ablation: fast level only (no slow whitening, all bytes as keys)',
    'B1': 'Ablation: RIGEL-CAD keys without whitening',
    'E4': 'Ablation: slow level from the current task only',
    'G': 'Ablation: generic feature-covariance whitening',
    'Q': 'Reference: hindsight k-centre over all past coresets (not streaming-feasible)',
    'FIFO': 'Ablation: FIFO key eviction',
}
DATASETS = {'mvtec': 'MVTec AD', 'visa': 'VisA', 'mpdd': 'MPDD'}


def _rename(obj):
    if isinstance(obj, dict):
        return {NAMES.get(k, k): _rename(v) for k, v in obj.items()}
    return obj


def _f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='results/hope_cad/g3e_paper_01/RESULTS.json')
    p.add_argument('--out', default='results/rigel_cad/paper')
    a = p.parse_args(argv)
    src = Path(a.src)
    raw = src.read_bytes()
    r = json.loads(raw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {'method': 'RIGEL-CAD', 'source': str(src), 'source_sha256': hashlib.sha256(raw).hexdigest(),
           'names': NAMES, 'tables': _rename({k: v for k, v in r.items() if k != 'plan_status'})}
    (out / 'RIGEL_CAD_RESULTS.json').write_text(json.dumps(res, indent=1))
    L = ['# RIGEL-CAD paper tables', '', f'Source: `{src}` (sha256 {res["source_sha256"][:16]}…). Final model, '
         'task-agnostic; native binned pixel AP, image AUROC; FM = forgetting.', '',
         '## Main (order a, seed 2711)', '', '| Dataset | Method | Pixel AP [CI95] | Image AUROC | FM | Bytes |',
         '|---|---|---|---|---|---|']
    for ds, d in r['order_a_seed2711'].items():
        for k in ('U', 'E@8'):
            x = d[k]
            ci = x.get('pixel_ap_ci95')
            L.append(f'| {DATASETS[ds]} | {NAMES[k]} | {_f(x["pixel_ap"])} [{_f(ci[0])}, {_f(ci[1])}] | '
                     f'{_f(x["image_auroc"])} | {_f(x["fm"])} | {x["bytes_max"]:,} |')
    L += ['', '## Robustness (3 orders x 3 seeds, mean ± std)', '', '| Dataset | Method | Pixel AP | Image AUROC |',
          '|---|---|---|---|']
    for ds, d in r['A2'].items():
        for k in ('U', 'E@8'):
            x = d[k]
            L.append(f'| {DATASETS[ds]} | {NAMES[k]} | {_f(x["pixel_ap"]["mean"])} ± {_f(x["pixel_ap"]["std"])} | '
                     f'{_f(x["image_auroc"]["mean"])} ± {_f(x["image_auroc"]["std"])} |')
    L += ['', '## Ablations at 8 MiB (paired RIGEL-CAD − X, pixel AP)', '', '| Dataset | X | Pixel AP of X | Δ [CI95] |',
          '|---|---|---|---|']
    for ds, d in r['A4'].items():
        for k in ('F', 'B1', 'E4', 'G', 'FIFO', 'Q'):
            x = d.get(k)
            if x and x.get('E_minus_ap'):
                e = x['E_minus_ap']
                L.append(f'| {DATASETS[ds]} | {NAMES[k]} | {_f(x["pixel_ap"])} | {e["delta"]:+.4f} '
                         f'[{e["ci95"][0]:+.4f}, {e["ci95"][1]:+.4f}] |')
    L += ['', '## Budget curve (order a, 3 seeds, pixel AP mean ± std)', '', '| Dataset | Method | Bytes | Pixel AP |',
          '|---|---|---|---|']
    for ds, d in r['A3'].items():
        for k in ('E@4', 'E@8', 'E@16', 'E@32', 'U'):
            x = d[k]
            L.append(f'| {DATASETS[ds]} | {NAMES[k]} | {int(x["bytes_max"]["mean"]):,} | '
                     f'{_f(x["pixel_ap"]["mean"])} ± {_f(x["pixel_ap"]["std"])} |')
    if isinstance(r.get('B'), dict):
        L += ['', '## Matched backbone ViT-B/8 @224 (ablation)', '', '| Dataset | Method | Pixel AP | Image AUROC |',
              '|---|---|---|---|']
        for ds, d in r['B'].items():
            if isinstance(d, dict):
                for k in ('U', 'E@8', 'F'):
                    L.append(f'| {DATASETS[ds]} | {NAMES[k]} | {_f(d[k]["pixel_ap"])} | {_f(d[k]["image_auroc"])} |')
    (out / 'RIGEL_CAD_TABLES.md').write_text('\n'.join(L) + '\n')
    print('wrote', out / 'RIGEL_CAD_RESULTS.json', out / 'RIGEL_CAD_TABLES.md')


if __name__ == '__main__':
    main()
