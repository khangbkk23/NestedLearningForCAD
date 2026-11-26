"""GG-01 FULL validation (registered in exps/rigel_gg01/PREREG.md §7). Runs only after DEV_VERDICT = GG01_DEV_PASS.

27 cells (MVTec AD / VisA / MPDD x orders a, b, c x seeds 2711-2713). In every cell the GG01 arm is scored at every
stage (forgetting) and the V1 arm at the final stage, which must equal the locked E@8 final-stage grids bitwise.
v1 metrics are the locked ones (exps.g3e_paper_01.tables.row, read-only); GG01 metrics use the same locked evaluator
(exps.nl03_pressure_00.evaluate.run_task) with the same arguments as exps.g3e_paper_01.run.metrics.

    python -B -m exps.rigel_gg01.full run       # all cells (GPU) + metrics (CPU), resumable
    python -B -m exps.rigel_gg01.full gate      # registered FULL gate -> FULL_VERDICT.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.rigel_gg01 import gg
from exps.rigel_gg01.dev import PREREG, guard, sha, wj

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/rigel_gg01/full'
CACHE = Path('/mnt/k/NLCAD/cache/rigel_gg01')
DEV_VERDICT = ROOT / 'results/hope_cad/rigel_gg01/dev/DEV_VERDICT.json'
DSETS = ('mvtec', 'visa', 'mpdd')
SEEDS = (2711, 2712, 2713)
GPU_CEILING_S = 6 * 3600


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'full.log', 'a') as f:
        f.write(s + '\n')


def cells():
    return [(ds, ok, s) for ds in DSETS for ok in 'abc' for s in SEEDS]


def tag_of(c):
    return f'{c[0]}_{c[1]}{c[2]}'


def gpu_used():
    p = OUT / 'LEDGER.jsonl'
    return sum(json.loads(l)['seconds'] for l in p.read_text().splitlines()) if p.exists() else 0.0


def locked_grid_dir(ds, ok, seed):
    """Directory of the locked E@8 grids of a cell (the source used by the G3E-PAPER-01 tables)."""
    from exps.g3e_paper_01 import tables as TB
    sp, mp_, arm = TB._src(ds, ok, seed, 'E@8')
    if 'nl03_multirate_01' in str(sp):
        return Path('/mnt/k/NLCAD/cache/nl03_multirate_01/bench') / f'{ds}_{ok}2711' / arm
    tag = sp.name[len('SCORE_'):-len('.json')]
    return Path('/mnt/k/NLCAD/cache/g3e_paper_01/cells') / tag / arm


@torch.no_grad()
def run_cell(c):
    from exps.nl03_multirate_01 import final as FN
    from exps.nl03_multirate_01 import run as RN
    from exps.nl03_pressure_00 import core as PC
    ds, ok, seed = c
    tag = tag_of(c)
    done = OUT / 'cells' / f'SCORE_{tag}.json'
    if done.exists():
        return
    guard()
    t0 = time.time()
    G = PC.bind()
    PC.use(G, ds)
    order = FN.bench_order(ds, ok)
    T = len(order)
    st = gg.Stream(['V1', 'GG01'], seed)
    fit_s = 0.0
    try:
        for l, t in enumerate(order):
            p = CACHE / 'fit' / ds / f"{t.split('/')[1]}_s{seed}.pt"
            if p.exists():
                d = torch.load(p, weights_only=True)
                f = gg.Fit(kc=d['kc'], S=d['S'], n=d['n'], n_tokens=d['n_tokens'], keys16=d['keys16'])
                dt = d['fit_s']
            else:
                x16, _ = RN.fit_tokens(G, t, order=FN.train_rows_sorted(G, t))
                t1 = time.time()
                f = gg.fit_task(x16, seed)
                torch.cuda.synchronize()
                dt = time.time() - t1
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'kc': f.kc, 'S': f.S, 'n': f.n, 'n_tokens': f.n_tokens, 'keys16': f.keys16, 'fit_s': dt}, p)
                del x16
                torch.cuda.empty_cache()
            fit_s += dt
            st.task(l, None, f, dt)
            log('CELL', tag, l, t, {a: st.arms[a].log[-1]['n_keys'] for a in st.arms},
                st.arms['GG01'].log[-1]['metric'])
        ex = PC.extract_rec(G, ds)
        ref_dir = locked_grid_dir(ds, ok, seed)
        parity, lat = {}, []
        for j, t in enumerate(order):
            cname = t.split('/')[1]
            PC.use(G, ds)
            q, rows = G.load_test(ds, cname, ex)
            gl = []
            for l in range(j, T):
                torch.cuda.synchronize()
                t1 = time.time()
                gl.append(st.energy('GG01', l, q))
                torch.cuda.synchronize()
                lat.append((time.time() - t1) / len(rows))
            grid = torch.stack(gl).reshape(T - j, len(rows), 28, 28).contiguous()
            p = CACHE / 'full' / tag / 'GG01' / f'{cname}.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': rows, 'grid': grid, 'stages': list(range(j, T)), 'task': t, 'pos': j, 'order': order}, p)
            v1 = st.energy('V1', T - 1, q).reshape(len(rows), 28, 28)
            ref = torch.load(ref_dir / f'{cname}.pt', weights_only=True)
            parity[t] = {'rows_equal': ref['rows'] == rows, 'final_bitwise': bool(torch.equal(v1, ref['grid'][-1]))}
            del q
            torch.cuda.empty_cache()
    finally:
        with open(OUT / 'LEDGER.jsonl', 'a') as fh:
            fh.write(json.dumps({'cell': tag, 'seconds': time.time() - t0, 'time': time.strftime('%FT%T')}) + '\n')
    m = st.arms
    K = st.state('GG01', T - 1)[0]
    wj(done, {'tag': tag, 'dataset': ds, 'order_key': ok, 'seed': seed, 'order': order,
              'bytes': {a: [r['bytes'] for r in m[a].log] for a in m},
              'physical': {a: gg.physical_bytes(m[a].log[-1]['n_keys'], True) for a in m},
              'update_s': {a: fit_s + m[a].select_s for a in m}, 'select_s': {a: m[a].select_s for a in m},
              'latency_ms_per_img_p50': float(np.median(lat) * 1e3), 'select_log': {a: m[a].log for a in m},
              'drift': st.drift(), 'cond': st.cond, 'v1_parity': parity,
              'parity_ok': all(v['rows_equal'] and v['final_bitwise'] for v in parity.values()),
              'seconds': time.time() - t0, 'peak_vram': torch.cuda.max_memory_allocated(),
              'prereg_sha256': sha(PREREG), 'n_keys_final': len(K)})
    log('CELL DONE', tag, '%.0fs' % (time.time() - t0), 'parity', all(v['final_bitwise'] for v in parity.values()))


def metrics_cell(c):
    from exps.nl03_pressure_00 import evaluate as PEV
    tag = tag_of(c)
    outp = OUT / 'cells' / f'METRICS_{tag}.json'
    if outp.exists():
        return
    sc = json.loads((OUT / 'cells' / f'SCORE_{tag}.json').read_text())
    boot = sc['order_key'] == 'a'
    args = [(t, j, [('GG01', str(CACHE / 'full' / tag / 'GG01' / f"{t.split('/')[1]}.pt"), boot, False)], 4)
            for j, t in enumerate(sc['order'])]
    res, boots, info = {}, {}, {}
    t0 = time.time()
    with mp.get_context('spawn').Pool(2) as pool:
        for t, r, b, inf in pool.imap_unordered(PEV.run_task, args):
            res[t], boots[t], info[t] = r, b, inf
    rec = {'tag': tag, 'cells': res, 'tasks': info, 'seconds': time.time() - t0}
    if boot:
        bp = CACHE / 'full' / f'boot_{tag}.npz'
        np.savez_compressed(bp, **{f'{t}||{a}||{m}': v[m] for t, bj in boots.items() for a, v in bj.items() for m in v})
        rec['boot_file'] = str(bp)
        rec['boot_sha256'] = sha(bp)
    wj(outp, rec)
    log('METRICS', tag, '%.0fs' % rec['seconds'])


def run():
    v = json.loads(DEV_VERDICT.read_text())
    assert v['verdict'] == 'GG01_DEV_PASS', f"STOP: FULL requires GG01_DEV_PASS, got {v['verdict']}"
    assert v['prereg_sha256'] == sha(PREREG), 'STOP: pre-registration changed'
    for c in cells():
        if not (OUT / 'cells' / f'SCORE_{tag_of(c)}.json').exists() and gpu_used() > GPU_CEILING_S:
            log('BUDGET STOP before', tag_of(c))
            break
        run_cell(c)
        metrics_cell(c)
    log('RUN DONE')


# ---------------------------------------------------------------------------------------------- gate
def gg_row(c):
    """Same summaries as exps.g3e_paper_01.tables.row, for the GG01 arm of a FULL cell."""
    tag = tag_of(c)
    sc = json.loads((OUT / 'cells' / f'SCORE_{tag}.json').read_text())
    me = json.loads((OUT / 'cells' / f'METRICS_{tag}.json').read_text())
    order, T, a = sc['order'], len(sc['order']), 'GG01'
    ap = {t: {int(l): v['ap'] for l, v in me['cells'][t][a].items()} for t in order}
    au = {t: {int(l): v['i_auroc'] for l, v in me['cells'][t][a].items()} for t in order}
    out = {'pixel_ap': float(np.mean([ap[t][T - 1] for t in order])),
           'image_auroc': float(np.mean([au[t][T - 1] for t in order])),
           'fm': float(np.mean([max(ap[t][l] for l in range(j, T - 1)) - ap[t][T - 1] for j, t in enumerate(order[:-1])])),
           'prequential_ap': float(np.mean([np.mean([ap[order[j]][l] for j in range(l + 1)]) for l in range(T)])),
           'acq_ap': float(np.mean([ap[t][j] for j, t in enumerate(order)])),
           'bytes_max': max(sc['bytes'][a]), 'update_s': sc['update_s'][a],
           'latency_ms_per_img': sc['latency_ms_per_img_p50'],
           'per_class': {t: {'final_ap': ap[t][T - 1], 'final_auroc': au[t][T - 1]} for t in order}}
    if 'boot_file' in me:
        bf = np.load(me['boot_file'])
        out['_boot'] = np.mean([bf[f'{t}||{a}||ap'] for t in order], axis=0)
    return out, sc


def gate():
    from exps.g3e_paper_01 import tables as TB
    rows, per_ds = {}, {}
    f1 = True
    f8 = True
    for c in cells():
        g, sc = gg_row(c)
        v = TB.row(c[0], c[1], c[2], 'E@8')
        f1 &= bool(sc['parity_ok']) and all(np.isfinite(sc['cond']))
        f8 &= sc['physical']['GG01']['total'] <= sc['physical']['V1']['total'] and \
            max(sc['bytes']['GG01']) <= max(sc['bytes']['V1'])
        rows[tag_of(c)] = {'GG01': {k: g[k] for k in ('pixel_ap', 'image_auroc', 'fm', 'acq_ap', 'prequential_ap',
                                                      'bytes_max', 'update_s', 'latency_ms_per_img')},
                           'V1': {k: v[k] for k in ('pixel_ap', 'image_auroc', 'fm', 'acq_ap', 'prequential_ap',
                                                    'bytes_max', 'update_s', 'latency_ms_per_img')},
                           'per_class_delta_ap': {t: g['per_class'][t]['final_ap'] - v['per_class'][t]['final_ap']
                                                  for t in g['per_class']}}
        if c[1] == 'a' and '_boot' in g and '_boot' in v:
            d = g['_boot'] - v['_boot']['ap']
            rows[tag_of(c)]['delta_ap_ci95'] = [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]
    for ds in DSETS:
        cs = [tag_of(c) for c in cells() if c[0] == ds]
        dm = lambda k: float(np.mean([rows[t]['GG01'][k] - rows[t]['V1'][k] for t in cs]))
        per_ds[ds] = {'mean_delta_ap': dm('pixel_ap'), 'mean_delta_auroc': dm('image_auroc'), 'mean_delta_fm': dm('fm'),
                      'mean_delta_acq': dm('acq_ap'), 'mean_delta_preq': dm('prequential_ap'),
                      'std_delta_ap': float(np.std([rows[t]['GG01']['pixel_ap'] - rows[t]['V1']['pixel_ap'] for t in cs],
                                                   ddof=1)),
                      'n_cells': len(cs), 'min_category_delta_primary': min(rows[f'{ds}_a2711']['per_class_delta_ap'].values())}
    g = {'F1_correctness': bool(f1),
         'F2_mean_delta_ge_0.005': float(np.mean([per_ds[d]['mean_delta_ap'] for d in DSETS])) >= 0.005,
         'F3_two_of_three_positive': sum(per_ds[d]['mean_delta_ap'] > 0 for d in DSETS) >= 2,
         'F4_no_dataset_below_-0.01': min(per_ds[d]['mean_delta_ap'] for d in DSETS) >= -0.01,
         'F5_no_category_loss_gt_0.03': min(per_ds[d]['min_category_delta_primary'] for d in DSETS) >= -0.03,
         'F6_forgetting_not_worse_gt_0.005': max(per_ds[d]['mean_delta_fm'] for d in DSETS) <= 0.005,
         'F7_auroc_not_worse_gt_0.005': min(per_ds[d]['mean_delta_auroc'] for d in DSETS) >= -0.005,
         'F8_physical_bytes_le_v1': bool(f8),
         'F9_no_mismatch': bool(f1)}
    if not (g['F1_correctness'] and g['F9_no_mismatch']):
        verdict = 'GG01_BLOCKED'
    elif all(g.values()):
        verdict = 'GG01_FULL_PASS'
    else:
        verdict = 'GG01_FULL_REJECTED'
    wj(OUT / 'FULL_VERDICT.json', {'verdict': verdict, 'gates': g, 'per_dataset': per_ds, 'cells': rows,
                                   'mean_delta_ap_3ds': float(np.mean([per_ds[d]['mean_delta_ap'] for d in DSETS])),
                                   'prereg_sha256': sha(PREREG), 'gpu_s': gpu_used(), 'time': time.strftime('%FT%T')})
    log('FULL VERDICT', verdict, g)


if __name__ == '__main__':
    {'run': run, 'gate': gate}[sys.argv[1]]()
