"""RIGEL-BUDGET-00: budget ablation of the frozen RIGEL-CAD 1.1 storage formats, 2 to 24 MiB (descriptive).

    python -B -m exps.rigel_budget_00.sweep <ds>      # GPU: mvtec | visa | mpdd (order a, seed 2711)
    python -B -m exps.rigel_budget_00.sweep summary

No new method and no selection: the two shipped storage formats are run unchanged at a fixed grid of budgets
registered here before any result of this sweep was seen:
    fp16 (v1)          4, 6, 8, 12, 16, 24 MiB       (fp16 needs >= 2.25 MiB for the slow statistic alone)
    int4_triangular    2, 3, 4, 6, 8, 12, 16, 24 MiB
The official test sets were evaluated in earlier work, so this is a descriptive curve on exposed benchmarks, not a
held-out validation, and nothing is chosen from it.

Per dataset (canonical K: token caches and cached v1 per-task fits, read-only):
  * the budget-independent per-task fit (coreset keys, S_t, n_t) is the cached v1 fit; RIGEL-ENG-01 checked that it
    equals the package's RigelCAD.compute_fit on every task. Each budget's model receives the same fits through
    RigelCAD.write_task, i.e. the same computation as fit_task;
  * every stage of every model is fingerprinted (sha256 of the stored keys / codes and of W). Budgets whose
    capacity never binds have identical fingerprints at every stage and are scored once (reported as identical_to);
  * every category is scored at every stage from its acquisition on (forgetting), metrics with rigel_cad.metrics
    (no bootstrap);
  * anchors: fp16 @ 4 / 8 / 16 MiB must equal the locked per-stage metrics (G3E-PAPER-01 A3 / NL03-MULTIRATE-01 E),
    int4 @ 8 MiB the locked RIGEL-CC-01 Q4T metrics.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from exps.rigel_q4t_int import verify as V
from rigel_cad import RigelCAD, RigelConfig
from rigel_cad import data as DATA
from rigel_cad import metrics as M
from rigel_cad.model import TaskFit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/rigel_cad/budget_sweep'
GRID = {'fp16': (4, 6, 8, 12, 16, 24), 'int4_triangular': (2, 3, 4, 6, 8, 12, 16, 24)}
A3 = ROOT / 'results/hope_cad/g3e_paper_01/cells/METRICS_A3_dinov2_{ds}_a2711.json'
DATASETS = {'mvtec': 'MVTec AD', 'visa': 'VisA', 'mpdd': 'MPDD'}
G3E_RES = ROOT / 'results/hope_cad/g3e_paper_01/RESULTS.json'


def name_of(storage, mib):
    return f'{"fp16" if storage == "fp16" else "q4t"}@{mib:g}'


def _h(*ts):
    h = hashlib.sha256()
    for t in ts:
        if t is not None:
            h.update(t.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:24]


def fingerprint(m, l):
    st = m.history[l]
    if st.storage == 'int4_triangular':
        return _h(st.q4, st.scale, st.W)
    return _h(st.keys, st.W)


def run(ds):
    from exps.nl03_multirate_01 import final as FN
    V.guard()
    order = FN.bench_order(ds, 'a')
    T = len(order)
    tag = f'{ds}_a2711'
    models = {name_of(s, b): RigelCAD(RigelConfig(budget_mib=b, storage=s), keep_history=True)
              for s, bs in GRID.items() for b in bs}
    write_s = {n: 0.0 for n in models}
    t0 = time.time()
    for l, t in enumerate(order):
        f = V.cached_fit(ds, t.split('/')[1])
        fit = TaskFit(f.keys16.cuda(), f.S, f.n, f.n_tokens)
        for n, m in models.items():
            V.sync()
            t1 = time.time()
            m.write_task(fit, t)
            V.sync()
            write_s[n] += time.time() - t1
        V.log('WRITE', tag, l, t, {n: len(m.fast) for n, m in models.items()})
        del fit
        torch.cuda.empty_cache()
    fps = {n: [fingerprint(m, l) for l in range(T)] for n, m in models.items()}
    scored, identical_to = [], {}
    for n in models:
        twin = next((s for s in scored if fps[s] == fps[n]), None)
        if twin is None:
            scored.append(n)
        else:
            identical_to[n] = twin
    V.log('UNIQUE', tag, scored, 'identical', identical_to)

    data_root = V.G_().data_root(ds)
    cells = {n: {} for n in scored}
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = V.test_tokens(ds, c)
        q = q.reshape(-1, 768).cuda()
        _, test = DATA.category_split(ds, data_root, c)
        cat = M.Category(data_root, test)
        assert cat.rows == list(rows), (t, 'row order')
        for n in scored:
            m = models[n]
            grid = torch.stack([m.patch_energy_grid(q, m.state_at(l)) for l in range(j, T)])
            cells[n][t], _ = M.evaluate_category(cat, grid, list(range(j, T)), None)
        V.log('EVAL', tag, c, {n: round(cells[n][t][T - 1]['ap'], 4) for n in scored})
        del q
        torch.cuda.empty_cache()

    # anchors (locked per-stage metrics)
    a3 = json.loads(Path(str(A3).format(ds=ds)).read_text())['cells']
    le = json.loads((V.LOCK_MET / f'METRICS_{tag}.json').read_text())['cells']
    lq = json.loads((V.CC_RES / f'METRICS_{tag}.json').read_text())['cells']
    resolve = lambda n: identical_to.get(n, n)
    anchors = {
        'fp16@4 == G3E A3 E@4': all(cells[resolve('fp16@4')][t][l] == a3[t]['E@4'][str(l)] for t in order
                                    for l in cells[resolve('fp16@4')][t]),
        'fp16@16 == G3E A3 E@16': all(cells[resolve('fp16@16')][t][l] == a3[t]['E@16'][str(l)] for t in order
                                      for l in cells[resolve('fp16@16')][t]),
        'fp16@8 == NL03 E': all(cells[resolve('fp16@8')][t][l] == le[t]['E'][str(l)] for t in order
                                for l in cells[resolve('fp16@8')][t]),
        'q4t@8 == CC-01 Q4T': all(cells[resolve('q4t@8')][t][l] == lq[t]['Q4T'][str(l)] for t in order
                                  for l in cells[resolve('q4t@8')][t]),
    }
    configs = {}
    for n, m in models.items():
        src = resolve(n)
        configs[n] = {'storage': m.config.storage, 'budget_mib': m.config.budget_mib,
                      'key_capacity': m.config.key_capacity, 'key_bytes': m.config.key_bytes,
                      'keys_per_stage': [len(m.history[l].keys) if m.history[l].keys is not None
                                         else len(m.history[l].q4) for l in range(T)],
                      'bytes_per_stage': [m.history[l].bytes for l in range(T)],
                      'capacity_binds': None,
                      'identical_to': identical_to.get(n), 'write_seconds': write_s[n],
                      **M.stream_summary(order, cells[src]), 'per_task_stage': cells[src]}
        configs[n]['bytes_max'] = max(configs[n]['bytes_per_stage'])
        configs[n]['capacity_binds'] = max(configs[n]['keys_per_stage']) == m.config.key_capacity
    out = {'tag': tag, 'dataset': ds, 'order': order, 'grid_registered': GRID, 'configs': configs,
           'anchors': anchors, 'anchors_all_equal': all(anchors.values()), 'seconds': time.time() - t0,
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    V.wj(OUT / f'SWEEP_{tag}.json', out)
    V.log('DONE', tag, 'anchors', anchors, {n: round(v['pixel_ap'], 4) for n, v in configs.items()})


def summary():
    ref = json.loads(G3E_RES.read_text())['A3']
    rows, L = {}, ['# RIGEL-BUDGET-00: budget curve (order a, seed 2711)', '',
                   'Final model, task-agnostic. Pixel AP = native binned AP; FM = forgetting. '
                   '"same as" = capacity never binds, identical state at every stage. '
                   'fp16 @ 32 MiB and unbounded U: locked G3E-PAPER-01 values (not recomputed).', '']
    for ds in ('mvtec', 'visa', 'mpdd'):
        p = OUT / f'SWEEP_{ds}_a2711.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows[ds] = {'anchors_all_equal': d['anchors_all_equal'], 'configs': {}}
        L += [f'## {DATASETS[ds]} (anchors equal: '
              f'{d["anchors_all_equal"]})', '',
              '| Storage | Budget (MiB) | Key capacity | Keys (final) | Bytes max | Pixel AP | Image AUROC | FM | Note |',
              '|---|---|---|---|---|---|---|---|---|']
        for n, x in d['configs'].items():
            rows[ds]['configs'][n] = {k: x[k] for k in ('storage', 'budget_mib', 'key_capacity', 'bytes_max', 'pixel_ap',
                                                        'image_auroc', 'forgetting_ap', 'identical_to', 'capacity_binds')}
            rows[ds]['configs'][n]['keys_final'] = x['keys_per_stage'][-1]
            note = f'same as {x["identical_to"]}' if x['identical_to'] else ('' if x['capacity_binds'] else 'cap not binding')
            L.append(f'| {x["storage"]} | {x["budget_mib"]:g} | {x["key_capacity"]:,} | {x["keys_per_stage"][-1]:,} | '
                     f'{x["bytes_max"]:,} | {x["pixel_ap"]:.4f} | {x["image_auroc"]:.4f} | {x["forgetting_ap"]:.4f} | {note} |')
        for k in ('E@32', 'U'):
            s = ref[ds][k]['per_seed']['2711']
            L.append(f'| fp16 (locked) | {"32" if k == "E@32" else "unbounded"} | — | — | {s["bytes_max"]:,} | '
                     f'{s["pixel_ap"]:.4f} | {s["image_auroc"]:.4f} | {s["fm"]:.4f} | G3E-PAPER-01 |')
            rows[ds][f'locked_{k}'] = {kk: s[kk] for kk in ('bytes_max', 'pixel_ap', 'image_auroc', 'fm')}
        L.append('')
    V.wj(OUT / 'SWEEP_SUMMARY.json', rows)
    (OUT / 'SWEEP_TABLES.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    summary() if sys.argv[1] == 'summary' else run(sys.argv[1])
