"""NANM-SYS-FINAL harness: P5 continuation on real data, P6 score/metric parity, P7 profiling (no optimisation).

All GPU work on physical GPU 2 (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2; bus 0x41 asserted).
The harness may import both nanm and the unchanged oracle; nanm never imports the oracle.
Phases (see agents/tasks/task_nanm_sys_01.md §19):
    p5a DS ARM SEED                 uninterrupted run, one checkpoint generation per boundary (A)
    p5b DS ARM SEED START STOP      fresh-process segment of the restart chain (B)
    p5cmp                           A vs B (bitwise) and A vs oracle normal_runA
    encoder                         nanm encoder on all train + test images vs oracle tokens; MPDD test tokens (oracle)
    oscore DS SEED                  oracle-here grids (G3, G3@C) from normal_runA
    nscore DS SEED                  nanm grids from A checkpoints + readout parity vs oracle readout functions
    metrics DS SRC                  sealed metric functions on SRC in {oracle, nanm}
    mcmp                            metric parity (A: nanm vs oracle-here; B: oracle-here vs sealed, descriptive)
    profile                         P7 measurements
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from exps.nanm_sys import regen as RG

CK = RG.CACHE / 'ckpt'
SC = RG.CACHE / 'scores_final'
N_EVAL = 16
B_CONFIGS = [('mpdd', 'G3', s) for s in (2711, 2712, 2713)] + [('visa', 'G3', 2711), ('visa', 'G3', 2712),
                                                               ('visa', 'G3@C', 2711)]
A_CONFIGS = [('mpdd', 'G3', s) for s in (2711, 2712, 2713)] + [('visa', a, s) for a in ('G3', 'G3@C')
                                                               for s in (2711, 2712, 2713)]


def guard():
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise SystemExit('physical GPU 2 only')
    p = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or p.pci_bus_id != 0x41:
        raise SystemExit(f'refusing bus {p.pci_bus_id:#x}')
    return {'physical_gpu': 2, 'pci_bus_id': p.pci_bus_id, 'name': p.name,
            'free_gib': torch.cuda.mem_get_info(0)[0] / 2**30}


def out():
    o = RG.CACHE.parent.parent / 'NestedLearningForCAD/results/hope_cad/nanm_sys/final'
    o.mkdir(parents=True, exist_ok=True)
    return o


def name(ds, arm, seed):
    return f'{ds}_{arm.replace("@", "at")}_s{seed}'


def system_for(arm, seed, device='cuda'):
    from nanm import G3Config, NANMSystem
    return NANMSystem.create(G3Config(seed=seed), device)


def fit_inputs(G, ds, c):
    data = G.load_train(ds, c)
    fi = G.fit_index(ds, data['rows'])
    order_sha = hashlib.sha256('\n'.join(data['rows'][i]['relative_path'] for i in fi).encode()).hexdigest()
    return data['features'][torch.as_tensor(fi)], {'fit_image_order_sha256': order_sha, 'n_fit_images': len(fi)}


def fit_one(G, nm, ds, arm, t, c, rec):
    tok, prov = fit_inputs(G, ds, c)
    k = None if arm == 'G3' else rec[ds]['sizes']['g3c'][t]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    rep = nm.fit_task(tok, task_label=c, k=k, provenance=prov)
    torch.cuda.synchronize()
    return rep, time.perf_counter() - t0


# ------------------------------------------------------------------ P5
def p5a(ds, arm, seed):
    dev = guard(); RG.numerics(); G, o = RG.setup()
    rec = {ds: G.A.rj(o.parent / 'regen/r1_runA' / f'normal_{ds}.json') if False else
           G.A.rj(G.ROOT / 'results/hope_cad/nanm_sys/regen/r1_runA' / f'normal_{ds}.json')}
    root = CK / 'A' / name(ds, arm, seed)
    if root.exists():
        raise SystemExit(f'exists: {root}')
    nm = system_for(arm, seed)
    log = []
    for t, c in enumerate(G.order(ds)):
        rep, dt = fit_one(G, nm, ds, arm, t, c, rec)
        t0 = time.perf_counter(); gen = nm.save_state(root); dts = time.perf_counter() - t0
        log.append({'stage': t, 'task': c, 'gen': gen, 'fit_s': dt, 'save_s': dts, **{k: v for k, v in rep.items()}})
    G.A.wj(out() / 'p5' / f'A_{name(ds, arm, seed)}.json', {'boundaries': log, 'device': dev,
                                                            'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
    print('P5A', ds, arm, seed, 'boundaries', len(log))


def p5b(ds, arm, seed, start, stop):
    from nanm import G3Config, NANMSystem
    dev = guard(); RG.numerics(); G, o = RG.setup()
    rec = {ds: G.A.rj(G.ROOT / 'results/hope_cad/nanm_sys/regen/r1_runA' / f'normal_{ds}.json')}
    root = CK / 'B' / name(ds, arm, seed)
    if start == 0:
        if root.exists():
            raise SystemExit(f'exists: {root}')
        nm = system_for(arm, seed)
    else:
        t0 = time.perf_counter()
        nm = NANMSystem.load_state(root, cfg=G3Config(seed=seed), device='cuda')       # fresh process
        load_s = time.perf_counter() - t0
        assert len(nm.state.keys) == start and nm.load_report['cond_rebuilt_equals_saved'], nm.load_report
    seg = []
    for t, c in list(enumerate(G.order(ds)))[start:stop]:
        rep, dt = fit_one(G, nm, ds, arm, t, c, rec)
        seg.append({'stage': t, 'gen': nm.save_state(root), 'fit_s': dt})
    G.A.wj(out() / 'p5' / f'B_{name(ds, arm, seed)}_{start}_{stop}.json',
           {'segment': [start, stop], 'pid': os.getpid(), 'loaded_from_checkpoint': start > 0,
            'load_s': load_s if start else None, 'boundaries': seg, 'device': dev})
    print('P5B', ds, arm, seed, start, stop)


def _eval_tokens(G, ds):
    ev = {}
    for c in G.order(ds):
        data = G.load_train(ds, c)
        ev[c] = data['features'][torch.as_tensor(G.roles(ds, data['rows'])['EVAL'][:N_EVAL])].clone()
    return ev


def p5cmp():
    from nanm import G3Config, NANMSystem
    from nanm import state as NS
    dev = guard(); RG.numerics(); G, o = RG.setup()
    res, fails = {}, []
    for ds, arm, seed in A_CONFIGS:
        nmn = name(ds, arm, seed)
        cfg = G3Config(seed=seed)
        rec = {ds: G.A.rj(G.ROOT / 'results/hope_cad/nanm_sys/regen/r1_runA' / f'normal_{ds}.json')}
        Wkey, Skey, nkey, kkey = ('G3', 'S1', 'n1', 'g3') if arm == 'G3' else ('G3@C', 'SC', 'nC', 'g3c')
        stage_o = G.load_stage(ds, seed, rec)
        genA = NS.list_generations(CK / 'A' / nmn)
        r = {'A_vs_oracle': [], 'B_vs_A': []}
        S_o, n_o, keys_o = [], [], []
        for t, c in enumerate(G.order(ds)):                                   # A vs oracle normal_runA, every boundary
            st = G.load_norm(ds, seed, c, rec)
            stA, _ = NS.read(CK / 'A' / nmn, genA[t], cfg.contract_sha256())
            k = rec[ds]['sizes'][kkey][t]
            S_o.append(st[Skey]); n_o.append(st[nkey]); keys_o.append(st['keys'][:k])
            ok = (torch.equal(stA.selected[t], st['kc'][:k]) and torch.equal(stA.union_keys(), torch.cat(keys_o))
                  and torch.equal(stA.S, sum(S_o)) and stA.n == sum(n_o))
            r['A_vs_oracle'].append(ok)
        sysA = NANMSystem.load_state(CK / 'A' / nmn, cfg=cfg, device='cuda', generation=genA[-1])
        r['A_final_W32_vs_oracle_stage'] = torch.equal(sysA.W, stage_o[len(genA) - 1][Wkey])
        if (ds, arm, seed) in B_CONFIGS:
            genB = NS.list_generations(CK / 'B' / nmn)
            ev = _eval_tokens(G, ds)
            for t in range(len(genB)):
                a = NANMSystem.load_state(CK / 'A' / nmn, cfg=cfg, device='cuda', generation=genA[t])
                b = NANMSystem.load_state(CK / 'B' / nmn, cfg=cfg, device='cuda', generation=genB[t])
                ok = {'selected': all(torch.equal(x, y) for x, y in zip(a.state.selected, b.state.selected)),
                      'task_keys': all(torch.equal(x, y) for x, y in zip(a.state.keys, b.state.keys)),
                      'union': torch.equal(a.state.union_keys(), b.state.union_keys()),
                      'offsets': [len(x) for x in a.state.keys] == [len(x) for x in b.state.keys],
                      'S': torch.equal(a.state.S, b.state.S), 'n': a.state.n == b.state.n,
                      'W64': torch.equal(a._cache[0], b._cache[0]), 'W32': torch.equal(a.W, b.W)}
                for qc in G.order(ds)[:t + 1]:
                    pa, pb = a.predict(ev[qc]), b.predict(ev[qc])
                    ok[f'eval[{qc}]'] = all(torch.equal(pa[x], pb[x]) for x in ('energy', 'map392', 'image_score'))
                r['B_vs_A'].append({'stage': t, **ok})
                fails += [f'{nmn}|stage{t}|{k}' for k, v in ok.items() if v is not True]
        fails += [f'{nmn}|A_vs_oracle|stage{i}' for i, v in enumerate(r['A_vs_oracle']) if not v]
        if not r['A_final_W32_vs_oracle_stage']:
            fails.append(f'{nmn}|A_final_W32')
        res[nmn] = r
        print('P5CMP', nmn, 'fails so far', len(fails), flush=True)
    verdict = 'P5_TIER_A_PASS' if not fails else 'P5_TIER_A_FAIL'
    G.A.wj(out() / 'p5' / 'P5_RESULT.json', {'verdict': verdict, 'failures': fails, 'configs': res, 'device': dev})
    print(verdict, fails[:5])


# ------------------------------------------------------------------ P6 inputs
def _sealed_visa_rows(G):
    return {c: list(torch.load(Path.home() / 'hope_cad_cache/bench_g3/scores' / f'visa_{c}.pt',
                               weights_only=True, mmap=True)['rows']) for c in G.order('visa')}


def encoder():
    from exps.nar_pilot_00 import full as R
    from nanm import FrozenPatchEncoder
    dev = guard(); G, o = RG.setup()
    R.configure()
    lk = G.A.rj(G.ROOT / 'results/hope_cad/bench_g3/LOCK.json')
    tm = {'mpdd': G.test_manifest('mpdd'), 'visa': _sealed_visa_rows(G)}
    for ds in ('mpdd', 'visa'):
        G.K.require(G.K_text(tm[ds]) == lk['test_manifest_sha256'][ds], f'{ds} test manifest != LOCK')
    oenc = G.encoder()
    ex = {}
    for c in G.order('mpdd'):                                                 # oracle-here MPDD test tokens
        p = G.FEAT / f'mpdd_{c}_test.pt'
        if not p.exists():
            G._extract(oenc, G.MPDD, [{'relative_path': r} for r in tm['mpdd'][c]], p)
        ex[p.name] = G._feat_meta(p)['sha256']
    ex.update(G.A.rj(G.ROOT / 'results/hope_cad/nanm_sys/regen/r2/EXTRACT_COMPLETE.json')['files'])
    del oenc; torch.cuda.empty_cache()
    B14 = G.B14()
    nenc = FrozenPatchEncoder.load(B14.ENC_PATH, B14.ENC_SHA, 'cuda')
    res, t0 = {}, time.monotonic()
    for ds in ('mpdd', 'visa'):
        root = G.MPDD if ds == 'mpdd' else G.ROOT / 'data/visa'
        for c in G.order(ds):
            for split in ('train', 'test'):
                rels = [r['relative_path'] for r in G.train_rows(ds, c)] if split == 'train' else tm[ds][c]
                tok = nenc.tokens([root / r for r in rels], batch=8)
                p = G.FEAT / f'{ds}_{c}_{split}.pt'
                G.K.require(G.A.sha(p) == (G._feat_meta(p)['sha256'] if split == 'train' else ex[p.name]), 'cache')
                d = torch.load(p, weights_only=True, mmap=True)
                G.K.require([r['relative_path'] for r in d['rows']] == rels, 'row order')
                res[f'{ds}/{c}/{split}'] = {'images': len(rels), 'bitwise': bool(torch.equal(tok, d['features']))}
                print('ENC', ds, c, split, res[f'{ds}/{c}/{split}'], flush=True)
    G.A.wj(out() / 'p6' / 'TEST_EXTRACT.json', {'files': ex})
    G.A.wj(out() / 'p6' / 'ENCODER.json', {'tokens': res, 'all_bitwise': all(v['bitwise'] for v in res.values()),
                                          'images': sum(v['images'] for v in res.values()),
                                          'seconds': time.monotonic() - t0, 'device': dev})
    print('ENCODER all bitwise', all(v['bitwise'] for v in res.values()))


def _arms(ds):
    return ('G3',) + (('G3@C',) if ds == 'visa' else ())


def oscore(ds, seed):
    """Oracle-here grids: exactly the whitened-arm loop of G.score, from normal_runA (G3, G3@C)."""
    dev = guard(); RG.numerics(); G, o = RG.setup()
    ex = G.A.rj(out() / 'p6' / 'TEST_EXTRACT.json')
    rec = {ds: G.A.rj(G.ROOT / 'results/hope_cad/nanm_sys/regen/r1_runA' / f'normal_{ds}.json')}
    ords, Tn, sz = G.order(ds), len(G.order(ds)), rec[ds]['sizes']
    bank = {'G3': 'g3', 'G3@C': 'g3c'}
    t0 = time.monotonic()
    with torch.no_grad():
        Q = {c: G.load_test(ds, c, ex) for c in ords}
        store = [G.load_norm(ds, seed, m, rec) for m in ords]
        W = G.load_stage(ds, seed, rec)
        grids = {c: {} for c in ords}
        for a in _arms(ds):
            stg = {c: [] for c in ords}
            for l in range(Tn):
                keys = torch.cat([x['keys'][:sz[bank[a]][t]] for t, x in enumerate(store[:l + 1])]).cuda()
                Wg = W[l][a].cuda()
                for c in ords[:l + 1]:
                    stg[c].append(G.whitened(Q[c][0], keys.float(), Wg))
            for c in ords:
                grids[c][f'{a}|s{seed}'] = torch.stack(stg[c])
    SC.mkdir(parents=True, exist_ok=True)
    files = {}
    for j, c in enumerate(ords):
        n = len(Q[c][1])
        g = {k: v.reshape(len(v), n, 28, 28) for k, v in grids[c].items()}
        p = SC / f'oracle_{ds}_{c}_s{seed}.pt'
        torch.save({'rows': Q[c][1], 'first_stage': j, 'grids': g}, p)
        files[p.name] = G.A.sha(p)
    G.A.wj(out() / 'p6' / f'OSCORE_{ds}_s{seed}.json', {'files': files, 'seconds': time.monotonic() - t0,
                                                        'device': dev})
    print('OSCORE', ds, seed, round(time.monotonic() - t0))


def nscore(ds, seed):
    """nanm grids from the A checkpoints (fresh loads per stage), with full readout parity vs the oracle readout."""
    from nanm import G3Config, NANMSystem
    from nanm import readout as NR
    from nanm import state as NS
    from exps.calm02 import native as Nat
    dev = guard(); RG.numerics(); G, o = RG.setup()
    ords, Tn = G.order(ds), len(G.order(ds))
    root = G.MPDD if ds == 'mpdd' else G.ROOT / 'data/visa'
    ex = G.A.rj(out() / 'p6' / 'TEST_EXTRACT.json')['files']
    osc = G.A.rj(out() / 'p6' / f'OSCORE_{ds}_s{seed}.json')['files']
    cfg = G3Config(seed=seed)
    tok, rows = {}, {}
    for c in ords:
        p = G.FEAT / f'{ds}_{c}_test.pt'
        G.K.require(G.A.sha(p) == ex[p.name], 'test tokens')                 # = nanm encoder tokens (ENCODER.json)
        d = torch.load(p, weights_only=True)
        tok[c], rows[c] = d['features'], [r['relative_path'] for r in d['rows']]
    t0, fails, n_checks = time.monotonic(), [], 0
    grids = {a: {c: [] for c in ords} for a in _arms(ds)}
    for a in _arms(ds):
        gens = NS.list_generations(CK / 'A' / name(ds, a, seed))
        G.K.require(len(gens) == Tn, 'A generations')
        for l in range(Tn):
            nm = NANMSystem.load_state(CK / 'A' / name(ds, a, seed), cfg=cfg, device='cuda', generation=gens[l])
            for c in ords[:l + 1]:
                grids[a][c].append(nm.predict(tok[c]))
    for j, c in enumerate(ords):
        pth = SC / f'oracle_{ds}_{c}_s{seed}.pt'
        G.K.require(G.A.sha(pth) == osc[pth.name], 'oracle grid file')
        og = torch.load(pth, weights_only=True)
        G.K.require(og['rows'] == rows[c] and og['first_stage'] == j, 'rows')
        save = {}
        for a in _arms(ds):
            key = f'{a}|s{seed}'
            preds = grids[a][c]
            gn = torch.stack([p['energy'] for p in preds])
            save[key] = gn
            for i, p in enumerate(preds):
                go = og['grids'][key][i]
                mo = G.R2.maps_of(go)
                checks = {'energy': torch.equal(p['energy'], go), 'map392': torch.equal(p['map392'], mo),
                          'image_score': torch.equal(p['image_score'], torch.from_numpy(G.A.image_scores(mo))),
                          'map224': torch.equal(NR.to_224(p['map392']), G.maps224(mo))}
                if i == len(preds) - 1:                                       # native maps at the final stage
                    checks['native'] = all(
                        torch.equal(NR.to_native(p['map392'][q], Nat.native_hw(root / rows[c][q])),
                                    torch.from_numpy(G.NL().map_native(mo[q], Nat.native_hw(root / rows[c][q]))))
                        for q in range(len(rows[c])))
                n_checks += len(checks)
                fails += [f'{ds}|s{seed}|{a}|{c}|stage{j + i}|{k}' for k, v in checks.items() if not v]
        torch.save({'rows': rows[c], 'first_stage': j, 'grids': save}, SC / f'nanm_{ds}_{c}_s{seed}.pt')
    G.A.wj(out() / 'p6' / f'NSCORE_{ds}_s{seed}.json', {'n_checks': n_checks, 'failures': fails,
                                                        'seconds': time.monotonic() - t0, 'device': dev})
    print('NSCORE', ds, seed, 'checks', n_checks, 'fails', len(fails))


# ------------------------------------------------------------------ P6 metrics (labels authorised for parity only)
def _cat_cells(args):
    ds, c, j, src = args
    import numpy as np
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    G, o = RG.setup()
    from exps.bench_aud_00 import run as AUD
    from exps.cadfinal import metrics as MX
    from exps.calm02 import native as N
    ords, Tn = G.order(ds), len(G.order(ds))
    ids = {r['relative_path']: r for r in G.test_rows(ds, c)}
    files = sorted(SC.glob(f'{src}_{ds}_{c}_s*.pt'))
    cells = {}
    root = G.data_root(ds)
    for f in files:
        d = torch.load(f, weights_only=True)
        rows = [ids[r] for r in d['rows']]
        hws = [N.native_hw(root / r['relative_path']) for r in rows]
        masks = [N.native_mask(root / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        m224 = [AUD.mask224(m) for m in masks]
        y = np.array([r['label'] for r in rows])
        for key, g in d['grids'].items():
            stages = range(j, Tn) if G.full_stage(key) else [Tn - 1]
            hi = float(g.max()) + 1e-12
            for l in stages:
                maps = G.R2.maps_of(g[l - j])
                img = G.A.image_scores(maps)
                s224 = G.maps224(maps).numpy()
                cell = {'i_auroc': float(roc_auc_score(y, img)),
                        'i_auroc_mapmax': float(roc_auc_score(y, maps.flatten(1).amax(1).numpy())),
                        'ap224': float(MX.exact_ap(list(s224), m224)['ap'])}
                if G.full_stage(key):
                    h = G.A.image_hists(maps, hws, masks, 0., hi)
                    cell['ap_native_binned'] = G.A.binned_ap(h[0].sum(0), h[1].sum(0))
                if l == Tn - 1:
                    cell['ap_native'] = float(G.A.exact_ap(maps, hws, masks))
                cells[f'{ds}|{key}|{l}|{c}'] = cell
    return cells


def metrics(ds, src):
    from multiprocessing import get_context
    G, o = RG.setup()
    t0 = time.monotonic()
    with get_context('spawn').Pool(len(G.order(ds))) as pool:
        parts = pool.map(_cat_cells, [(ds, c, j, src) for j, c in enumerate(G.order(ds))])
    cells = {k: v for p in parts for k, v in p.items()}
    G.A.wj(out() / 'p6' / f'CELLS_{src}_{ds}.json', cells)
    print('METRICS', ds, src, len(cells), round(time.monotonic() - t0))


def _macro(G, cells, ds):
    ords, Tn = G.order(ds), len(G.order(ds))
    keys = sorted({k.split('|')[1] + '|' + k.split('|')[2] for k in cells if k.startswith(ds + '|')})
    res = {}
    for key in keys:
        cv = lambda l, c, m: cells[f'{ds}|{key}|{l}|{c}'][m]
        r = {m: float(sum(cv(Tn - 1, c, m) for c in ords) / Tn) for m in ('ap_native', 'ap224', 'i_auroc',
                                                                           'i_auroc_mapmax')}
        if G.full_stage(key):
            mat = {(l, j): cv(l, c, 'ap_native_binned') for j, c in enumerate(ords) for l in range(j, Tn)}
            f = G.chaudhry(mat, Tn)
            r.update({'fm_native_binned': f['fm'], 'acq_minus_final': f['acq_minus_final'],
                      'acquisition_macro': float(sum(mat[j, j] for j in range(Tn)) / Tn)})
        res[key] = r
    return res


def mcmp():
    G, o = RG.setup()
    res, fails = {}, []
    for ds in ('mpdd', 'visa'):
        cn = G.A.rj(out() / 'p6' / f'CELLS_nanm_{ds}.json')
        co = G.A.rj(out() / 'p6' / f'CELLS_oracle_{ds}.json')
        sealed = G.A.rj(G.ROOT / 'results/hope_cad/bench_g3' / f'CELLS_{ds}.json')
        diffA = [k for k in co if cn.get(k) != co[k]] + [k for k in cn if k not in co]
        fails += [f'{ds}|{k}' for k in diffA]
        mo, mn = _macro(G, co, ds), _macro(G, cn, ds)
        sealed_macro = G.A.rj(G.ROOT / 'results/hope_cad/bench_g3' / f'RESULT_{ds}.json')['macro']
        B = {}
        for key, v in mo.items():
            sk = f'{ds}|{key}'
            s = sealed_macro.get(sk, {})
            B[key] = {m: {'oracle_here': v[m], 'sealed': s.get(m), 'delta': (v[m] - s[m]) if m in s else None}
                      for m in ('ap_native', 'ap224', 'i_auroc', 'i_auroc_mapmax')}
        Bcells = {}
        for k, v in co.items():
            if k in sealed:
                Bcells[k] = {m: v[m] - sealed[k][m] for m in v if m in sealed[k]}
        res[ds] = {'A_cells_compared': len(co), 'A_cells_identical': len(co) - len(diffA), 'A_macro_nanm': mn,
                   'A_macro_oracle': mo, 'A_macro_identical': mn == mo, 'B_macro_vs_sealed': B,
                   'B_cell_max_abs_delta': {m: max(abs(d[m]) for d in Bcells.values() if m in d)
                                            for m in ('ap_native', 'ap224', 'i_auroc', 'ap_native_binned')}}
    verdict = 'P6_SCIENTIFIC_PARITY_PASS' if not fails else 'P6_SCIENTIFIC_PARITY_FAIL'
    G.A.wj(out() / 'p6' / 'P6_METRICS.json', {'verdict_metric_part': verdict, 'failures': fails, 'datasets': res})
    print(verdict, len(fails))


# ------------------------------------------------------------------ P7 (measurement only)
def profile():
    import resource
    import numpy as np
    from nanm import FrozenPatchEncoder, G3Config, NANMSystem
    from nanm import geometry as NG
    from nanm import memory as NM
    from nanm.encoder import preprocess
    from nanm import state as NS
    dev = guard(); RG.numerics(); G, o = RG.setup()
    q = lambda ts: {'p50_ms': float(np.median(ts) * 1e3), 'p95_ms': float(np.quantile(ts, .95) * 1e3),
                    'mean_ms': float(np.mean(ts) * 1e3)}
    sync = torch.cuda.synchronize
    res = {'device': dev}
    B14 = G.B14()
    enc = FrozenPatchEncoder.load(B14.ENC_PATH, B14.ENC_SHA, 'cuda')
    rows = G.train_rows('visa', 'candle')
    paths = [G.ROOT / 'data/visa' / rows[i]['relative_path'] for i in G.roles('visa', rows)['EVAL'][:53]]
    for ds_bank, cfgname in (('visa', 'visa_G3_s2711'), ('mpdd', 'mpdd_G3_s2711')):
        nm = NANMSystem.load_state(CK / 'A' / cfgname, cfg=G3Config(seed=2711), device='cuda')
        pre, encs, ret, rd, e2e = [], [], [], [], []
        for k in range(53):
            p = paths[max(0, k - 3)]
            sync(); t0 = time.perf_counter(); x = preprocess(p)[None]; t1 = time.perf_counter()
            z = enc.tokens_from_batch(x.cuda()).cpu(); sync(); t2 = time.perf_counter()
            q16 = NM.unit(z.reshape(-1, 768)).half()
            e = nm.reader.energies(q16, nm._cache[3], nm._cache[4]).reshape(1, 28, 28); sync(); t3 = time.perf_counter()
            nm.readout(e); t4 = time.perf_counter()
            if k >= 3:
                pre.append(t1 - t0); encs.append(t2 - t1); ret.append(t3 - t2); rd.append(t4 - t3); e2e.append(t4 - t0)
        res[f'batch1_{cfgname}'] = {'preprocess': q(pre), 'encoder_fp32': q(encs), 'retrieval': q(ret),
                                    'readout': q(rd), 'end_to_end': q(e2e), 'bank_keys': sum(len(k) for k in nm.state.keys)}
    xb = torch.stack([preprocess(p) for p in paths[:32]])
    for bs in (8, 32):
        ts = []
        for _ in range(4):
            sync(); t0 = time.perf_counter()
            for i in range(0, 32, bs):
                enc.tokens_from_batch(xb[i:i + bs].cuda())
            sync(); ts.append(time.perf_counter() - t0)
        res[f'encoder_throughput_bs{bs}_img_per_s'] = 32 / float(np.median(ts[1:]))
    tok, _ = fit_inputs(G, 'visa', 'pcb1')                                    # largest VisA task: boundary costs
    nm = NANMSystem.create(G3Config(seed=2711), 'cuda')
    sync(); t0 = time.perf_counter(); fit16 = nm.writer.prepare(tok); sync(); t1 = time.perf_counter()
    k = NM.quota(0.01, len(fit16)); idx = nm.writer.write(fit16, k); sync(); t2 = time.perf_counter()
    S, n = nm.geometry.task_statistic(fit16, idx.cuda()); sync(); t3 = time.perf_counter()
    W64, cond = nm.geometry.geometry(S, n); t4 = time.perf_counter()
    res['boundary_visa_pcb1_single_task'] = {'prepare_s': t1 - t0, 'kcentre_s': t2 - t1, 'residual_s': t3 - t2,
                                             'geometry_s': t4 - t3, 'k': k, 'n_tokens': len(fit16)}
    fits = [json.loads((out() / 'p5' / f'A_{c}.json').read_text()) for c in ('visa_G3_s2711', 'mpdd_G3_s2711')]
    res['fit_task_wall_from_p5a'] = {c: sum(b['fit_s'] for b in f['boundaries'])
                                     for c, f in zip(('visa_G3_s2711', 'mpdd_G3_s2711'), fits)}
    res['save_s_from_p5a'] = {c: [round(b['save_s'], 4) for b in f['boundaries']][-1]
                              for c, f in zip(('visa_G3_s2711', 'mpdd_G3_s2711'), fits)}
    sizes = {}
    for cfgname in ('mpdd_G3_s2711', 'visa_G3_s2711', 'visa_G3atC_s2711'):
        root = CK / 'A' / cfgname
        g = NS.list_generations(root)[-1]
        t0 = time.perf_counter(); s = NANMSystem.load_state(root, cfg=G3Config(seed=2711), device='cuda'); sync()
        lt = time.perf_counter() - t0
        sizes[cfgname] = {'serialized_bytes': sum(p.stat().st_size for p in (root / g).iterdir()),
                          'payload_bytes': (root / g / NS.PAYLOAD).stat().st_size,
                          'manifest_bytes': (root / g / 'manifest.json').stat().st_size,
                          'load_and_rebuild_s': lt, 'ledger': s.state.ledger()}
    res['checkpoints'] = sizes
    res['encoder_param_bytes'] = sum(p.numel() * p.element_size() for p in enc.model.parameters())
    res['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
    res['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    G.A.wj(out() / 'p7' / 'PROFILE.json', res)
    print(json.dumps(res, default=str)[:1500])


if __name__ == '__main__':
    ph, *a = sys.argv[1:]
    conv = [int(x) if x.isdigit() else x for x in a]
    globals()[ph](*conv)
