"""NANM-SYS-01 R1/R2: regenerate missing producer artifacts with the UNCHANGED BENCH-G3 oracle.

Owner authorization 2026-11-20 (R1 approved, R2 conditional on R1 PASS, R3 deferred). Registration:
agents/tasks/task_nanm_sys_01.md section 17. This harness only orchestrates: every scientific operation is a call into
the sealed, hash-locked `exps.bench_g3.run` (G) and its imports. Module attributes of G are rebound IN-PROCESS ONLY so
that nothing is read from or written to the sealed result/cache namespaces. No label, mask or label column is read.

Run with (GPU 0 is forbidden; the harness checks the PCI bus):
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_sys.regen preflight
    python -B -m exps.nanm_sys.regen extract_train
    python -B -m exps.nanm_sys.regen encoder_check
    python -B -m exps.nanm_sys.regen normal --run A        # then --run B (separate process)
    python -B -m exps.nanm_sys.regen compare_r1
    python -B -m exps.nanm_sys.regen extract_test          # R2, only after R1 PASS
    python -B -m exps.nanm_sys.regen score --seed 2711     # then 2712, 2713 within caps
    python -B -m exps.nanm_sys.regen compare_r2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

ALLOWED_BUS = {0x25: 1, 0x41: 2}          # physical GPU 1 and GPU 2 (nvidia-smi PCI order); GPU 0 = bus 0x01
CACHE = Path('/media/RAID5Array/backup_home/tindd4/nlcad/hope_cad_cache/nanm_sys_regen')
FIRST16 = 16
REL_FP = 1e-5                             # registered R2 FP_LEVEL bound (descriptive, Tier C)


def guard_env(gpu):
    if not gpu:
        return None
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('1', '2'):
        raise SystemExit('set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=1 or 2 (GPU 0 forbidden)')
    import torch
    p = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or p.pci_bus_id not in ALLOWED_BUS:
        raise SystemExit(f'refusing device: bus {p.pci_bus_id:#x}')
    free, total = torch.cuda.mem_get_info(0)
    if free < 6.5 * 2**30:
        raise SystemExit(f'RESOURCE_BLOCKED: only {free / 2**30:.1f} GiB free on GPU {ALLOWED_BUS[p.pci_bus_id]}')
    torch.cuda.set_per_process_memory_fraction(5 * 2**30 / total)
    return {'physical_gpu': ALLOWED_BUS[p.pci_bus_id], 'pci_bus_id': p.pci_bus_id, 'name': p.name,
            'free_gib_at_start': free / 2**30}


def setup(run=None):
    """Import the sealed oracle and rebind its namespaces to the regeneration directories (in-process only)."""
    from exps.bench_g3 import run as G
    G.verify_lock()
    out = G.ROOT / 'results/hope_cad/nanm_sys/regen'
    G.FEAT = CACHE / 'features'
    if run:
        G.OUT = out / f'r1_run{run}'
        G.NORM = CACHE / f'normal_run{run}'
    else:
        G.OUT = out / 'unused_out'          # nothing must be written through G.OUT outside the normal phase
        G.NORM = CACHE / 'normal_runA'      # canonical oracle-here state (read-only after R1)
    G.load_train = _load_train(G)
    G.train_rows = _train_rows(G)
    return G, out


def _train_rows(G):
    """Label-free train identities: identical to the train halves of CD._mvtec / CD._visa."""
    visa_csv = G.ROOT / 'data/visa/split_csv/1cls.csv'

    def train_rows(ds, c):
        if ds == 'mpdd':
            return [{'relative_path': p.relative_to(G.MPDD).as_posix()}
                    for p in sorted((G.MPDD / c / 'train' / 'good').glob('*.png'))]
        with open(visa_csv) as f:
            mine = [{'image': r['image']} for r in csv.DictReader(f) if r['object'] == c and r['split'] == 'train']
        return [{'relative_path': r['image']} for r in sorted(mine, key=lambda r: r['image'])]
    return train_rows


def _load_train(G):
    import torch

    def load_train(ds, c):                  # same checks as G.load_train's MPDD branch, regenerated cache
        p = G.FEAT / f'{ds}_{c}_train.pt'
        G.K.require(G.A.sha(p) == G._feat_meta(p)['sha256'], f'regenerated train cache changed: {ds} {c}')
        data = torch.load(p, weights_only=True, mmap=True)
        G.K.require([r['relative_path'] for r in data['rows']] == [r['relative_path'] for r in G.train_rows(ds, c)],
                    'rows')
        return data
    return load_train


def numerics(threads=2):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(threads)


def env_record():
    import platform
    import sys
    import torch
    return {'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'threads': torch.get_num_threads(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER')}


def stamp(out, name, rec):
    import torch
    rec = dict(rec, env=env_record(), finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    if torch.cuda.is_initialized():
        rec['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
    p = out / 'logs' / f'{name}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
    print('WROTE', p, flush=True)


# ------------------------------------------------------------------ R1
def preflight(dev):
    G, out = setup()
    man = G.A.rj(G.ROOT / 'results/hope_cad/bench_g3/MPDD_MANIFEST.json')['counts']
    vb = G.A.rj(G.ROOT / 'results/hope_cad/vbench_00/LOCK.json')['inputs']
    counts = {}
    for c in G.MPDD_ORDER:
        rows = G.train_rows('mpdd', c)
        G.K.require(len(rows) == man[c]['train_good'], f'MPDD train count {c}')
        G.K.require(len(G.fit_index('mpdd', rows)) == man[c]['n_fit_images'], f'MPDD FIT count {c}')
        counts[f'mpdd/{c}'] = len(rows)
    for c in G.order('visa'):
        rows = G.train_rows('visa', c)
        G.K.require(len(rows) == vb[c]['n'], f'VisA train count {c}')
        counts[f'visa/{c}'] = len(rows)
    sealed = {ds: G.A.rj(G.ROOT / f'results/hope_cad/bench_g3/normal_{ds}.json')['sizes'] for ds in G.DSETS}
    for ds in G.DSETS:
        G.K.require(G.n_fit_list(ds) == sealed[ds]['n_fit'], f'n_fit {ds}')
        G.K.require(G.sizes_for(ds) == sealed[ds], f'sizes {ds}')
    stamp(out, 'preflight', {'train_counts': counts, 'sizes_equal_sealed': True, 'device': dev})


def extract_train(dev):
    G, out = setup()
    enc = G.encoder()                       # R.configure(): deterministic, TF32 off, 2 threads, 5 GiB; sha check
    rec = {}
    for ds in G.DSETS:
        root = G.MPDD if ds == 'mpdd' else G.ROOT / 'data/visa'
        for c in G.order(ds):
            p = G.FEAT / f'{ds}_{c}_train.pt'
            if p.exists() and p.with_suffix('.json').exists():
                G.K.require(G.A.sha(p) == G._feat_meta(p)['sha256'], f'existing cache inconsistent {p.name}')
                continue
            rec[f'{ds}/{c}'] = G._extract(enc, root, G.train_rows(ds, c), p)
            G.A.log('EXTRACT train', ds, c, round(rec[f'{ds}/{c}'], 1))
    stamp(out, 'extract_train', {'seconds': rec, 'device': dev})


def encoder_check(dev):
    import torch
    G, out = setup()
    enc = G.encoder()                       # independently loaded encoder, new process
    rows = G.train_rows('mpdd', 'bracket_black')[:FIRST16]
    p = CACHE / 'encoder_check' / 'mpdd_bracket_black_first16.pt'
    if p.exists():
        p.unlink(); p.with_suffix('.json').unlink()
    G._extract(enc, G.MPDD, rows, p)
    a = torch.load(p, weights_only=True)['features']
    b = torch.load(G.FEAT / 'mpdd_bracket_black_train.pt', weights_only=True, mmap=True)['features'][:FIRST16]
    eq = bool(torch.equal(a, b))
    stamp(out, 'encoder_check', {'images': [r['relative_path'] for r in rows], 'bitwise_equal': eq,
                                 'max_abs_diff': float((a - b).abs().max()), 'device': dev})
    G.K.require(eq, 'ENCODER DETERMINISM FAILED')


def normal(dev, run):
    numerics()
    G, out = setup(run)
    G.NORM.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for ds in G.DSETS:
        if not (G.OUT / f'normal_{ds}.json').exists():
            G._normal_ds(ds)                # unchanged sealed function
    stamp(out, f'normal_run{run}', {'seconds': time.monotonic() - t0, 'device': dev})


def _cmp_obj(a, b, path, bad):
    import torch
    if isinstance(a, torch.Tensor):
        if not (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)):
            bad.append(path)
        elif a.is_floating_point() and not bool(torch.isfinite(a).all()):
            bad.append(path + ' (non-finite)')
    elif isinstance(a, dict):
        if set(a) != set(b):
            bad.append(path + ' keys')
        for k in a:
            if k in b:
                _cmp_obj(a[k], b[k], f'{path}.{k}', bad)
    elif a != b:
        bad.append(path)


def compare_r1(dev):
    import torch
    G, out = setup()
    res = {'files_compared': 0, 'tensor_mismatches': [], 'json': {}, 'sealed': {}}
    for ds in G.DSETS:
        ja, jb = (G.A.rj(out / f'r1_run{r}' / f'normal_{ds}.json') for r in 'AB')
        sealed = G.A.rj(G.ROOT / f'results/hope_cad/bench_g3/normal_{ds}.json')
        strip = lambda j: {k: v for k, v in j.items() if k not in ('seconds', 'seeds')}
        conds_equal = all(ja['seeds'][s]['cond'] == jb['seeds'][s]['cond'] for s in ja['seeds'])
        res['json'][ds] = {'A_eq_B_excluding_seconds_and_file_hashes': strip(ja) == strip(jb) and conds_equal}
        cond_rel = {}
        for s in sealed['seeds']:
            for st, arms in sealed['seeds'][s]['cond'].items():
                for arm, v in arms.items():
                    cond_rel[f's{s}|stage{st}|{arm}'] = abs(ja['seeds'][s]['cond'][st][arm] - v) / v
        res['sealed'][ds] = {'sizes_equal': ja['sizes'] == sealed['sizes'], 'bytes_equal': ja['bytes'] == sealed['bytes'],
                             'cond_rel_diff_max': max(cond_rel.values()), 'cond_rel_diff': cond_rel}
        for s in ja['seeds']:
            for name in ja['seeds'][s]['files']:
                a = torch.load(CACHE / 'normal_runA' / name, weights_only=True)
                b = torch.load(CACHE / 'normal_runB' / name, weights_only=True)
                _cmp_obj(a, b, name, res['tensor_mismatches'])
                res['files_compared'] += 1
                if name.endswith('_stage.pt'):
                    continue
                n_keys = len(a['kc'])
                G.K.require(len(a['keys']) == n_keys and int(a['n1']) == a['n'] - G.sizes_for(ds)['g3'][
                    G.order(ds).index(name.split('_s')[1].split('_', 1)[1][:-3])], f'n1 semantics {name}')
    man = {}
    for d in (CACHE / 'features', CACHE / 'normal_runA', CACHE / 'normal_runB', CACHE / 'encoder_check',
              out / 'r1_runA', out / 'r1_runB'):
        for p in sorted(d.rglob('*')):
            if p.is_file():
                man[str(p)] = {'sha256': G.A.sha(p), 'bytes': p.stat().st_size}
    res['pass_determinism'] = not res['tensor_mismatches'] and all(v['A_eq_B_excluding_seconds_and_file_hashes']
                                                                   for v in res['json'].values())
    res['pass_sealed_integers'] = all(v['sizes_equal'] and v['bytes_equal'] for v in res['sealed'].values())
    G.A.wj(out / 'r1' / 'MANIFEST.json', man)
    stamp(out / 'r1', 'R1_RESULT', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'sealed'}, default=str)[:2000])
    print({ds: {k: v for k, v in r.items() if k != 'cond_rel_diff'} for ds, r in res['sealed'].items()})


# ------------------------------------------------------------------ R2 (image-only)
def _sealed_scores(G):
    sc = G.A.rj(G.ROOT / 'results/hope_cad/bench_g3/SCORES_COMPLETE.json')['files']
    return {c: Path.home() / 'hope_cad_cache/bench_g3/scores' / f'visa_{c}.pt' for c in G.order('visa')}, sc


def extract_test(dev):
    import torch
    G, out = setup()
    r1 = G.A.rj(out / 'r1' / 'logs' / 'R1_RESULT.json')
    enc_ok = G.A.rj(out / 'logs' / 'encoder_check.json')['bitwise_equal']
    G.K.require(r1['pass_determinism'] and r1['pass_sealed_integers'] and enc_ok, 'R1 not PASS: R2 not allowed')
    paths, sc = _sealed_scores(G)
    tm = {}
    for c, p in paths.items():
        G.K.require(G.A.sha(p) == sc[p.name], f'sealed score hash {p.name}')
        tm[c] = list(torch.load(p, weights_only=True, mmap=True)['rows'])
    lk = G.A.rj(G.ROOT / 'results/hope_cad/bench_g3/LOCK.json')      # verified in setup() before rebinding
    G.K.require(G.K_text(tm) == lk['test_manifest_sha256']['visa'], 'image-only manifest != LOCK test manifest')
    enc = G.encoder()
    files = {}
    for c in G.order('visa'):
        p = G.FEAT / f'visa_{c}_test.pt'
        if not (p.exists() and p.with_suffix('.json').exists()):
            G._extract(enc, G.ROOT / 'data/visa', [{'relative_path': r} for r in tm[c]], p)
        files[p.name] = G._feat_meta(p)['sha256']
        G.K.require(G.A.sha(p) == files[p.name], 'test cache')
    G.A.wj(out / 'r2' / 'EXTRACT_COMPLETE.json', {'files': files, 'manifest_sha256': G.K_text(tm)})
    stamp(out / 'r2', 'extract_test', {'n_images': {c: len(v) for c, v in tm.items()}, 'device': dev})


def score(dev, seed):
    """Exactly the VisA loop structure of G.score for arms G3/G1b/G3@C (whitened) and A0b (euclid); no E-full/A0@C."""
    import torch
    numerics()
    G, out = setup()
    ex = G.A.rj(out / 'r2' / 'EXTRACT_COMPLETE.json')
    rec = {'visa': G.A.rj(out / 'r1_runA' / 'normal_visa.json')}
    ds, ords, Tn = 'visa', G.order('visa'), 12
    sz = rec[ds]['sizes']
    t0 = time.monotonic()
    with torch.no_grad():
        Q = {c: G.load_test(ds, c, ex) for c in ords}
        bank_key = {'G3': 'g3', 'G1b': 'g3', 'G3@C': 'g3c', 'A0b': 'a0b'}
        out_g = {c: {} for c in ords}
        per = {c: torch.empty(Tn, len(Q[c][0])) for c in ords}
        store = [G.load_norm(ds, seed, m, rec) for m in ords]
        W = G.load_stage(ds, seed, rec)
        for a in ('A0b',):
            for t, m in enumerate(ords):
                keys = store[t]['keys'][:sz[bank_key[a]][t]].cuda().float()
                for c in ords:
                    G.euclid(Q[c][0], keys, per[c][t])
            for j, c in enumerate(ords):
                out_g[c][f'{a}|s{seed}'] = torch.stack([per[c][:l + 1].amin(0) for l in range(j, Tn)])
        for a in ('G3', 'G1b', 'G3@C'):
            stg = {c: [] for c in ords}
            for l in range(Tn):
                keys = torch.cat([x['keys'][:sz[bank_key[a]][t]] for t, x in enumerate(store[:l + 1])]).cuda()
                Wg = W[l][a].cuda()
                for c in ords[:l + 1]:
                    stg[c].append(G.whitened(Q[c][0], keys.float(), Wg))
            for c in ords:
                out_g[c][f'{a}|s{seed}'] = torch.stack(stg[c])
            G.A.log('SCORE', a, seed, round(time.monotonic() - t0, 1)); G.A.caps()
        tie = neartie(G, Q, store, W, sz, ords) if seed == 2711 else None
        files = {}
        d = CACHE / 'scores'
        d.mkdir(parents=True, exist_ok=True)
        for j, c in enumerate(ords):
            n = len(Q[c][1])
            g = {k: v.reshape(len(v), n, 28, 28) for k, v in out_g[c].items()}
            G.K.require(all(len(v) == Tn - j and bool(torch.isfinite(v).all()) for v in g.values()), f'grid {c}')
            p = d / f'visa_{c}_s{seed}.pt'
            torch.save({'rows': Q[c][1], 'first_stage': j, 'grids': g}, p)
            files[p.name] = G.A.sha(p)
    G.A.wj(out / 'r2' / f'SCORES_s{seed}.json', {'files': files, 'seconds': time.monotonic() - t0, 'neartie': tie})
    stamp(out / 'r2', f'score_s{seed}', {'seconds': time.monotonic() - t0, 'device': dev})


def neartie(G, Q, store, W, sz, ords):
    """Label-free diagnostic: share of query patches whose whitened top-2 exemplar gap < 2^-10 * top-1 (G3 final)."""
    import torch
    Wg = W[len(ords) - 1]['G3'].cuda()
    mw = torch.cat([x['keys'][:sz['g3'][t]] for t, x in enumerate(store)]).cuda().float() @ Wg
    res = {}
    for c in ords:
        q16, tot, close = Q[c][0], 0, 0
        for i in range(0, len(q16), G.QC):
            v = G.R2.sqdist(q16[i:i + G.QC].cuda().float() @ Wg, mw).topk(2, dim=1, largest=False).values
            close += int(((v[:, 1] - v[:, 0]) < v[:, 0] * 2 ** -10).sum()); tot += len(v)
        res[c] = {'patches': tot, 'near_ties': close}
    return res


def compare_r2(dev):
    import torch
    G, out = setup()
    paths, sc = _sealed_scores(G)
    seeds = sorted(int(p.stem.split('_s')[1]) for p in (out / 'r2').glob('SCORES_s*.json'))
    rows, summary = {}, {}
    for c, p in paths.items():
        G.K.require(G.A.sha(p) == sc[p.name], 'sealed score hash')
        sealed = torch.load(p, weights_only=True)
        for seed in seeds:
            regen = torch.load(CACHE / 'scores' / f'visa_{c}_s{seed}.pt', weights_only=True)
            G.K.require(regen['rows'] == sealed['rows'] and regen['first_stage'] == sealed['first_stage'], 'rows')
            for key, r in regen['grids'].items():
                s = sealed['grids'][key]
                G.K.require(r.shape == s.shape and r.dtype == s.dtype, f'shape/dtype {c} {key}')
                d = (r - s).abs()
                rel = d / s.abs().clamp_min(1e-30)
                per_stage = [{'stage': sealed['first_stage'] + i, 'max_abs': float(d[i].max()),
                              'max_rel': float(rel[i].max()), 'frac_equal': float((d[i] == 0).float().mean())}
                             for i in range(len(r))]
                q = torch.quantile(rel.flatten()[torch.randperm(rel.numel(), generator=torch.Generator().manual_seed(0))[
                    :2_000_000]].double(), torch.tensor([.5, .99, .999], dtype=torch.float64))
                rows[f'{c}|{key}'] = {'frac_equal': float((d == 0).float().mean()), 'max_abs': float(d.max()),
                                      'max_rel': float(rel.max()), 'rel_q50_q99_q999': q.tolist(),
                                      'sealed_max': float(s.max()), 'per_stage': per_stage}
    for key in sorted({k.split('|', 1)[1] for k in rows}):
        cells = [v for k, v in rows.items() if k.split('|', 1)[1] == key]
        mx = max(v['max_rel'] for v in cells)
        summary[key] = {'max_rel': mx, 'max_abs': max(v['max_abs'] for v in cells),
                        'frac_equal_min': min(v['frac_equal'] for v in cells),
                        'class': 'BITWISE' if all(v['frac_equal'] == 1 for v in cells)
                        else 'FP_LEVEL' if mx <= REL_FP else 'DIVERGENT'}
    anchor = 'ESTABLISHED' if all(v['class'] != 'DIVERGENT' for v in summary.values()) else \
        'CROSS-MACHINE ANCHOR UNRESOLVED'
    G.A.wj(out / 'r2' / 'R2_COMPARE.json', {'summary': summary, 'cells': rows, 'anchor': anchor, 'seeds': seeds,
                                            'rel_fp_bound': REL_FP})
    stamp(out / 'r2', 'compare_r2', {'anchor': anchor})
    print(json.dumps(summary, indent=1), anchor)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('preflight', 'extract_train', 'encoder_check', 'normal', 'compare_r1',
                                      'extract_test', 'score', 'compare_r2'))
    ap.add_argument('--run', choices=('A', 'B'))
    ap.add_argument('--seed', type=int, choices=(2711, 2712, 2713))
    a = ap.parse_args()
    gpu = a.phase in ('extract_train', 'encoder_check', 'normal', 'extract_test', 'score')
    dev = guard_env(gpu)
    {'preflight': lambda: preflight(dev), 'extract_train': lambda: extract_train(dev),
     'encoder_check': lambda: encoder_check(dev), 'normal': lambda: normal(dev, a.run),
     'compare_r1': lambda: compare_r1(dev), 'extract_test': lambda: extract_test(dev),
     'score': lambda: score(dev, a.seed), 'compare_r2': lambda: compare_r2(dev)}[a.phase]()
