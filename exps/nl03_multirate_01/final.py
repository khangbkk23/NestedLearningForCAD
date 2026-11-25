"""NL03-MULTIRATE-01 Phases L (FINAL LOCK), N (MVTec AD 2 confirmation) and O (final benchmark).

Real anomaly test data is touched ONLY after FINAL_LOCK.json exists and verifies. Every locked arm is scored once.

    python -B -m exps.nl03_multirate_01.final lock
    python -B -m exps.nl03_multirate_01.final bench <ds> [order]       # ds in mvtec | visa | mpdd
    python -B -m exps.nl03_multirate_01.final bench_metrics <ds> [order]
    python -B -m exps.nl03_multirate_01.final ad2_extract | ad2_run | ad2_package | ad2_metrics
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.nl03_multirate_01 import core as CO
from exps.nl03_multirate_01 import run as RN
from exps.nl03_multirate_01 import spec as SP

OUT, CACHE, ROOT = SP.OUT, SP.CACHE, SP.ROOT
FINAL = OUT / 'FINAL_LOCK.json'
MANIFEST = OUT / 'SOURCE_MANIFEST.json'
BENCH_OUT = OUT / 'bench'
AD2_OUT = OUT / 'ad2'
AD2_CACHE = CACHE / 'ad2'
CONF_PREFIX = ('conf',)
AD2_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
RULES = {
    'arm_set': 'U (canonical G3 unbounded), E (G3-E 8 MiB task boundary), CB (CB_SELECTION), F (flat streaming '
               'Euclidean 8 MiB = C5 rule), P (BMRM C=8 8 MiB; omitted only if NL03_PROTOCOL_BLOCKED)',
    'order_seed': 'order a, seed 2711 (confirmation); benchmark orders b, c for U, E, P only if budget allows',
    'training_normals': 'full train/good (100 %) in chronological (relative_path) order, including CALIB/EVAL roles '
                        'and the NL-02 SELECTED pool (released after this lock); tau recomputed from full-train counts',
    'ad2_stream': '8 scenarios, alphabetical, one CL stream, fit on train/ only',
    'ad2_threshold': 'per scenario and arm: theta = max A1 image score over validation/ images at the scenario '
                     'acquisition stage; binary map = pixel map > theta; no test split used',
    'ad2_maps': 'final-stage A1 maps (392) bilinearly resized to native resolution; float32 TIFF + binary PNG',
    'metrics': 'native pixel AP (binned, BENCH-G3 primitives), image AUROC (A1 top-1 %), FM, prequential mean, '
               'persistent bytes, update time, inference latency',
    'budget': 'N-O <= 8 GPU-h wall with auto-stop; priority AD2 package > AD2 test_public > MVTec AD > VisA > MPDD',
}


def sha(p):
    return RN.SY._fsha(p)


def log(*a):
    RN.log(*a)


# ------------------------------------------------------------------ Phase L
def arm_set():
    v = RN.rj(OUT / 'DEV_VERDICT.json')
    cb = RN.rj(OUT / 'CB_SELECTION.json')['CB']
    arms = ['U', 'E', 'CB', 'F'] + ([] if v['verdict'] == 'NL03_PROTOCOL_BLOCKED' else ['P'])
    return arms, cb, v['verdict']


def lock():
    assert not FINAL.exists(), 'FINAL_LOCK exists'
    from exps.nl03_multirate_01 import plan as PL
    PL.verify()
    arms, cb, verdict = arm_set()
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'exps/nl03_multirate_01').glob('*.py'))
    files += ['agents/tasks/task_nl03_multirate_01.md', 'agents/tasks/task_nl03_multirate_01_registration.md']
    arts = [f'results/hope_cad/nl03_multirate_01/{n}' for n in
            ('PREREG_LOCK.json', 'IMPL_LOCK.json', 'DEV_VERDICT.json', 'DEV_RESULTS.json', 'CB_SELECTION.json',
             'DEV_STATUS.json', 'AUDIT.json', 'TESTS.json', 'SMOKE.json', 'PROJECTION.json')]
    man = {f: sha(ROOT / f) for f in files + arts if (ROOT / f).exists()}
    canon = RN.rj(OUT / 'PREREG_LOCK.json')['canonical_sources']
    RN.wj(MANIFEST, {'files': man, 'canonical_sources': canon, 'time': time.strftime('%FT%T')})
    rec = {'arms': arms, 'CB': cb, 'dev_verdict': verdict, 'rules': RULES, 'manifest_sha256': sha(MANIFEST),
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           'time': time.strftime('%FT%T'), 'statement': 'nothing changes after this point'}
    with open(FINAL, 'x') as f:
        json.dump(rec, f, indent=1)
    log('FINAL_LOCK', sha(FINAL)[:16], arms, cb)


def verify_final():
    rec = RN.rj(FINAL)
    man = RN.rj(MANIFEST)
    assert sha(MANIFEST) == rec['manifest_sha256'], 'manifest changed'
    bad = [f for f, h in man['files'].items() if sha(ROOT / f) != h]
    bad += [f for f, h in man['canonical_sources'].items() if sha(ROOT / f) != h]
    assert not bad, f'FINAL_LOCK_FAIL {bad}'
    return rec


def conf_used():
    return RN.gpu_used(CONF_PREFIX)


class BudgetStop(RuntimeError):
    pass


def guard_budget(extra=0.0):
    if conf_used() + extra > SP.CONF_GPU_CEILING_S:
        raise BudgetStop(f'confirmation GPU ceiling reached ({conf_used() / 3600:.2f} h)')


# ------------------------------------------------------------------ stream learner over full train (U, E, CB, F, P)
def arm_impl(name, cb):
    return {'U': 'U', 'E': 'C0', 'CB': cb, 'F': 'C5', 'P': 'P'}[name]


class Learner:
    """Runs the locked arms over a sequence of task token sets; snapshots (keys fp16 CPU, W) at every stage."""

    def __init__(self, arms, cb, counts, seed=2711, budget=8 * SP.MiB):
        self.names = list(arms)
        self.cb, self.seed, self.budget = cb, seed, budget
        self.tau = SP.tau_of(counts)
        mb = [arm_impl(a, cb) for a in arms if arm_impl(a, cb) not in ('U', 'C0')]
        self.fasts, self.slows, self.bindk = CO.plan_arms(mb, budget, self.tau, seed)
        self.e_cap = (budget - SP.S_B) // SP.KEY_B          # C0 / G3-E rule (pressure-00 E) at the same budget
        self.e_K = torch.empty(0, SP.D, dtype=torch.float16, device='cuda')
        self.e_S, self.e_n = 0, 0
        self.u_blocks, self.u_S, self.u_n = [], 0, 0
        self.snap = {a: [] for a in arms}
        self.timing = {a: 0.0 for a in arms}
        self.bytes = {a: [] for a in arms}
        self.step = 0

    @torch.no_grad()
    def task(self, x16, l):
        import nanm.g3 as g3
        t0 = time.time()
        if 'U' in self.names or 'E' in self.names:
            k = g3.task_size(len(x16))
            st = g3.fit_task(x16, k, self.seed)
            dt = time.time() - t0
            if 'U' in self.names:
                self.u_blocks.append(x16[st.kc[:k].to(x16.device)].cpu()); self.u_S = self.u_S + st.S
                self.u_n += st.n; self.timing['U'] += dt
            if 'E' in self.names:                        # same canonical C_t / S_t as U, then kc_select to the cap
                from exps.nl03_pressure_00 import core as PC
                t1 = time.time()
                keys = x16[st.kc[:k].to(x16.device)]
                cand = torch.cat([self.e_K, keys])
                keep = PC.kc_select(torch.arange(len(cand)), self.e_cap, CO._Ids(cand), self.seed)
                self.e_K = cand[keep.to(cand.device)]
                self.e_S = self.e_S + st.S; self.e_n += st.n
                assert len(self.e_K) * SP.KEY_B + SP.S_B <= self.budget
                self.timing['E'] += dt + time.time() - t1
        need_cnt = any(f.counts for f in self.fasts.values())
        nimg = len(x16) // 784
        for b in range(0, nimg, SP.MB_IMAGES):
            xb = x16[b * 784:min(nimg, b + SP.MB_IMAGES) * 784]
            t1 = time.time()
            keys, S_b, n_b, cnt = CO.write_candidates(xb, self.seed, need_cnt)
            tc = time.time() - t1
            for k_, f in self.fasts.items():
                t2 = time.time()
                ev = f.step(keys, cnt, l, self.step)
                for arm, kk in self.bindk.items():
                    if kk == k_:
                        self.slows[arm].after_fast(f, ev, S_b, n_b, self.step)
                dt = time.time() - t2
                for a in self.names:
                    if arm_impl(a, self.cb) in self.bindk and self.bindk[arm_impl(a, self.cb)] == k_:
                        self.timing[a] += dt + tc
            self.step += 1
        self.snapshot()

    def state(self, a):
        from exps.nanm_r2b import run as B
        if a == 'U':
            W, _ = B.whiten(self.u_S, self.u_n)
            n = sum(len(b) for b in self.u_blocks)
            return ('U_prefix', len(self.u_blocks)), W.float(), n * SP.KEY_B + SP.S_B
        if a == 'E':
            W, _ = B.whiten(self.e_S, self.e_n)
            return self.e_K.cpu(), W.float(), len(self.e_K) * SP.KEY_B + SP.S_B
        arm = arm_impl(a, self.cb)
        f = self.fasts[self.bindk[arm]]; s = self.slows[arm]
        return f.K.cpu(), s.W(), SP.bytes_of(arm, len(f.K), s.n_pending())

    def snapshot(self):
        for a in self.names:
            K, W, b = self.state(a)
            self.snap[a].append((K, None if W is None else W.cpu()))
            self.bytes[a].append(b)

    def keys_at(self, a, l):
        K, W = self.snap[a][l]
        if isinstance(K, tuple):                          # U: union of the first n task blocks
            K = torch.cat(self.u_blocks[:K[1]])
        return K, W


def train_rows_sorted(G, task):
    from exps.nl03_pressure_00 import core as PC
    ds, c = task.split('/')
    PC.use(G, ds)
    return sorted(r['relative_path'] for r in G.train_rows(ds, c))


def bench_order(ds, ok='a'):
    from exps.nl03_pressure_00 import core as PC
    if ds == 'mvtec':
        from exps.calm02 import data as CD
        from exps.nanm_eval_01 import common as U
        od = U.orders(CD.MVTEC_ORDER)
        return [f'mvtec/{c}' for c in od[ok]]
    return PC.orders(ds)[ok]


def bench(ds, ok='a'):
    """Phase O scoring: learn over full train (100 %), then grids of the official test set at every stage >= j."""
    rec = verify_final()
    from exps.nl03_pressure_00 import core as PC
    tag = f'{ds}_{ok}2711'
    done = BENCH_OUT / f'SCORE_{tag}.json'
    if done.exists():
        log('SKIP', tag); return
    arms = rec['arms'] if ok == 'a' else [a for a in rec['arms'] if a in ('U', 'E', 'P')]
    G = PC.bind()
    order = bench_order(ds, ok)
    counts = [len(train_rows_sorted(G, t)) for t in order]
    t0 = time.time()
    try:
        guard_budget()
        L = Learner(arms, rec['CB'], counts)
        for l, t in enumerate(order):
            x16, n = RN.fit_tokens(G, t, order=train_rows_sorted(G, t))
            L.task(x16, l)
            del x16; torch.cuda.empty_cache()
            log('BENCH', tag, l, t, {a: L.bytes[a][-1] for a in arms})
            guard_budget(time.time() - t0)
        t_state = time.time() - t0
        ex = PC.extract_rec(G, ds)
        lat = {a: [] for a in arms}
        for j, t in enumerate(order):
            c = t.split('/')[1]
            PC.use(G, ds)
            q, rows = G.load_test(ds, c, ex)
            for a in arms:
                gl = []
                for l in range(j, len(order)):
                    K, W = L.keys_at(a, l)
                    t1 = time.time()
                    e = CO_energy(q, K, W)
                    lat[a].append((time.time() - t1) / len(rows))
                    gl.append(e)
                grid = torch.stack(gl).reshape(len(gl), len(rows), 28, 28).contiguous()
                p = CACHE / 'bench' / tag / a / f'{c}.pt'
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'rows': rows, 'grid': grid, 'stages': list(range(j, len(order))), 'task': t, 'pos': j,
                            'order': order}, p)
            del q; torch.cuda.empty_cache()
            guard_budget(time.time() - t0)
    finally:
        RN.ledger_add({'phase': 'conf_bench', 'cell': tag, 'seconds': time.time() - t0, 'gpu': True})
    RN.wj(done, {'tag': tag, 'order': order, 'arms': arms, 'CB': rec['CB'], 'tau': L.tau, 'counts': counts,
                 'bytes': L.bytes, 'update_s': L.timing, 'state_s': t_state, 'seconds': time.time() - t0,
                 'latency_ms_per_img_p50': {a: float(np.median(v) * 1e3) for a, v in lat.items()},
                 'peak_vram': torch.cuda.max_memory_allocated()})
    log('BENCH DONE', tag)


@torch.no_grad()
def CO_energy(q16, K16, W):
    from exps.bench_g3 import run as G0
    import nanm.g3 as g3
    keys = K16.cuda().float()
    return g3.score(q16, keys, W.cuda()) if W is not None else G0.euclid(q16, keys)


def bench_metrics(ds, ok='a'):
    """CPU metrics with the NL03-PRESSURE-00 evaluator (native pixel AP binned, image AUROC, bootstrap at the final
    stage) on the locked benchmark grids; pool of 2 workers."""
    from exps.nl03_pressure_00 import evaluate as PEV
    verify_final()
    tag = f'{ds}_{ok}2711'
    sc = RN.rj(BENCH_OUT / f'SCORE_{tag}.json')
    outp = BENCH_OUT / f'METRICS_{tag}.json'
    if outp.exists():
        return
    args = []
    for j, t in enumerate(sc['order']):
        c = t.split('/')[1]
        jobs = [(a, str(CACHE / 'bench' / tag / a / f'{c}.pt'), ok == 'a', False) for a in sc['arms']]
        args.append((t, j, jobs, 4))
    res, boots, info = {}, {}, {}
    t0 = time.time()
    with mp.get_context('spawn').Pool(2) as pool:
        for t, r, b, inf in pool.imap_unordered(PEV.run_task, args):
            res[t], boots[t], info[t] = r, b, inf
            log('BMETRIC', tag, t, '%.0fs' % inf['seconds'])
    bp = CACHE / 'bench' / f'boot_{tag}.npz'
    np.savez_compressed(bp, **{f'{t}||{a}||{m}': v[m] for t, bj in boots.items() for a, v in bj.items() for m in v})
    RN.wj(outp, {'tag': tag, 'cells': res, 'tasks': info, 'boot_file': str(bp), 'boot_sha256': sha(bp),
                 'seconds': time.time() - t0})
    RN.ledger_add({'phase': 'cpu_bench_metrics', 'cell': tag, 'seconds': time.time() - t0, 'gpu': False})


def bench_table(ds, ok='a'):
    tag = f'{ds}_{ok}2711'
    sc = RN.rj(BENCH_OUT / f'SCORE_{tag}.json')
    me = RN.rj(BENCH_OUT / f'METRICS_{tag}.json')
    order, T = sc['order'], len(sc['order'])
    bf = np.load(me['boot_file']) if ok == 'a' else None
    out = {}
    for a in sc['arms']:
        ap = {t: {int(l): v['ap'] for l, v in me['cells'][t][a].items()} for t in order}
        au = {t: {int(l): v['i_auroc'] for l, v in me['cells'][t][a].items()} for t in order}
        fin_ap = float(np.mean([ap[t][T - 1] for t in order]))
        fin_au = float(np.mean([au[t][T - 1] for t in order]))
        fm = float(np.mean([max(ap[t][l] for l in range(j, T - 1)) - ap[t][T - 1] for j, t in enumerate(order[:-1])]))
        preq = float(np.mean([np.mean([ap[order[j]][l] for j in range(l + 1)]) for l in range(T)]))
        acq = float(np.mean([ap[t][j] for j, t in enumerate(order)]))
        row = {'pixel_ap': fin_ap, 'image_auroc': fin_au, 'fm': fm, 'prequential_ap': preq, 'acq_ap': acq,
               'bytes_final': sc['bytes'][a][-1], 'bytes_max': max(sc['bytes'][a]), 'update_s': sc['update_s'][a],
               'latency_ms_per_img': sc['latency_ms_per_img_p50'][a], 'per_task_final_ap': {t: ap[t][T - 1] for t in order},
               'per_task_final_auroc': {t: au[t][T - 1] for t in order}}
        if bf is not None:
            reps = np.mean([bf[f'{t}||{a}||ap'] for t in order], axis=0)
            row['pixel_ap_ci95'] = [float(x) for x in np.nanpercentile(reps, [2.5, 97.5])]
            reps = np.mean([bf[f'{t}||{a}||auroc'] for t in order], axis=0)
            row['image_auroc_ci95'] = [float(x) for x in np.nanpercentile(reps, [2.5, 97.5])]
        out[a] = row
    if bf is not None and 'P' in out:
        for a in sc['arms']:
            if a == 'P':
                continue
            d = np.mean([bf[f'{t}||P||ap'] - bf[f'{t}||{a}||ap'] for t in order], axis=0)
            out['P'][f'ap_minus_{a}'] = {'delta': out['P']['pixel_ap'] - out[a]['pixel_ap'],
                                         'ci95': [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]}
    return out


# ------------------------------------------------------------------ Phase N: MVTec AD 2
def ad2_root():
    m = RN.rj(OUT / 'ad2' / 'AD2_MANIFEST.json')
    return Path(m['root']), m


def ad2_files(root, scen, split):
    base = root / scen / split
    fs = sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in AD2_EXT
                and 'ground_truth' not in p.relative_to(base).parts)
    return [str(p.relative_to(root)) for p in fs]


def ad2_mask_of(root, rel):
    """test_public ground truth: <scen>/test_public/ground_truth/bad/<stem>_mask.png (searched, not assumed)."""
    p = root / rel
    scen_split = root / Path(rel).parts[0] / 'test_public'
    stem = p.stem
    gt = scen_split / 'ground_truth'
    cands = [q for q in gt.rglob('*') if q.is_file() and q.stem in (stem, stem + '_mask')] if gt.exists() else []
    return cands[0] if cands else None


@torch.no_grad()
def ad2_extract():
    """Canonical front end + encoder for all AD 2 splits (images decoded only after the FINAL LOCK)."""
    verify_final()
    from exps.bench_g3 import run as G0
    from exps.dinomaly import encoder as E
    from exps.nanm_r0 import audit as A
    from exps.nar_b14_01 import run as b14
    root, man = ad2_root()
    enc = G0.encoder()
    t0 = time.time()
    try:
        for scen in sorted(man['per_scenario_split_counts']):
            for split in SP_AD2_SPLITS:
                p = AD2_CACHE / 'feat' / f'{scen}__{split}.pt'
                if p.exists():
                    continue
                rels = ad2_files(root, scen, split)
                toks, hws = [], []
                for i in range(0, len(rels), 8):
                    xs = []
                    for r in rels[i:i + 8]:
                        from PIL import Image
                        with Image.open(root / r) as im:
                            hws.append((im.size[1], im.size[0]))
                        xs.append(E.preprocess(root / r))
                    x = torch.stack(xs).cuda()
                    toks.append(A.unit(b14.tokens(enc, x).reshape(-1, SP.D)).half().reshape(len(xs), 784, SP.D).cpu())
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'rows': rels, 'hws': hws, 'tokens': torch.cat(toks) if toks else torch.empty(0, 784, SP.D)}, p)
                log('AD2 FEAT', scen, split, len(rels))
                guard_budget(time.time() - t0)
    finally:
        RN.ledger_add({'phase': 'conf_ad2_extract', 'seconds': time.time() - t0, 'gpu': True})


SP_AD2_SPLITS = ('train', 'validation', 'test_public', 'test_private', 'test_private_mixed')


def ad2_load(scen, split):
    return torch.load(AD2_CACHE / 'feat' / f'{scen}__{split}.pt', weights_only=False)


def ad2_run():
    """Fit the locked arms over the 8-scenario stream (train/ only, chronological), thresholds from validation/ at
    acquisition, test_public grids at every stage >= acquisition, final-stage grids for the private splits."""
    rec = verify_final()
    root, man = ad2_root()
    scens = sorted(man['per_scenario_split_counts'])
    done = AD2_OUT / 'AD2_SCORE.json'
    if done.exists():
        return
    arms = rec['arms']
    counts = [len(ad2_load(s, 'train')['rows']) for s in scens]
    L = Learner(arms, rec['CB'], counts)
    theta = {a: {} for a in arms}
    t0 = time.time()
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    try:
        for l, s in enumerate(scens):
            tr = ad2_load(s, 'train')
            x16 = tr['tokens'].reshape(-1, SP.D).cuda()
            L.task(x16, l)
            del x16, tr; torch.cuda.empty_cache()
            va = ad2_load(s, 'validation')
            for a in arms:
                K, W = L.keys_at(a, l)
                g = CO_energy(va['tokens'].reshape(-1, SP.D), K, W).reshape(len(va['rows']), 28, 28)
                theta[a][s] = float(A.image_scores(R2.maps_of(g)).max())
            log('AD2 STAGE', l, s, {a: round(theta[a][s], 4) for a in arms})
            guard_budget(time.time() - t0)
        for j, s in enumerate(scens):
            tp = ad2_load(s, 'test_public')
            for a in arms:
                gl = [CO_energy(tp['tokens'].reshape(-1, SP.D), *L.keys_at(a, l)) for l in range(j, len(scens))]
                grid = torch.stack(gl).reshape(len(gl), len(tp['rows']), 28, 28)
                p = AD2_CACHE / 'grids' / a / f'{s}__test_public.pt'
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'rows': tp['rows'], 'hws': tp['hws'], 'grid': grid, 'stages': list(range(j, len(scens)))}, p)
            for split in ('test_private', 'test_private_mixed'):
                d = ad2_load(s, split)
                for a in arms:
                    K, W = L.keys_at(a, len(scens) - 1)
                    g = CO_energy(d['tokens'].reshape(-1, SP.D), K, W).reshape(len(d['rows']), 28, 28)
                    p = AD2_CACHE / 'grids' / a / f'{s}__{split}.pt'
                    torch.save({'rows': d['rows'], 'hws': d['hws'], 'grid': g}, p)
            guard_budget(time.time() - t0)
    finally:
        RN.ledger_add({'phase': 'conf_ad2_run', 'seconds': time.time() - t0, 'gpu': True})
    RN.wj(done, {'scenarios': scens, 'arms': arms, 'CB': rec['CB'], 'tau': L.tau, 'counts': counts, 'theta': theta,
                 'bytes': L.bytes, 'update_s': L.timing, 'seconds': time.time() - t0})


def _native(m392, hw):
    import cv2
    return cv2.resize(m392.numpy().astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def ad2_package(arm_order=None):
    """Write AD2_SUBMISSION/<arm>/ in the documented layout (CODE_UTILS_MISSING unless official utils are on K:)."""
    import cv2
    from exps.nanm_r2 import run as R2
    rec = verify_final()
    sc = RN.rj(AD2_OUT / 'AD2_SCORE.json')
    utils = sorted(str(p) for p in Path('/mnt/k').iterdir()
                   if p.is_dir() and 'mvtec' in p.name.lower() and any(k in p.name.lower() for k in ('util', 'code', 'eval')))
    status = {'code_utils': utils or 'CODE_UTILS_MISSING', 'arms': {}}
    for a in arm_order or (['P'] + [x for x in sc['arms'] if x != 'P'] if 'P' in sc['arms'] else sc['arms']):
        base = AD2_OUT / 'AD2_SUBMISSION' / a
        if (base / 'DONE').exists():
            status['arms'][a] = 'DONE'; continue
        free = os.statvfs('/mnt/k').f_bavail * os.statvfs('/mnt/k').f_frsize
        n_written = 0
        for s in sc['scenarios']:
            for split in ('test_private', 'test_private_mixed'):
                d = torch.load(AD2_CACHE / 'grids' / a / f'{s}__{split}.pt', weights_only=False)
                maps = R2.maps_of(d['grid'])
                for r, hw, m in zip(d['rows'], d['hws'], maps):
                    nat = _native(m, hw)
                    name = Path(r).stem
                    pa = base / 'anomaly_images' / s / split / f'{name}.tiff'
                    pt = base / 'anomaly_images_thresholded' / s / split / f'{name}.png'
                    pa.parent.mkdir(parents=True, exist_ok=True); pt.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(pa), nat)
                    cv2.imwrite(str(pt), ((nat > sc['theta'][a][s]) * 255).astype(np.uint8))
                    n_written += 1
                if os.statvfs('/mnt/e').f_bavail * os.statvfs('/mnt/e').f_frsize < 30 * 2**30:
                    raise SystemExit('STOP: E: free space below 30 GiB during packaging')
        (base / 'README.md').write_text(README.format(arm=a, impl=arm_impl(a, rec['CB']), utils=status['code_utils']))
        (base / 'DONE').write_text(time.strftime('%FT%T'))
        status['arms'][a] = f'DONE ({n_written} maps)'
        log('AD2 PACKAGE', a, n_written)
    RN.wj(AD2_OUT / 'AD2_PACKAGE.json', status)


README = """# MVTec AD 2 server submission: arm {arm} (NL03-MULTIRATE-01, HELD_OUT_CONFIRMATION)

- Arm: {arm} (implementation `{impl}`); locked at FINAL_LOCK; order = 8 scenarios alphabetical; seed 2711; fit on train/ only.
- anomaly_images/<scenario>/<split>/<image>.tiff: float32 final-stage anomaly map at native resolution (A1 readout,
  392 map bilinearly resized).
- anomaly_images_thresholded/<scenario>/<split>/<image>.png: binary (0/255), pixel map > theta, theta = max A1 image
  score over that scenario's validation/ images at its acquisition stage (registered before any test access).
- Official code utils: {utils}. If CODE_UTILS_MISSING: check the layout and dtype (the server may require float16
  TIFF) with the official `mvtec_ad_2_public` utilities before upload.
- Upload: the owner uploads to benchmark.mvtec.com. Claude never submits.
"""


def ad2_metrics():
    """test_public local scores (HELD_OUT_CONFIRMATION): native pixel AP (binned), image AUROC, per-stage forgetting."""
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    verify_final()
    root, man = ad2_root()
    sc = RN.rj(AD2_OUT / 'AD2_SCORE.json')
    scens = sc['scenarios']
    out = {a: {} for a in sc['arms']}
    for s in scens:
        d0 = torch.load(AD2_CACHE / 'grids' / sc['arms'][0] / f'{s}__test_public.pt', weights_only=False)
        rows, hws = d0['rows'], d0['hws']
        y = np.array([0 if '/good/' in f'/{r}' else 1 for r in rows])
        masks = []
        for r, hw in zip(rows, hws):
            mp_ = ad2_mask_of(root, r)
            masks.append(np.zeros(hw, np.uint8) if mp_ is None else
                         (np.asarray(Image.open(mp_).convert('L')) > 0).astype(np.uint8))
        for a in sc['arms']:
            d = torch.load(AD2_CACHE / 'grids' / a / f'{s}__test_public.pt', weights_only=False)
            per = {}
            for i, l in enumerate(d['stages']):
                maps = R2.maps_of(d['grid'][i])
                img = A.image_scores(maps)
                pos, tot = A.image_hists(maps, hws, masks, 0., float(maps.max()) + 1e-12)
                per[l] = {'pixel_ap': A.binned_ap(pos.sum(0), tot.sum(0)),
                          'image_auroc': float(roc_auc_score(y, img)) if 0 < y.sum() < len(y) else None}
            out[a][s] = per
        log('AD2 METRIC', s)
    T = len(scens)
    tab = {}
    for a in sc['arms']:
        fin = [out[a][s][T - 1] for s in scens]
        fm = [max(out[a][s][l]['pixel_ap'] for l in range(j, T - 1)) - out[a][s][T - 1]['pixel_ap']
              for j, s in enumerate(scens[:-1])]
        tab[a] = {'pixel_ap': float(np.mean([f['pixel_ap'] for f in fin])),
                  'image_auroc': float(np.mean([f['image_auroc'] for f in fin if f['image_auroc'] is not None])),
                  'fm': float(np.mean(fm)), 'per_scenario_final': {s: out[a][s][T - 1] for s in scens},
                  'au_pro': 'NOT_COMPUTED (official code utils absent)'}
    RN.wj(AD2_OUT / 'AD2_TEST_PUBLIC.json', {'label': 'HELD_OUT_CONFIRMATION', 'table': tab, 'per_stage': out})


def smoke_learner():
    """Pre-lock check on DEVELOPMENT normal data only (VisA FIT_dev, 2 tasks): Learner bytes, E == C0 rule (bitwise),
    U == canonical union, energies on CALIB tokens. No test data."""
    from exps.nl03_pressure_00 import core as PC
    G = PC.bind()
    order = RN.orders('visa')['a'][:2]
    cb = RN.rj(OUT / 'CB_SELECTION.json')['CB']
    L = Learner(['U', 'E', 'CB', 'F', 'P'], cb, RN.fit_dev_counts('visa'))
    c0 = CO.TaskLevel(8 * SP.MiB, 2711)
    t0 = time.time()
    for l, t in enumerate(order):
        x16, n = RN.fit_tokens(G, t)
        L.task(x16, l); c0.boundary(x16, l)
        del x16
    out = {'E_equals_C0_keys': bool(torch.equal(L.e_K, c0.K)),
           'E_W_close_C0': bool(torch.allclose(L.state('E')[1], c0.W(), rtol=1e-4, atol=1e-5)),
           'bytes': L.bytes, 'U_keys': sum(len(b) for b in L.u_blocks)}
    Q = RN.SY.load(order[0])
    for a in L.names:
        K, W = L.keys_at(a, 1)
        e = CO_energy(Q['calib'].reshape(-1, SP.D), K, W)
        out[f'energy_finite_{a}'] = bool(torch.isfinite(e).all())
    out['bounded_ok'] = all(max(v) <= 8 * SP.MiB for a, v in L.bytes.items() if a != 'U')
    out['seconds'] = time.time() - t0
    RN.ledger_add({'phase': 'dev_final_smoke', 'seconds': time.time() - t0, 'gpu': True})
    RN.wj(OUT / 'FINAL_SMOKE.json', out)
    log('FINAL SMOKE', out)
    assert out['E_equals_C0_keys'] and out['bounded_ok'] and all(v for k, v in out.items() if k.startswith('energy'))


if __name__ == '__main__':
    ph, rest = sys.argv[1], sys.argv[2:]
    {'smoke_learner': smoke_learner, 'lock': lock, 'bench': bench, 'bench_metrics': bench_metrics, 'ad2_extract': ad2_extract, 'ad2_run': ad2_run,
     'ad2_package': ad2_package, 'ad2_metrics': ad2_metrics}[ph](*rest)
