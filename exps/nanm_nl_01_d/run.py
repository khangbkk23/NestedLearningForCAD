"""NANM-NL-01D driver. Phases: audit -> lock -> run. Protocol: agents/tasks/task_nanm_nl_01_d.md (frozen).

    source docs/recovery/recovery_env.sh
    python -B -m exps.nanm_nl_01_d.run audit|lock|run

Writes only results/hope_cad/nanm_nl_01_d/ (E:). Reads the NANM-REPRO-01 caches on K: read-only (mmap); no
derived token cache is written anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from exps.nanm_nl_01_d import core as C
from exps.nanm_r0 import audit as A
from nanm import g3

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/hope_cad/nanm_nl_01_d'
CACHE = Path('/mnt/k/NLCAD/cache/nanm_repro_01')
SRC = {'mpdd': CACHE / 'bench_g3/features', 'visa': CACHE / 'vbench_00/features'}
TASKFILE = ROOT / 'agents/tasks/task_nanm_nl_01_d.md'
CODE = [Path(__file__), ROOT / 'exps/nanm_nl_01_d/core.py', ROOT / 'exps/nanm_nl_01_d/test_core.py', TASKFILE]
VHDX = Path('/mnt/d/wsl/Ubuntu_new/ext4.vhdx')
SEEDS = (2711, 2712, 2713)
KS = (256, 512)
PERMS = {'a': (0, 1, 2, 3), 'b': (3, 2, 1, 0), 'c': (1, 3, 0, 2)}
ROLE_N = {'FIT': (16, 64), 'FEEDBACK': (4, 8), 'EVAL': (8, 256)}
GEOM_SEED = 2711
GiB = 2 ** 30
DEPLOY = ('B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6')
SCB_K_SET = ('B0', 'B1', 'B2', 'B3', 'B5')
SCB_FB_SET = SCB_K_SET + ('B6',)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fsha(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def storage():
    return {'D_free': shutil.disk_usage('/mnt/d').free, 'E_free': shutil.disk_usage('/mnt/e').free,
            'K_free': shutil.disk_usage('/mnt/k').free, 'VHDX': VHDX.stat().st_size, 'time': time.strftime('%FT%T')}


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------ data
def pool():
    out = [('mpdd', p.name[5:-9]) for p in sorted(SRC['mpdd'].glob('mpdd_*_train.pt'))]
    out += [('visa', p.name[:-9]) for p in sorted(SRC['visa'].glob('*_train.pt'))]
    return out


def fpath(ds, c):
    return SRC[ds] / (f'mpdd_{c}_train.pt' if ds == 'mpdd' else f'{c}_train.pt')


def task_id(ds, c):
    return f'{ds}/{c}'


def selection():
    ranked = sorted(pool(), key=lambda t: C.sha('NANM-NL-01D/tasks/v1|' + task_id(*t)))
    return ranked[:4], ranked[4:8], ranked


def roles(paths):
    order = sorted(paths, key=lambda p: C.sha('NANM-NL-01D/roles/v1|' + p))
    out, k = {}, 0
    for r, (n, _) in ROLE_N.items():
        out[r] = order[k:k + n]; k += n
    return out


def patch_idx(p, m):
    return sorted(range(784), key=lambda j: C.sha(f'NANM-NL-01D/patch/v1|{p}|{j}'))[:m]


def load_vectors(ds, c, want=('FIT', 'FEEDBACK', 'EVAL')):
    """Unit-norm fp16 vectors (G3 key convention) of the registered patches; returns dict role -> [n, 768] fp16."""
    d = torch.load(fpath(ds, c), mmap=True, map_location='cpu', weights_only=False)
    paths = [r['relative_path'] for r in d['rows']]
    pos = {p: i for i, p in enumerate(paths)}
    rl = roles(paths); out = {}
    for r in want:
        m = ROLE_N[r][1]; vs = []
        for p in rl[r]:
            vs.append(A.unit(d['features'][pos[p]][patch_idx(p, m)].float()).half())
        out[r] = torch.cat(vs)
    return out


def geometry(fit16):
    st = g3.fit_task(fit16, k=g3.task_size(len(fit16)), seed=GEOM_SEED)
    W, cond = g3.stage_whitener([st])
    return W.double(), float(cond)


# ------------------------------------------------------------------ phases
def audit():
    sel, des, ranked = selection()
    rec = {'storage_before': storage(), 'selected': [task_id(*t) for t in sel], 'design': [task_id(*t) for t in des],
           'ranked': [task_id(*t) for t in ranked]}
    rec['git_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec['git_status'] = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True).stdout
    cmp = json.loads((ROOT / 'results/hope_cad/nanm_repro_01/COMPARE.json').read_text())
    rec['g3_repro'] = {'gate2': cmp['gate2_tierA_mpdd_prefix']['pass'], 'gate4': cmp['gate4_tierA_full_streams']['pass'],
                       'scores_hist_equal': f"{cmp['tierH_scores']['n_equal']}/{cmp['tierH_scores']['n']}"}
    rec['oracle_lock_sha256'] = fsha(ROOT / 'results/hope_cad/bench_g3/LOCK.json')
    rec['nanm_g3_sha256'] = fsha(ROOT / 'nanm/g3.py')
    st = rec['storage_before']
    ok_storage = st['D_free'] >= 30 * GiB
    files, seen_paths, seen_content, problems = {}, {}, {}, []
    for grp, tasks in (('SELECTED', sel), ('DESIGN', des)):
        for ds, c in tasks:
            p = fpath(ds, c); meta = json.loads(p.with_suffix('.json').read_text())
            t0 = time.time(); h = fsha(p)
            d = torch.load(p, mmap=True, map_location='cpu', weights_only=False)
            paths = [r['relative_path'] for r in d['rows']]
            rl = roles(paths)
            n_need = sum(n for n, _ in ROLE_N.values())
            if len(paths) < n_need:
                problems.append(f'{ds}/{c}: only {len(paths)} images')
            for r, ps in rl.items():
                for q in ps:
                    key = (task_id(ds, c), q)
                    if q in seen_paths:
                        problems.append(f'path reused {q}')
                    seen_paths[q] = (grp, task_id(ds, c), r)
                    ch = meta['source_images'].get(q)
                    if ch is None:
                        problems.append(f'no content hash {q}')
                    elif ch in seen_content:
                        problems.append(f'content duplicate {q} ~ {seen_content[ch]}')
                    else:
                        seen_content[ch] = key + (r,)
            files[task_id(ds, c)] = {'group': grp, 'file': str(p), 'sha256': h, 'json_sha256': meta['sha256'],
                                     'equal': h == meta['sha256'], 'n_train_images': len(paths),
                                     'roles': rl, 'hash_seconds': round(time.time() - t0, 1)}
            log('custody', task_id(ds, c), grp, h == meta['sha256'], len(paths))
    rec['files'] = files
    rec['custody_pass'] = all(f['equal'] for f in files.values())
    rec['roles_problems'] = problems
    rec['storage_ok'] = ok_storage
    rec['pass'] = rec['custody_pass'] and not problems and ok_storage
    wjson(RUN / 'AUDIT.json', rec)
    log('AUDIT', 'pass' if rec['pass'] else 'FAIL', problems[:5], rec['selected'], rec['design'])


def lock():
    au = RUN / 'AUDIT.json'
    assert json.loads(au.read_text())['pass'], 'audit failed'
    assert not (RUN / 'RESULT.json').exists(), 'results already exist; lock must precede evaluation'
    rec = {'files': {str(p.relative_to(ROOT)): fsha(p) for p in CODE + [au]}, 'time': time.strftime('%FT%T'),
           'note': 'frozen before any EVAL metric or oracle inspection'}
    wjson(RUN / 'PREREG_LOCK.json', rec)
    log('LOCK', json.dumps(rec['files'], indent=0))


# ------------------------------------------------------------------ streams
class Stream:
    """One order of 4 tasks with fixed geometry. Policy-side data (FIT z, FEEDBACK z, metadata) and evaluator-side
    data (EVAL z, distance matrix) are held separately; only the former is ever passed to a policy."""

    def __init__(self, tasks, vec, nimg, with_eval=True):
        self.tasks = tasks
        self.W, self.cond = geometry(vec[tasks[0]]['FIT'])
        z = lambda v: v.double() @ self.W
        self.fit = [z(vec[t]['FIT']) for t in tasks]
        self.fb = [z(vec[t]['FEEDBACK']) for t in tasks]
        self.meta = [{'task': t, 'n_train_images': nimg[t]} for t in tasks]
        if with_eval:
            self._ev = torch.cat([z(vec[t]['EVAL']) for t in tasks])
            self._fitall = torch.cat(self.fit)
            self.D = C.sqdist(self._ev, self._fitall)                       # [8192, 4096] evaluator only
            self.lref = torch.stack([self.D[p * C.N_EV:(p + 1) * C.N_EV].min(1).values.mean() for p in range(4)])
            assert float(self.lref.min()) >= C.FLOOR, 'L_ref below registered floor'

    def ell(self, bank):
        cols = torch.cat([p * C.N_FIT + idx for p, idx in bank])
        m = self.D[:, cols].min(1).values.view(4, C.N_EV).mean(1)
        return m / self.lref


def make_policy(arm, K, seed, schedule):
    if arm in ('B3', 'B6'):
        return C.ContCorePolicy(K, seed, extra=C.b6_extra(K) if arm == 'B6' else None)
    return C.QuotaPolicy(arm, K, seed, schedule=schedule.get(K) if schedule else None)


def run_policy(stream, arm, K, seed, schedule, fit=None, fb=None):
    pol = make_policy(arm, K, seed, schedule)
    fit = fit or stream.fit; fb = fb or stream.fb
    banks, byts, t0 = [], [], time.process_time()
    for t in range(4):
        banks.append(pol.step(t, fit[t], stream.meta[t], P=fb[t] if arm == 'B4' else None))
        byts.append(pol.bytes(t))
    return pol, banks, byts, time.process_time() - t0


def cols(bank):
    return set(torch.cat([p * C.N_FIT + idx for p, idx in bank]).tolist())


def summarize(stream, banks):
    ell = torch.full((4, 4), float('nan'), dtype=torch.float64)
    for t, b in enumerate(banks):
        ell[t, :t + 1] = stream.ell(b)[:t + 1]
    hist = [float(ell[t, :t].mean()) for t in range(1, 4)]
    return {'ell': ell.tolist(), 'hist': hist, 'H': sum(hist) / 3, 'final_hist': hist[-1],
            'A': float(torch.stack([ell[t, t] for t in range(4)]).mean()),
            'quota': [[int(sum(len(idx) for p, idx in b if p == i)) for i in range(4)] for b in banks],
            'evicted': [0] + [len(cols(banks[t - 1]) - cols(banks[t])) for t in range(1, 4)]}


def design_schedule(des, vecd, nimg):
    """B5: B4 run on the DESIGN tasks (FIT + FEEDBACK only), mean quota shares per boundary and position."""
    traces = {K: [] for K in KS}
    for name, perm in PERMS.items():
        st = Stream([des[i] for i in perm], vecd, nimg, with_eval=False)
        for seed in SEEDS:
            for K in KS:
                pol, _, _, _ = run_policy(st, 'B4', K, seed, None)
                traces[K].append(pol.trace)
    sched = {K: [[sum(tr[t][p] for tr in traces[K]) / (len(traces[K]) * K) for p in range(t + 1)] for t in range(4)]
             for K in KS}
    return sched, traces


def oracle_C(stream, orders, K):
    u = K // C.UNITS; out = []
    for p in range(4):
        cm = torch.cummin(stream.D[:, p * C.N_FIT + orders[p]], 1).values            # [Nev, 1024]
        out.append(cm[:, torch.arange(1, C.UNITS + 1) * u - 1].T.contiguous())        # [UNITS, Nev]
    return out


def lattice_bank(orders, q):
    return [(p, orders[p][:qi]) for p, qi in enumerate(q)]


def writer_orders(stream, seed):
    return [C.ffs(stream.fit[p], C.N_FIT, start=C.start_index(seed, stream.tasks[p], C.N_FIT)) for p in range(4)]


def run():
    lk = json.loads((RUN / 'PREREG_LOCK.json').read_text())
    for f, h in lk['files'].items():
        assert fsha(ROOT / f) == h, f'lock mismatch {f}'
    au = json.loads((RUN / 'AUDIT.json').read_text())
    st0 = storage(); tstart = time.time()
    sel, des = au['selected'], au['design']
    nimg = {t: au['files'][t]['n_train_images'] for t in sel + des}
    vec = {t: load_vectors(*t.split('/')) for t in sel}
    vecd = {t: load_vectors(*t.split('/'), want=('FIT', 'FEEDBACK')) for t in des}       # DESIGN EVAL never loaded
    log('vectors loaded')
    sched, dtr = design_schedule(des, vecd, nimg)
    log('B5 schedule', json.dumps(sched))
    streams = {n: Stream([sel[i] for i in perm], vec, nimg) for n, perm in PERMS.items()}
    res = {'cells': {}, 'geometry_cond': {n: s.cond for n, s in streams.items()}, 'b5_schedule': sched,
           'b5_design_traces': dtr}
    pols = {}
    for n, s in streams.items():
        for seed in SEEDS:
            for K in KS:
                cell = f'{n}|{seed}|{K}'; res['cells'][cell] = {}
                for arm in DEPLOY:
                    pol, banks, byts, sec = run_policy(s, arm, K, seed, sched)
                    r = summarize(s, banks); r.update(bytes=byts, dist_evals=pol.ctr.dist, cpu_seconds=round(sec, 3))
                    if arm == 'B4':
                        r['guard_overrides'] = pol.guard_overrides
                    res['cells'][cell][arm] = r; pols[(cell, arm)] = (pol, banks)
                log('cell', cell, {a: round(res['cells'][cell][a]['H'], 4) for a in DEPLOY})
    # ---- strongest conventional baselines (registered: argmin mean_cells H)
    cells_of = lambda K: [c for c in res['cells'] if c.endswith(f'|{K}')]
    mean_H = lambda arm, K: sum(res['cells'][c][arm]['H'] for c in cells_of(K)) / 9
    scb = {K: {'K': min(SCB_K_SET, key=lambda a: mean_H(a, K)), 'fb': min(SCB_FB_SET, key=lambda a: mean_H(a, K))}
           for K in KS}
    res['scb'] = scb
    log('SCB', scb)
    # ---- privileged oracles
    for n, s in streams.items():
        for seed in SEEDS:
            orders = writer_orders(s, seed)
            for K in KS:
                cell = f'{n}|{seed}|{K}'; cr = res['cells'][cell]
                ref = [torch.tensor(cr[scb[K]['K']]['ell'][t], dtype=torch.float64) for t in range(4)]
                Cm = oracle_C(s, orders, K); u = K // C.UNITS
                path, _, flags = C.o1_dp(Cm, s.lref, ref)
                b1 = [lattice_bank(orders, [v * u for v in q]) for q in path]
                r = summarize(s, b1); r.update(path_units=path, constraint_dropped=flags); cr['O1'] = r
                o2a = C.o2a_best(Cm, s.lref, ref)
                b2a = [lattice_bank(orders, [v * u for v in q]) for q, _, _ in o2a]
                r = summarize(s, b2a); r['constraint_dropped'] = [f for _, _, f in o2a]; cr['O2a'] = r
                b2b = []
                for t in range(4):
                    Z = torch.cat(s.fit[:t + 1])
                    keep = C.ffs(Z, K, start=C.start_index(seed, s.tasks[t], len(Z)))
                    b2b.append(C._group([(int(k) // C.N_FIT, int(k) % C.N_FIT) for k in keep]))
                cr['O2b'] = summarize(s, b2b); pols[(cell, 'O2b')] = (None, b2b)
                h2 = [min(a, b) for a, b in zip(cr['O2a']['hist'], cr['O2b']['hist'])]
                cr['O2'] = {'hist': h2, 'H': sum(h2) / 3}
                # marginal coverage: B4 final state, probe vs EVAL one-unit decrements
                pol, banks = pols[(cell, 'B4')]
                q = pol.q; P = torch.cat(s.fb); dp, de = [], []
                base_e = float(s.ell(banks[-1]).mean())
                pm = [C.prefix_min(P, s.fit[p][pol.order[p][:q[p]]]) for p in range(4)]
                fq = lambda qv: torch.stack([pm[i][:, qv[i] - 1] for i in range(4)]).min(0).values.view(4, C.N_FB).mean(1)
                f0 = fq(q)
                for i in range(4):
                    if q[i] > u:
                        qq = list(q); qq[i] -= u
                        dp.append(float((fq(qq) / f0).mean()) - 1.0)
                        de.append(float(s.ell(lattice_bank(pol.order, qq)).mean()) - base_e)
                cr['marginal'] = {'d_probe': dp, 'd_eval': de, 'spearman': spearman(dp, de)}
            log('oracles', n, seed)
    # ---- Gate 1 mechanics on registered check cell
    res['gate1'] = gate1(streams['a'], sched, res, pols, au, st0)
    res['storage_after'] = storage(); res['storage_before'] = st0
    res['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    res['seconds'] = round(time.time() - tstart, 1)
    res['decision'] = decide(res)
    wjson(RUN / 'RESULT.json', res)
    log('DECISION', json.dumps(res['decision'], indent=0))


def spearman(a, b):
    if len(a) < 2:
        return None
    rk = lambda x: torch.tensor(x).argsort().argsort().double()
    ra, rb = rk(a), rk(b); ra -= ra.mean(); rb -= rb.mean()
    den = float((ra.norm() * rb.norm()))
    return float((ra * rb).sum()) / den if den else None


def gate1(s, sched, res, pols, au, st0):
    g = {'custody': au['custody_pass'], 'roles_disjoint': not au['roles_problems'],
         'eval_never_passed_to_policy': 'structural: Stream.step inputs are FIT/FEEDBACK/meta only'}
    K, seed = 256, 2711; cell = f'a|{seed}|{K}'
    causal, determ = {}, {}
    gen = torch.Generator().manual_seed(0)
    for arm in DEPLOY:
        _, banks = pols[(cell, arm)]
        _, b2, _, _ = run_policy(s, arm, K, seed, sched)
        determ[arm] = all(cols(x) == cols(y) for x, y in zip(banks, b2))
        ok = True
        for t0 in (1, 2, 3):
            fit = [x if t < t0 else torch.randn(x.shape, generator=gen, dtype=torch.float64) for t, x in enumerate(s.fit)]
            fb = [x if t < t0 else torch.randn(x.shape, generator=gen, dtype=torch.float64) for t, x in enumerate(s.fb)]
            _, b3, _, _ = run_policy(s, arm, K, seed, sched, fit=fit, fb=fb)
            ok &= all(cols(banks[t]) == cols(b3[t]) for t in range(t0))
        causal[arm] = ok
    g['causal_future_invariance'] = causal; g['determinism'] = determ
    resur, budget, finite = {}, {}, True
    for (c, arm), (pol, banks) in pols.items():
        if arm == 'O2b':
            continue
        cr = res['cells'][c]; Kc = int(c.split('|')[2])
        resur.setdefault(arm, True)
        for t in range(1, 4):
            new = set(range(t * C.N_FIT, (t + 1) * C.N_FIT))
            resur[arm] &= cols(banks[t]) <= (cols(banks[t - 1]) | new)
        bk = Kc * (C.KEY_B + C.ID_B) + C.W_B + 128
        for t, b in enumerate(cr[arm]['bytes']):
            lim = {'B4': bk + C.N_FB * (t + 1) * C.PROBE_B, 'B6': cr['B4']['bytes'][t]}.get(arm, bk)
            budget.setdefault(arm, True); budget[arm] &= b <= lim
        e = torch.tensor(cr[arm]['ell'], dtype=torch.float64); e = e[~e.isnan()]
        finite &= bool(torch.isfinite(e).all()) and float(e.min()) >= 1 - 1e-9
    for c, cr in res['cells'].items():
        q1 = cr['O1']['path_units']
        resur.setdefault('O1', True)
        resur['O1'] &= all(q1[t][i] <= q1[t - 1][i] for t in range(1, 4) for i in range(t))
    g['no_resurrection'] = resur; g['bytes_within_budget'] = budget; g['finite_ell_ge_1'] = finite
    st1 = storage()
    g['storage'] = {'D_free_min_ok': min(st0['D_free'], st1['D_free']) >= 30 * GiB,
                    'vhdx_growth_bytes': st1['VHDX'] - st0['VHDX']}
    g['pass'] = (g['custody'] and g['roles_disjoint'] and all(causal.values()) and all(determ.values())
                 and all(resur.values()) and all(budget.values()) and finite and g['storage']['D_free_min_ok']
                 and g['storage']['vhdx_growth_bytes'] <= 0)
    return g


def decide(res):
    out = {}
    cells = res['cells']
    mH = lambda arm, K: sum(cells[c][arm]['H'] for c in cells if c.endswith(f'|{K}')) / 9
    rel = lambda x, y, K: 1 - mH(x, K) / mH(y, K)
    for K in KS:
        sk, sf = res['scb'][K]['K'], res['scb'][K]['fb']
        cK = [c for c in cells if c.endswith(f'|{K}')]
        acq = lambda arm: sum(cells[c][arm]['A'] for c in cK) / 9
        h = {}
        for arm in ('B4', sf):
            acc = {}
            for c in cK:
                tasks = [res_tasks(c)[p] for p in range(4)]
                for t in range(1, 4):
                    for i in range(t):
                        acc.setdefault(tasks[i], []).append(cells[c][arm]['ell'][t][i])
            h[arm] = {k: sum(v) / len(v) for k, v in acc.items()}
        worst = max(h['B4'][k] / h[sf][k] for k in h['B4'])
        out[K] = {'SCB_K': sk, 'SCB_fb': sf, 'mean_H': {a: mH(a, K) for a in DEPLOY + ('O1', 'O2a', 'O2b', 'O2')},
                  'rel_O1_vs_SCB_K': rel('O1', sk, K), 'rel_O2_vs_SCB_K': rel('O2', sk, K),
                  'rel_O2a_vs_SCB_K': rel('O2a', sk, K), 'rel_O2b_vs_SCB_K': rel('O2b', sk, K),
                  'rel_B4_vs_SCB_fb': rel('B4', sf, K), 'rel_B4_vs_B5': rel('B4', 'B5', K),
                  'B4_cells_better': sum(cells[c]['B4']['H'] < cells[c][sf]['H'] for c in cK),
                  'acq_ratio_B4_vs_SCB_fb': acq('B4') / acq(sf), 'worst_task_ratio_B4_vs_SCB_fb': worst,
                  'excess_SCB_K': mH(sk, K) - 1, 'excess_O2b': mH('O2b', K) - 1}
    g1 = res['gate1']['pass']
    head = all(out[K]['rel_O1_vs_SCB_K'] >= 0.05 for K in KS)
    if not head:
        v = ('NL01D_IRREVERSIBILITY_DOMINANT' if all(out[K]['rel_O2_vs_SCB_K'] >= 0.05 for K in KS)
             else 'NL01D_NO_ACTIONABLE_HEADROOM')
    else:
        g3ok = all(out[K]['rel_B4_vs_SCB_fb'] >= 0.05 and out[K]['B4_cells_better'] >= 8
                   and out[K]['acq_ratio_B4_vs_SCB_fb'] <= C.ACQ_TOL and out[K]['worst_task_ratio_B4_vs_SCB_fb'] <= C.WORST_TOL
                   for K in KS)
        v = 'NL01D_CAUSAL_ALLOCATION_PROMISING' if g3ok else 'NL01D_NOT_SUPPORTED'
    out['feedback_necessity_claim_allowed'] = {K: out[K]['rel_B4_vs_B5'] >= 0.01 for K in KS}
    out['gate1_pass'] = g1
    out['verdict'] = v if g1 else f'NL01D_PROTOCOL_BLOCKED (scientific branch would be {v}; see gate1)'
    return out


_TASKS = {}


def res_tasks(cell):
    return _TASKS[cell.split('|')[0]]


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    ph = sys.argv[1]
    if ph == 'run':
        au = json.loads((RUN / 'AUDIT.json').read_text())
        for n, perm in PERMS.items():
            _TASKS[n] = [au['selected'][i] for i in perm]
    {'audit': audit, 'lock': lock, 'run': run}[ph]()
