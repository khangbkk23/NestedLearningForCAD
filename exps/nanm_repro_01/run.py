"""NANM-REPRO-01: reproduce BENCH-G3 / NANM-G3 with the UNCHANGED oracle (agents/tasks/task_nanm_repro_01.md).

The oracle (exps/bench_g3/run.py, exps/bench_g3/metrics_split.py, exps/vbench_00/run.py) is imported and called
as-is. Only module-level path constants are rebound to run-owned locations (`bind`):
    bench_g3:  OUT -> results/hope_cad/nanm_repro_01/bench_g3 (E:, small JSON)
               CACHE/FEAT/NORM/SCORE -> $NANM_REPRO_CACHE/bench_g3 (K:, token caches, states, scores)
    vbench_00: OUT -> results/hope_cad/nanm_repro_01/vbench_00, FEAT -> $NANM_REPRO_CACHE/vbench_00/features
The historical directories results/hope_cad/{bench_g3,vbench_00} are read-only references (Tier-H).
Each phase runs in its own process with the same torch flags as the oracle's CLI.

    python -B -m exps.nanm_repro_01.run <phase>
phases: gate1 custody extract_mpdd gate2 extract_visa vlock normal_mpdd normal_visa relock extract_test score gate4
        metrics_mpdd metrics_visa compare
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from exps.bench_g3 import run as G
from exps.nanm_r0 import audit as A
from exps.nanm_r2b import run as B
from exps.nar_pilot_00 import core as K
from exps.vbench_00 import run as VB
from nanm import g3

ROOT = A.ROOT
RUN = ROOT / 'results/hope_cad/nanm_repro_01'
OUT, VOUT = RUN / 'bench_g3', RUN / 'vbench_00'
HIST, HISTV = ROOT / 'results/hope_cad/bench_g3', ROOT / 'results/hope_cad/vbench_00'
CACHE = Path(os.environ.get('NANM_REPRO_CACHE', '/mnt/k/NLCAD/cache/nanm_repro_01'))
CMP = RUN / 'COMPARE.json'
GiB = 2 ** 30
D_RESERVE, K_RESERVE = 30 * GiB, 10 * GiB
TOKEN_BYTES = 784 * 768 * 4                       # one fp32 image token grid as cached by the oracle
D_MOUNT = Path('/mnt/d')


# ------------------------------------------------------------------ storage policy
def _drvfs_drive(p: Path):
    """Windows drive letter of the DrvFS mount holding p, or None (e.g. WSL root ext4)."""
    p = p.resolve()
    best = None
    for line in Path('/proc/mounts').read_text().splitlines():
        src, mnt, typ = line.split()[:3]
        if typ in ('9p', 'drvfs', 'v9fs') and len(src) >= 2 and src[1] == ':' and (
                str(p) == mnt or str(p).startswith(mnt.rstrip('/') + '/')):
            if best is None or len(mnt) > len(best[1]):
                best = (src[0].upper(), mnt)
    return best[0] if best else None


def preflight(target: Path, need: int, what: str):
    """Refuse (STORAGE_BLOCKED) unless target is on a non-D DrvFS volume with need + reserve free, and D keeps 30 GiB."""
    target.mkdir(parents=True, exist_ok=True)
    drive = _drvfs_drive(target)
    d_free = shutil.disk_usage(D_MOUNT).free
    t_free = shutil.disk_usage(target).free
    rec = {'what': what, 'target': str(target), 'drive': drive, 'need_bytes': need, 'target_free': t_free,
           'd_free': d_free, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    ok = drive is not None and drive != 'D' and t_free - need >= K_RESERVE and d_free >= D_RESERVE
    rec['ok'] = ok
    log = RUN / 'logs/preflight.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write(json.dumps(rec) + '\n')
    K.require(ok, f'STORAGE_BLOCKED: {json.dumps(rec)}')
    A.log('PREFLIGHT ok', what, drive, f'need {need / GiB:.2f} GiB', f'free {t_free / GiB:.1f} GiB',
          f'D free {d_free / GiB:.1f} GiB')


def bind():
    K.require(_drvfs_drive(CACHE) not in (None, 'D'), f'cache root {CACHE} is not on a non-D DrvFS volume')
    G.OUT, G.CACHE = OUT, CACHE / 'bench_g3'
    G.FEAT, G.NORM, G.SCORE = G.CACHE / 'features', G.CACHE / 'normal', G.CACHE / 'scores'
    VB.OUT, VB.FEAT = VOUT, CACHE / 'vbench_00/features'
    for p in (OUT, VOUT):
        p.mkdir(parents=True, exist_ok=True)


def cli_flags():
    torch.backends.cuda.matmul.allow_tf32 = False      # identical to `python -m exps.bench_g3.run <phase>`


def merge_cmp(section, obj):
    cur = A.rj(CMP) if CMP.exists() else {}
    cur[section] = obj
    A.wj(CMP, cur)


def eq_table(new: dict, old: dict):
    keys = sorted(set(new) | set(old))
    rows = {k: {'new': new.get(k), 'hist': old.get(k), 'equal': new.get(k) == old.get(k)} for k in keys}
    return {'n': len(keys), 'n_equal': sum(r['equal'] for r in rows.values()), 'rows': rows}


# ------------------------------------------------------------------ gate 1 (CPU)
def gate1():
    hl, hv = A.rj(HIST / 'LOCK.json'), A.rj(HISTV / 'LOCK.json')
    src = {**hl['sources'], **hv['sources']}
    now = {p: A.sha(ROOT / p) for p in src}
    b14 = G.B14()
    ck = A.sha(b14.ENC_PATH)
    oracle = ('exps/nanm_r0/audit.py', 'exps/nanm_r1/run.py', 'exps/nanm_r2/run.py', 'exps/nanm_r2b/run.py',
              'exps/bench_g3/run.py', 'exps/bench_g3/metrics_split.py', 'exps/vbench_00/run.py',
              'nanm/g3.py', 'nanm/checkpoint.py', 'exps/nanm_repro_01/run.py')
    rec = {'sources_vs_hist_locks': eq_table(now, src), 'checkpoint': {'path': str(b14.ENC_PATH), 'sha256': ck,
                                                                     'locked': b14.ENC_SHA, 'equal': ck == b14.ENC_SHA},
           'source_sha256': {p: A.sha(ROOT / p) for p in oracle},
           'env': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else None, 'hist_lock_torch': [hl['torch'], hv['torch']]},
           'data': {ds: {c: (G.data_root(ds) / c).is_dir() for c in G.order(ds)} for ds in G.DSETS}}
    try:
        rec['env']['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                              capture_output=True, text=True).stdout.strip()
    except OSError:
        rec['env']['driver'] = None
    rec['pass'] = (rec['sources_vs_hist_locks']['n'] == rec['sources_vs_hist_locks']['n_equal']
                   and rec['checkpoint']['equal'] and all(all(v.values()) for v in rec['data'].values()))
    merge_cmp('gate1', rec)
    K.require(rec['pass'], 'gate1 failed')
    A.log('GATE1 pass', rec['env'])


# ------------------------------------------------------------------ gate 3a: custody + extraction
def custody():
    cli_flags()
    G.custody()
    new, old = A.rj(OUT / 'MPDD_MANIFEST.json')['files'], A.rj(HIST / 'MPDD_MANIFEST.json')['files']
    t = eq_table(new, old)
    merge_cmp('mpdd_custody', {k: v for k, v in t.items() if k != 'rows'} | {'all_equal': t['n'] == t['n_equal']})
    K.require(t['n'] == t['n_equal'], 'MPDD dataset differs from the historical custody manifest')


def extract_mpdd():
    cli_flags()
    need = sum(len(G.train_rows('mpdd', c)) for c in G.order('mpdd')) * TOKEN_BYTES
    preflight(G.FEAT, need, 'mpdd train tokens')
    G.extract_train()
    new = {p.name: G._feat_meta(p)['sha256'] for p in sorted(G.FEAT.glob('mpdd_*_train.pt'))}
    merge_cmp('tierH_mpdd_train_tokens', eq_table(new, A.rj(HIST / 'PREP_SEAL.json')['mpdd_train_features']))


def extract_visa():
    need = sum(len(VB.train_rows(c)) for c in VB.ORDER) * TOKEN_BYTES
    preflight(VB.FEAT, need, 'visa train tokens')
    VB.extract()                                       # calls R.configure() itself, as the historical CLI did


def vlock():
    """Re-issue the VisA train-cache lock consumed by G.load_train (paths on K:, hashes of this run)."""
    K.require(not (VOUT / 'LOCK.json').exists(), 'vlock exists')
    inputs = {}
    for c in VB.ORDER:
        p = VB.FEAT / f'{c}_train.pt'
        m = A.rj(p.with_suffix('.json'))
        K.require(A.sha(p) == m['sha256'] and m['encoder_sha256'] == G.B14().ENC_SHA, f'train cache {c}')
        inputs[c] = {'path': str(p), 'sha256': m['sha256'], 'n': m['n']}
    hist = A.rj(HISTV / 'LOCK.json')
    A.wj(VOUT / 'LOCK.json', {'inputs': inputs, 'repro_of': str(HISTV / 'LOCK.json'),
                              'note': 'NANM-REPRO-01 re-issued input lock; only `inputs` is read by bench_g3'})
    merge_cmp('tierH_visa_train_tokens', eq_table({c: v['sha256'] for c, v in inputs.items()},
                                                  {c: v['sha256'] for c, v in hist['inputs'].items()}))


# ------------------------------------------------------------------ gate 2: same-device parity, MPDD 2-task prefix
@torch.no_grad()
def gate2():
    cli_flags()
    sz, ords = G.sizes_for('mpdd'), G.order('mpdd')[:2]
    res = {}
    for seed in G.SEEDS:
        so, sn, ok = [], [], {}
        for t, c in enumerate(ords):
            fit = G.fit_tokens('mpdd', c)
            need = max(sz['g3'][t], sz['a0b'][t])                       # oracle _normal_ds for MPDD
            kc = A.greedy_coreset(fit, need, seed=seed)
            S1, n1 = B.residual_stats(fit.float(), kc[:sz['g3'][t]].cuda())
            keys = fit[kc.cuda()].cpu()
            st = g3.fit_task(fit, sz['g3'][t], seed, n_select=need)
            ok[c] = {'kc': torch.equal(st.kc, kc), 'keys': torch.equal(st.keys, keys[:sz['g3'][t]]),
                     'S': torch.equal(st.S, S1), 'n': st.n == n1}
            so.append({'keys': keys, 'S1': S1, 'n1': n1}); sn.append(st)
            del fit; torch.cuda.empty_cache()
        q = {}
        for c in ords:                                                     # EVAL-role train/good tokens (no test)
            data = G.load_train('mpdd', c)
            idx = torch.as_tensor(G.roles('mpdd', data['rows'])['EVAL'])
            q[c] = A.unit(data['features'][idx].reshape(-1, G.D).cuda()).half().cpu()
        for l in range(2):
            W3, c3 = B.whiten(sum(x['S1'] for x in so[:l + 1]), sum(x['n1'] for x in so[:l + 1]))
            Wn, cn = g3.stage_whitener(sn[:l + 1])
            ko = torch.cat([x['keys'][:sz['g3'][j]] for j, x in enumerate(so[:l + 1])]).cuda()
            kn = g3.union_keys(sn[:l + 1]).cuda()
            ok[f'stage{l}'] = {'W': torch.equal(Wn, W3.float()), 'cond': cn == c3}
            for c in ords[:l + 1]:
                a = G.whitened(q[c], ko.float(), W3.float().cuda())
                b = g3.score(q[c], kn, Wn.cuda())
                ok[f'stage{l}'][f'score_{c}'] = {'equal': torch.equal(a, b), 'max_abs': float((a - b).abs().max()),
                                                'n': len(a)}
        res[seed] = ok
        A.log('GATE2', seed, json.dumps(ok))
    passed = all(v is True or (isinstance(v, dict) and all(
        (w is True) or (isinstance(w, dict) and w.get('equal', True)) for w in v.values()))
                 for r in res.values() for v in r.values())
    merge_cmp('gate2_tierA_mpdd_prefix', {'device': torch.cuda.get_device_name(0), 'tasks': list(ords),
                                          'results': res, 'pass': passed})
    K.require(passed, 'NANM_PARITY_FAIL at gate 2')


# ------------------------------------------------------------------ gate 3b: normal, relock, test, score
def normal(ds):
    cli_flags()
    preflight(G.NORM, (2 if ds == 'mpdd' else 12) * 3 * 40 * 2 ** 20, f'normal states {ds}')
    G.normal(ds)
    new = A.rj(OUT / f'normal_{ds}.json')
    old = A.rj(HIST / f'normal_{ds}.json')
    files_new = {f: h for s in new['seeds'].values() for f, h in s['files'].items()}
    files_old = {f: h for s in old['seeds'].values() for f, h in s['files'].items()}
    merge_cmp(f'tierH_normal_{ds}', {'files': eq_table(files_new, files_old), 'sizes_equal': new['sizes'] == old['sizes'],
                                     'bytes_equal': new['bytes'] == old['bytes'], 'seconds': new['seconds']})


def relock():
    """Run-owned LOCK/PREP_SEAL in the oracle's schema so G.verify_lock() works unchanged."""
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    K.require('Status: LOCKED' in G.TASK.read_text(), 'task file not LOCKED')
    K.require(not any(G.FEAT.glob('*_test.pt')), 'test tokens exist before lock')
    hl = A.rj(HIST / 'LOCK.json')
    files = {n: A.sha(OUT / n) for n in ('MPDD_MANIFEST.json', 'NORMAL.json')}
    feats = {p.name: G._feat_meta(p)['sha256'] for p in sorted(G.FEAT.glob('mpdd_*_train.pt'))}
    for n, h in feats.items():
        K.require(A.sha(G.FEAT / n) == h, 'feature cache changed ' + n)
    A.wj(OUT / 'PREP_SEAL.json', {'files': files, 'mpdd_train_features': feats,
                                 'task_sha256_at_prep_seal': A.sha(G.TASK),
                                 'repro_note': 'PARITY.json and COST.json not produced: VisA-DEV parity inputs '
                                               '(cmsb devr caches) no longer exist; cost is not a reproduction gate'})
    tm = {ds: G.test_manifest(ds) for ds in G.DSETS}
    for ds in G.DSETS:
        K.require(G.K_text(tm[ds]) == hl['test_manifest_sha256'][ds], f'test manifest differs from history: {ds}')
    sources = {s: A.sha(ROOT / s) for s in G.SOURCES}
    K.require(sources == hl['sources'], 'oracle sources differ from the historical LOCK')
    lock = dict(hl)
    lock.update({'sources': sources, 'prep_seal_sha256': A.sha(OUT / 'PREP_SEAL.json'), 'torch': torch.__version__,
                 'repro_of_lock_sha256': A.sha(HIST / 'LOCK.json'),
                 'repro_note': 'NANM-REPRO-01 run lock; protocol fields copied from the historical LOCK.json'})
    A.wj(OUT / 'LOCK.json', lock)
    A.log('RELOCK', A.sha(OUT / 'LOCK.json'))


def extract_test():
    cli_flags()
    need = sum(len(G.test_rows(ds, c)) for ds in G.DSETS for c in G.order(ds)) * TOKEN_BYTES
    preflight(G.FEAT, need, 'test tokens')
    G.extract_test()
    merge_cmp('tierH_test_tokens', eq_table(A.rj(OUT / 'EXTRACT_COMPLETE.json')['files'],
                                            A.rj(HIST / 'EXTRACT_COMPLETE.json')['files']))


def score():
    cli_flags()
    preflight(G.SCORE, 2 * GiB, 'score grids')
    G.score()
    new = A.rj(OUT / 'SCORES_COMPLETE.json')
    merge_cmp('tierH_scores', eq_table(new['files'], A.rj(HIST / 'SCORES_COMPLETE.json')['files']) |
              {'seconds': new['seconds']})


# ------------------------------------------------------------------ gate 4: nanm.g3 full streams vs oracle outputs
@torch.no_grad()
def gate4():
    cli_flags()
    rec, ex, sc = A.rj(OUT / 'NORMAL.json'), A.rj(OUT / 'EXTRACT_COMPLETE.json'), A.rj(OUT / 'SCORES_COMPLETE.json')
    res, t0 = {}, time.monotonic()
    for ds in G.DSETS:
        sz, ords = rec[ds]['sizes'], G.order(ds)
        Tn = len(ords)
        Q = {c: G.load_test(ds, c, ex) for c in ords}
        S = {}
        for c in ords:
            p = G.SCORE / f'{ds}_{c}.pt'
            K.require(A.sha(p) == sc['files'][p.name], 'score file changed')
            S[c] = torch.load(p, weights_only=True)['grids']
        for seed in G.SEEDS:
            store = [G.load_norm(ds, seed, m, rec) for m in ords]
            Wst = G.load_stage(ds, seed, rec)
            states, ok = [], {'state': {}, 'W': {}, 'score': {}}
            for t, c in enumerate(ords):
                fit = G.fit_tokens(ds, c)
                need = [sz['g3'][t], sz['a0b'][t]] + ([sz['g3c'][t], sz['a0c30'][t]] if ds == 'visa' else [])
                if ds == 'visa' and seed == 2711:
                    need.append(sz['r2b_a0c'][t])
                st = g3.fit_task(fit, sz['g3'][t], seed, n_select=max(need))
                o = store[t]
                ok['state'][c] = (torch.equal(st.kc, o['kc']) and torch.equal(st.keys, o['keys'][:sz['g3'][t]])
                                  and torch.equal(st.S, o['S1']) and st.n == o['n1'])
                states.append(st)
                del fit; torch.cuda.empty_cache()
            maxdiff = 0.0
            for l in range(Tn):
                W, _ = g3.stage_whitener(states[:l + 1])
                ok['W'][l] = torch.equal(W, Wst[l]['G3'])
                keys = g3.union_keys(states[:l + 1]).cuda()
                Wg = W.cuda()
                for j, c in enumerate(ords[:l + 1]):
                    a = g3.score(Q[c][0], keys, Wg)
                    b = S[c][f'G3|s{seed}'][l - j].reshape(-1)
                    eq = torch.equal(a, b)
                    maxdiff = max(maxdiff, float((a - b).abs().max()))
                    ok['score'][f'{l}|{c}'] = eq
            res[f'{ds}|s{seed}'] = {'states_equal': sum(ok['state'].values()), 'n_tasks': Tn,
                                    'W_equal': sum(ok['W'].values()), 'scores_equal': sum(ok['score'].values()),
                                    'n_score_cells': len(ok['score']), 'max_abs_diff': maxdiff}
            A.log('GATE4', ds, seed, res[f'{ds}|s{seed}'], round(time.monotonic() - t0))
            del store
    passed = all(v['states_equal'] == v['n_tasks'] and v['W_equal'] == v['n_tasks'] and
                 v['scores_equal'] == v['n_score_cells'] for v in res.values())
    merge_cmp('gate4_tierA_full_streams', {'results': res, 'pass': passed, 'seconds': time.monotonic() - t0})
    K.require(passed, 'NANM_PARITY_FAIL at gate 4')


# ------------------------------------------------------------------ gate 5: metrics (original evaluation code)
def metrics(ds):
    cli_flags()
    from exps.bench_g3 import metrics_split as MS
    MS.main(ds)


def compare():
    """Historical metric differences (MPDD only: VisA metrics were never produced historically)."""
    out = {}
    for ds in G.DSETS:
        p = OUT / f'RESULT_{ds}.json'
        if not p.exists():
            continue
        new = A.rj(p)
        hp = HIST / f'RESULT_{ds}.json'
        if hp.exists():
            old = A.rj(hp)
            mac = {k: {m: {'new': v[m], 'hist': old['macro'][k][m], 'diff': v[m] - old['macro'][k][m]}
                       for m in v if m in old['macro'].get(k, {})} for k, v in new['macro'].items()}
            out[ds] = {'macro': mac, 'max_abs_macro_diff': max(abs(x['diff']) for v in mac.values() for x in v.values()),
                       'decisions_new': new['decisions'][ds], 'decisions_hist': old['decisions'][ds]}
        else:
            out[ds] = {'macro': new['macro'], 'decisions_new': new['decisions'][ds], 'hist': None}
    merge_cmp('gate5_metrics', out)


PHASES = {'gate1': gate1, 'custody': custody, 'extract_mpdd': extract_mpdd, 'gate2': gate2,
          'extract_visa': extract_visa, 'vlock': vlock, 'normal_mpdd': lambda: normal('mpdd'),
          'normal_visa': lambda: normal('visa'), 'relock': relock, 'extract_test': extract_test, 'score': score,
          'gate4': gate4, 'metrics_mpdd': lambda: metrics('mpdd'), 'metrics_visa': lambda: metrics('visa'),
          'compare': compare}

if __name__ == '__main__':
    K.require(len(sys.argv) == 2 and sys.argv[1] in PHASES, f'phase in {list(PHASES)}')
    bind()
    t = time.monotonic()
    PHASES[sys.argv[1]]()
    A.log('PHASE DONE', sys.argv[1], round(time.monotonic() - t, 1), 's')
