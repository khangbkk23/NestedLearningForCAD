"""B325: independent replication of Khang's C / F / HOPE-S on MVTec (card: replication/experiments/B325/card.md).

Numerical code is Khang's own ``exps/`` package at the repository root, imported unmodified (MIT). The published
results were produced with Khang's code at commit 5217cb1; ``replication/khang_source.json`` pins the SHA-256 of every
repository file this driver can import, and ``khang_code_intact()`` refuses to run if any of them changed. This driver
only rebinds Khang's module paths (data root, encoder path, BASE directory, run directory) the same way his own
``bind()`` functions do, and replaces his lock / seal plumbing, which reads historical result files that are not in git.

Phases (each a subcommand; ``job`` runs them in order with one process per GPU):
    gate          CPU: encoder hash, BASE hash, MVTec train images byte-identical to Khang's manifests, disk.
    extract-train GPU: frozen DINOv2-reg B/14 train tokens (fp32), Khang's nar_b14_01.extract verbatim.
    make-base     GPU: model_new(seed) -> BASE.pt (one per seed; shared by every arm of that seed).
    train         GPU: one arm (C / F / H), 15 tasks, hope_s_01.acquire verbatim.
    fidelity      GPU: nar_lite_01.fidelity verbatim on normal train images (bf16 gate).
    extract-test  GPU: lite-encoder test tokens in the scoring dtype, after every arm of the job is sealed.
    score         GPU: nar_lite_01.expert_grids + prefix_grids verbatim.
    metrics       CPU: nar_lite_01.metrics('smooth') and ('raw') verbatim.
    label         CPU: card section 10 rules, computed from RESULT.json files only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPLICATION = Path(__file__).resolve().parents[1]          # replication/
REPO = REPLICATION.parent                                   # repository root: Khang's exps/, models/, dataset/
HOPE = REPO
CONFIG = REPLICATION / 'configs' / 'b325.json'
KHANG_SOURCE = REPLICATION / 'khang_source.json'
# Arm -> (gamma0, learn). C: plain patch loss; F: fixed hard-patch gamma 2.1 (post hoc at seed 2711, frozen);
# H: HOPE-S, gamma from 0 and lambda learned. All arms run hope_s_01.acquire, so CREDIT / gamma-direction
# gradients are computed in every arm and discarded unless learn=True (Khang's matched-compute design).
ARMS = {'C': (0.0, False), 'F': (2.1, False), 'H': (0.0, True)}
TOL_PORT = 0.005
TOL_GAMMA = 0.10


# ---------------------------------------------------------------- pure helpers (no torch)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:              # create-exclusive, as Khang's write_json
        json.dump(value, f, indent=1, sort_keys=True)


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def log(*args):
    print('B325', *args, flush=True)


def config():
    return read(CONFIG)


def text_sha(path):
    """SHA-256 of a text file with CRLF normalised to LF (a Windows checkout may rewrite line endings)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def khang_code_intact():
    """Every pinned file of Khang's code (the import closure of this driver, B326 and Khang's gate tests) still equals
    the bytes used for the published runs (commit 5217cb1). Returns that commit."""
    src = read(KHANG_SOURCE)
    bad = [p for p, h in src['files'].items() if not (HOPE / p).is_file() or text_sha(HOPE / p) != h]
    require(not bad, f"Khang's code differs from the pinned {src['commit'][:7]} files: {bad[:5]}")
    return src['commit']


def common_dtype(fidelities):
    """Card s10.1: one precision for every arm of a seed; bf16 only if every arm passes."""
    require(len(fidelities) > 0, 'no fidelity record')
    return 'bfloat16' if all(f.get('passed') is True for f in fidelities) else 'float32'


def label_nl(deltas, lows):
    """Card s10, Q-NL over the two new seeds; deltas = H - F final Pixel AP, lows = CI lower bounds."""
    require(len(deltas) == 2 == len(lows), 'Q-NL needs exactly two complete seeds')
    mean = sum(deltas) / 2
    if all(d >= .010 for d in deltas) and all(lo > 0 for lo in lows):
        return 'NL_SIGNAL'
    if all(d <= -.010 for d in deltas):
        return 'NL_NEGATIVE'
    if abs(mean) < .005 and all(abs(d) < .010 for d in deltas):
        return 'NL_NULL_REPLICATED'
    return 'NL_MIXED'


def label_hp(deltas):
    """Card s10, Q-HP over the two new seeds; deltas = F - C final Pixel AP (no CI by design)."""
    require(len(deltas) == 2, 'Q-HP needs exactly two complete seeds')
    mean = sum(deltas) / 2
    if all(d >= .005 for d in deltas) and mean >= .010:
        return 'HARDPATCH_SUPPORTED'
    if all(d <= -.005 for d in deltas):
        return 'HARDPATCH_NEGATIVE'
    if all(d < .005 for d in deltas):
        return 'HARDPATCH_NOT_SUPPORTED'
    return 'HARDPATCH_MIXED'


def label_port(results, reference):
    """Card s5 + s10: AP/AUROC decide; gamma fingerprint only qualifies the label."""
    f, h = results['F'], results['H']
    ap_ok = (abs(f['final_pixel_ap'] - reference['F']['final_pixel_ap']) <= TOL_PORT
             and abs(f['final_image_auroc'] - reference['F']['final_image_auroc']) <= TOL_PORT
             and abs(h['final_pixel_ap'] - reference['H']['final_pixel_ap']) <= TOL_PORT)
    ref_g = reference['H']['gamma_end']
    gamma = {k: results['H_gamma_end'][k] for k in ref_g}
    gamma_ok = all(abs(gamma[k] - v) <= TOL_GAMMA * abs(v) for k, v in ref_g.items())
    label = 'PORT_DIVERGES' if not ap_ok else 'PORT_FAITHFUL' if gamma_ok else 'PORT_FAITHFUL_GAMMA_DRIFT'
    return {'label': label, 'ap_auroc_within_tolerance': ap_ok, 'gamma_within_tolerance': gamma_ok,
            'gamma_end': gamma, 'base_identity': 'sidecar sha256 only (decision of 2026-11-19, card s9)'}


def label_c_path(c, reference):
    """Card s11: our C (acquire, gamma 0) against Khang's C (fit_task) at seed 2711 with his BASE."""
    d_ap = c['final_pixel_ap'] - reference['final_pixel_ap']
    d_auc = c['final_image_auroc'] - reference['final_image_auroc']
    ok = abs(d_ap) <= TOL_PORT and abs(d_auc) <= TOL_PORT
    return {'label': 'C_PATH_FAITHFUL' if ok else 'C_PATH_DIVERGES', 'delta_pixel_ap': d_ap, 'delta_image_auroc': d_auc}


# ---------------------------------------------------------------- Khang's modules, rebound
def khang(data, weights, hist=None, run=None):
    """Import Khang's modules and rebind their path globals (same mechanism as his bind())."""
    if str(HOPE) not in sys.path:
        sys.path.insert(0, str(HOPE))
    import torch
    from exps.dinomaly import encoder as E
    from exps.hope_s_01 import run as H
    from exps.nar_b14_01 import run as B
    from exps.nar_lite_01 import run as L
    from exps.nar_patch_01 import run as P
    from exps.nar_pilot_00 import core as K, full as R
    from exps.nar_pilot_00.detector import prefix_grids
    from exps.nar_warm_01 import run as W
    R.DATA = Path(data)
    B.ENC_PATH = Path(weights)
    if hist is not None:
        W.HIST = Path(hist)                       # directory holding BASE.pt
    if run is not None:
        L.OUT = L.REF = Path(run)                 # lite detector reads REF/stage15.pt; metrics read OUT
    return SimpleNamespace(torch=torch, E=E, H=H, B=B, L=L, P=P, K=K, R=R, W=W, prefix_grids=prefix_grids)


class Deadline:
    """Budget stand-in for Khang's helpers: they only call .check()."""
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def check(self, *args, **kwargs):
        require(time.monotonic() < self.end, 'phase deadline reached')


def environment(m):
    t = m.torch
    return {'torch': t.__version__, 'cuda_build': t.version.cuda, 'python': sys.version.split()[0],
            'device': t.cuda.get_device_name(0) if t.cuda.is_available() else None,
            'khang_code_commit': khang_code_intact()}


# ---------------------------------------------------------------- phases
def gate(a):
    cfg = config()
    commit = khang_code_intact()
    m = khang(a.data, a.weights)
    require(sha(a.weights) == m.B.ENC_SHA == cfg['encoder_sha256'], 'encoder hash')
    out = {'encoder_sha256': m.B.ENC_SHA, 'khang_code_commit': commit}
    if a.base:
        require(sha(a.base) == cfg['khang_base_sha256'], 'Khang BASE.pt hash')
        out['khang_base_sha256'] = cfg['khang_base_sha256']
    images = 0
    for c in m.R.ORDER:
        man = Path(a.manifests) / f'{c}_train.json'
        require(sha(man) == cfg['khang_train_manifests'][c], f'manifest hash {c}')
        ref = read(man)
        rows = m.R.rows_for(c, 'train')
        paths = [r['relative_path'] for r in rows]
        require(len(paths) == ref['n'] == len(ref['source_images']) and set(paths) == set(ref['source_images']),
                f'train identities differ from Khang: {c}')
        for p in paths:
            require(sha(m.R.DATA / p) == ref['source_images'][p], f'image bytes differ from Khang: {p}')
        images += len(paths)
    out['train_images_identical'] = images
    Path(a.cache).mkdir(parents=True, exist_ok=True)   # disk_usage fails on a missing directory
    free = shutil.disk_usage(a.cache).free
    require(free > cfg['disk_free_min_bytes'], f'disk headroom {free}')
    out['disk_free_bytes'] = free
    write_json(Path(a.out) / 'GATE.json', out)
    log('GATE_PASS', images, 'images')


def extract_train(a):
    m = khang(a.data, a.weights)
    m.R.configure()
    require(sha(m.B.ENC_PATH) == m.B.ENC_SHA, 'encoder hash')
    m.B.FEAT = Path(a.feat)
    enc = m.E.load(m.B.ENC_PATH, 'base', 'cuda')
    compare = {}
    for c in m.R.ORDER:
        rec = m.B.extract(c, 'train', enc)
        if a.manifests:                            # informational only: cross-GPU bitwise equality not expected
            compare[c] = rec['sha256'] == read(Path(a.manifests) / f'{c}_train.json')['sha256']
    write_json(Path(a.feat) / 'EXTRACT_TRAIN.json', {'feature_sha256_equals_khang': compare, **environment(m)})
    log('EXTRACT_TRAIN_DONE', sum(compare.values()), 'of', len(compare), 'feature files bitwise equal to Khang')


def make_base(a):
    m = khang(a.data, a.weights)
    m.R.configure()
    m.R.SEED = a.seed
    _, _, base = m.R.model_new(None)
    target = Path(a.out) / 'BASE.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    m.R.save(target, base)
    rec = {'seed': a.seed, 'sha256': sha(target), **environment(m)}
    if a.compare:
        ref = m.torch.load(a.compare, weights_only=True)
        require(ref.keys() == base.keys(), 'BASE keys differ')
        diff = max(float((ref[k].float() - base[k].float()).abs().max()) for k in ref)
        rec.update(compare_sha256=sha(a.compare), bitwise_equal=sha(a.compare) == rec['sha256'], max_abs_diff=diff)
    write_json(Path(a.out) / 'BASE.json', rec)
    log('BASE', a.seed, rec['sha256'], rec.get('bitwise_equal'))


def train(a):
    gamma0, learn = ARMS[a.arm]
    m = khang(a.data, a.weights, hist=a.hist)
    t, K, R, W, H = m.torch, m.K, m.R, m.W, m.H
    R.configure()
    R.SEED = a.seed
    R.loss = m.P.patch_loss                        # normal_eval logging as in hope_s_01/02 execute()
    out = Path(a.out)
    require(not out.exists(), f'run directory exists, preserve: {out}')
    base_path = Path(a.hist) / 'BASE.pt'
    require(sha(base_path) == a.base_sha, 'BASE hash differs from the one recorded for this seed')
    budget = Deadline(a.seconds)
    base = t.load(base_path, weights_only=True)
    model, loras, _ = R.model_new(base)
    fixed = K.init_state(loras, a.seed)
    lam, gamma = t.zeros(len(loras), dtype=t.float64), gamma0
    archive, seconds = [], 0.
    for stage, c in enumerate(R.ORDER):
        path = Path(a.feat) / f'{c}_train.pt'
        require(sha(path) == read(path.with_suffix('.json'))['sha256'], f'train cache hash {c}')
        data = t.load(path, weights_only=True, mmap=True)
        roles = R.assert_cache(data, c)
        init = W.initial_state(fixed, archive)
        K.set_state(loras, init)
        before = R.normal_eval(model, data['features'], roles['EVAL'])
        lam_out, g_out, trace, wall = H.acquire(model, loras, data['features'], roles, stage, lam, gamma,
                                                learn=learn, budget=budget)
        if not learn:
            require(t.equal(lam_out, lam) and g_out == gamma, 'fixed policy changed')
        require(len(trace['fit']) == R.CFG.steps and len(trace['credit']) == len(trace['h_gamma'])
                == R.CFG.steps // R.CFG.m, 'matched fast / slow-compute budget')
        lam, gamma = lam_out, g_out
        endpoint = K.get_state(loras)
        after = R.normal_eval(model, data['features'], roles['EVAL'])
        archive.append(endpoint)
        seconds += wall
        write_json(out / f'stage{stage + 1:02}.json', {
            'stage': stage + 1, 'category': c, 'arm': a.arm, 'seed': a.seed, 'before': before, 'after': after,
            'seconds': wall, 'gamma_end': gamma, 'lam': lam.tolist(), 'trace': trace,
            'initial_digest': W.digest(init), 'endpoint_digest': W.digest(endpoint)})
        log('TRAIN', a.arm, a.seed, stage + 1, c, 'gamma', round(gamma, 4), 'tail', round(trace['credit'][-1], 4),
            's', round(wall, 1))
        del data
    final = out / 'stage15.pt'
    R.save(final, {'archive': archive, 'phi': {}, 'lam': lam, 'gamma': gamma, 'cfg': R.CFG.dict(),
                   'stage': 15, 'arm': a.arm, 'seed': a.seed, 'base_sha256': a.base_sha})
    write_json(out / 'TRAIN_COMPLETE.json', {
        'arm': a.arm, 'seed': a.seed, 'gamma0': gamma0, 'learn': learn, 'base_sha256': a.base_sha,
        'stage15_sha256': sha(final), 'records': {f'stage{i:02}.json': sha(out / f'stage{i:02}.json')
                                                  for i in range(1, 16)},
        'update_seconds': seconds, 'cuda_peak_bytes': t.cuda.max_memory_allocated(), **environment(m)})
    log('SEALED', a.arm, a.seed, round(seconds, 1), 's')


def verify_sealed(run):
    run = Path(run)
    tc = read(run / 'TRAIN_COMPLETE.json')
    require(sha(run / 'stage15.pt') == tc['stage15_sha256'], f'sealed checkpoint changed: {run}')
    for p, h in tc['records'].items():
        require(sha(run / p) == h, f'sealed record changed: {run / p}')
    return tc


def fidelity(a):
    verify_sealed(a.run)
    m = khang(a.data, a.weights, hist=a.hist, run=a.run)
    m.R.configure()
    fid = m.L.fidelity(m.torch.bfloat16, Deadline(a.seconds))
    write_json(Path(a.run) / 'FIDELITY.json', fid)
    log('FIDELITY', a.run, fid.get('relative_mean_abs_error'), fid['passed'])


def extract_test(a):
    for run in a.sealed:                           # test access only after every arm of the job is sealed
        verify_sealed(run)
    m = khang(a.data, a.weights)
    m.R.configure()
    t, R, L = m.torch, m.R, m.L
    dtype = getattr(t, a.dtype)
    enc = L.lite_encoder(dtype)
    out = Path(a.out)
    for c in R.ORDER:
        target = out / f'{c}_test.pt'
        require(not target.exists(), f'test cache exists, preserve: {target}')
        rows = R.rows_for(c, 'test')
        feats = t.empty(len(rows), 784, 768, dtype=dtype)
        with t.no_grad():
            for i in range(0, len(rows), 8):          # identical to nar_lite_01.score
                x = t.stack([m.E.preprocess(R.DATA / r['relative_path']) for r in rows[i:i + 8]]).cuda().to(dtype)
                feats[i:i + len(x)] = L.tokens(enc, x).cpu()
        require(bool(t.isfinite(feats).all()), 'feature finiteness')
        target.parent.mkdir(parents=True, exist_ok=True)
        R.save(target, {'rows': rows, 'features': feats})
        write_json(target.with_suffix('.json'), {'sha256': sha(target), 'n': len(rows), 'dtype': a.dtype})
        log('EXTRACT_TEST', a.dtype, c, len(rows))


def score(a):
    verify_sealed(a.run)
    m = khang(a.data, a.weights, hist=a.hist, run=a.run)
    m.R.configure()
    t, R, L = m.torch, m.R, m.L
    dtype = getattr(t, a.dtype)
    model, loras, archive, _ = L.lite_detector(dtype)
    files, start = {}, time.monotonic()
    for c in R.ORDER:
        path = Path(a.test) / f'{c}_test.pt'
        meta = read(path.with_suffix('.json'))
        require(meta['dtype'] == a.dtype and sha(path) == meta['sha256'], f'test cache {c}')
        data = t.load(path, weights_only=True, mmap=True)
        require(data['features'].dtype == dtype, 'test cache dtype')
        prefixes = m.prefix_grids(L.expert_grids(model, loras, archive, data['features'], dtype))
        target = Path(a.run) / f'scores_{c}.pt'
        R.save(target, {'prefix_grids': prefixes, 'rows': data['rows'], 'dtype': a.dtype})
        files[target.name] = sha(target)
        log('SCORE', a.run, c, len(data['rows']))
        del data, prefixes
    write_json(Path(a.run) / 'SCORES_COMPLETE.json', {'seconds': time.monotonic() - start, 'files': files,
                                                      'dtype': a.dtype})


def metrics(a):
    tc = verify_sealed(a.run)
    m = khang(a.data, a.weights, run=a.run)
    budget = Deadline(a.seconds)
    a1, raw = m.L.metrics('smooth', budget), m.L.metrics('raw', budget)
    run = Path(a.run)
    gamma = {f'stage{i:02}': read(run / f'stage{i:02}.json')['gamma_end'] for i in range(1, 16)}
    write_json(run / 'RESULT.json', {
        'arm': tc['arm'], 'seed': tc['seed'], 'a1': a1, 'raw': raw, 'gamma_end': gamma,
        'fidelity': read(run / 'FIDELITY.json'), 'dtype': read(run / 'SCORES_COMPLETE.json')['dtype'],
        'update_seconds': tc['update_seconds'], 'train_complete_sha256': sha(run / 'TRAIN_COMPLETE.json'),
        'exposure': 'historically exposed MVTec; exploratory only'})
    log('RESULT', tc['arm'], tc['seed'], round(a1['final_pixel_ap'], 4), round(a1['final_image_auroc'], 4))


def label(a):
    cfg = config()
    m = khang(a.data, a.weights)
    work, out = Path(a.work), {}
    job = cfg['jobs'][a.job]
    res = {}
    for s in job['seeds']:
        for arm in job['arms']:
            p = work / f's{s}' / arm / 'RESULT.json'
            res[s, arm] = read(p)['a1'] if p.exists() else None
    complete = [s for s in job['seeds'] if all(res[s, arm] is not None for arm in job['arms'])]
    out['complete_seeds'] = complete
    pairs = {}
    for s in complete:
        for name, (x, y) in {'H_minus_F': ('H', 'F'), 'F_minus_C': ('F', 'C')}.items():
            if x in job['arms'] and y in job['arms']:
                pairs[f'{s}:{name}'] = m.L.ci(res[s, x], res[s, y])
    out['deltas'] = pairs
    if a.job == 'C':                             # card s11: acquire(gamma=0) path vs Khang's fit_task C
        s = job['seeds'][0]
        out['c_path'] = label_c_path(res[s, 'C'], cfg['khang_reference']['C']) if s in complete else             {'label': 'INCOMPLETE'}
    elif a.job == 'Q':
        s = job['seeds'][0]
        if s in complete:
            gam = read(work / f's{s}' / 'H' / 'RESULT.json')['gamma_end']
            out['port'] = label_port({'F': res[s, 'F'], 'H': res[s, 'H'], 'H_gamma_end': gam}, cfg['khang_reference'])
        else:
            out['port'] = {'label': 'INCOMPLETE'}
    else:
        if complete == job['seeds']:               # card s10.2: no label unless every seed has the full triple
            nl = [pairs[f'{s}:H_minus_F'] for s in complete]
            out['Q_NL'] = label_nl([d['point'] for d in nl], [d['lo'] for d in nl])
            out['Q_HP'] = label_hp([pairs[f'{s}:F_minus_C']['point'] for s in complete])
        else:
            out['Q_NL'] = out['Q_HP'] = 'NO_LABEL_INCOMPLETE'
    write_json(work / 'LABELS.json', out)
    log('LABELS', json.dumps({k: v for k, v in out.items() if k != 'deltas'}))


# ---------------------------------------------------------------- orchestration
def run_on_gpus(commands, gpus):
    """One child per command, command i pinned to gpus[i]. With fewer GPUs than commands (e.g. a single local GPU),
    the commands run one after another, round robin over the given GPUs; results do not depend on the schedule."""
    if len(gpus) >= len(commands):
        run_parallel(commands, gpus=list(gpus[:len(commands)]))
    else:
        for i, cmd in enumerate(commands):
            run_parallel([cmd], gpus=[gpus[i % len(gpus)]])


def run_parallel(commands, gpus=None):
    """Start one child per command (optionally pinned to a GPU); fail if any child fails."""
    procs = []
    for i, cmd in enumerate(commands):
        env = os.environ.copy()
        if gpus is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpus[i])
        log('START', ' '.join(map(str, cmd)), 'gpu', None if gpus is None else gpus[i])
        procs.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__)), *map(str, cmd)], env=env))
    codes = [p.wait() for p in procs]
    require(all(c == 0 for c in codes), f'child failure {codes}')


def job(a):
    cfg = config()
    spec = cfg['jobs'][a.job]
    work, cache = Path(a.work), Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    io = ['--data', a.data, '--weights', a.weights]
    feat = cache / 'train_fp32'
    caps = spec['phase_seconds']                       # per-phase time limits (seconds)
    run_parallel([['gate', *io, '--manifests', a.khang, '--cache', cache, '--out', work,
                   *(['--base', Path(a.khang) / 'BASE.pt'] if spec['base'] == 'khang' else [])]])
    run_parallel([['extract-train', *io, '--feat', feat, '--manifests', a.khang]], gpus=[gpus_first(a, spec)])
    seeds = spec['seeds']
    gpus = [int(g) for g in a.gpus.split(',')] if getattr(a, 'gpus', None) else spec['gpus']
    hist = {}
    if spec['base'] == 'khang':
        s = seeds[0]
        hist[s] = Path(a.khang)
        run_parallel([['make-base', *io, '--seed', s, '--out', work / 'base_probe', '--compare',
                       Path(a.khang) / 'BASE.pt']], gpus=[gpus[0]])        # informational: R's generation path on T4
        base_sha = {s: cfg['khang_base_sha256']}
    else:
        run_on_gpus([['make-base', *io, '--seed', s, '--out', work / f's{s}'] for s in seeds], gpus)
        for s in seeds:
            hist[s] = work / f's{s}'
        base_sha = {s: read(work / f's{s}' / 'BASE.json')['sha256'] for s in seeds}
    lanes = spec['lanes']                              # one lane per GPU: list of (seed, arm) run in order
    runs = [(s, arm) for lane in lanes for s, arm in lane]
    rdir = lambda s, arm: work / f's{s}' / arm

    def by_lane(make):
        """Run each lane sequentially, lanes in parallel (one process per lane via a nested job-lane call)."""
        cmds = [['lane', '--steps', json.dumps([[str(x) for x in make(s, arm)] for s, arm in lane])]
                for lane in lanes]
        run_on_gpus(cmds, gpus)

    by_lane(lambda s, arm: ['train', *io, '--arm', arm, '--seed', s, '--hist', hist[s], '--base-sha', base_sha[s],
                            '--feat', feat, '--out', rdir(s, arm), '--seconds', caps['train_arm']])
    by_lane(lambda s, arm: ['fidelity', *io, '--hist', hist[s], '--run', rdir(s, arm),
                            '--seconds', caps['fidelity_arm']])
    dtype = {s: common_dtype([read(rdir(s2, arm) / 'FIDELITY.json') for s2, arm in runs if s2 == s]) for s in seeds}
    write_json(work / 'PRECISION.json', {str(s): d for s, d in dtype.items()})
    sealed = [x for s, arm in runs for x in ('--sealed', rdir(s, arm))]
    for d in sorted(set(dtype.values())):
        run_parallel([['extract-test', *io, '--dtype', d, '--out', cache / f'test_{d}', *sealed]], gpus=[gpus[0]])
    by_lane(lambda s, arm: ['score', *io, '--hist', hist[s], '--run', rdir(s, arm), '--dtype', dtype[s],
                            '--test', cache / f'test_{dtype[s]}'])
    for i in range(0, len(runs), spec['metric_workers']):
        run_parallel([['metrics', *io, '--run', rdir(s, arm), '--seconds', caps['metrics_arm']]
                      for s, arm in runs[i:i + spec['metric_workers']]])
    run_parallel([['label', *io, '--job', a.job, '--work', work]])


def gpus_first(a, spec):
    return int(a.gpus.split(',')[0]) if getattr(a, 'gpus', None) else spec['gpus'][0]


def lane(a):
    for step in json.loads(a.steps):
        run_parallel([step])                          # inherits this lane's CUDA_VISIBLE_DEVICES


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='phase', required=True)

    def cmd(name, fn, *extra):
        q = sub.add_parser(name)
        if name != 'lane':
            q.add_argument('--data', required=True)
            q.add_argument('--weights', required=True)
        for args, kw in extra:
            q.add_argument(*args, **kw)
        q.set_defaults(fn=fn)

    req = lambda *n, **kw: (n, {'required': True, **kw})
    opt = lambda *n, **kw: (n, kw)
    cmd('gate', gate, req('--manifests'), req('--cache'), req('--out'), opt('--base'))
    cmd('extract-train', extract_train, req('--feat'), opt('--manifests'))
    cmd('make-base', make_base, req('--seed', type=int), req('--out'), opt('--compare'))
    cmd('train', train, req('--arm', choices=tuple(ARMS)), req('--seed', type=int), req('--hist'),
        req('--base-sha'), req('--feat'), req('--out'), req('--seconds', type=float))
    cmd('fidelity', fidelity, req('--hist'), req('--run'), req('--seconds', type=float))
    cmd('extract-test', extract_test, req('--dtype', choices=('bfloat16', 'float32')), req('--out'),
        req('--sealed', action='append'))
    cmd('score', score, req('--hist'), req('--run'), req('--dtype', choices=('bfloat16', 'float32')), req('--test'))
    cmd('metrics', metrics, req('--run'), req('--seconds', type=float))
    cmd('label', label, req('--job', choices=('Q', 'R', 'C')), req('--work'))
    cmd('job', job, req('--job', choices=('Q', 'R', 'C')), req('--khang'), req('--work'), req('--cache'),
        opt('--gpus', help='comma-separated GPU ids overriding the config, e.g. 0 for a single GPU'))
    cmd('lane', lane, req('--steps'))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
