"""CMS-B gated step runner (S1-S4), agents/tasks/task_cms_bounded_00.md.

A step = candidate arm(s) vs a reference arm, 2 seeds, 12 VisA tasks (full 1cls train, alphabetical), dev sets only.
    python -B -m exps.cmsb.step lock S1
    python -B -m exps.cmsb.step execute S1 --owner-authorized
Reference 'F_last' = the F chain (warm single LoRA, gamma 2.1) scored with its LAST expert only (bounded, 1 level);
'F_archive' (min over all experts, unbounded) is reported as a ceiling, never gated against.
"""
from __future__ import annotations

import argparse
import gc
import json
import signal
import time
from dataclasses import replace

import numpy as np
import torch

from exps.cmsb import cms as C, dev as D, s0 as S0
from exps.hope_s_01 import run as H
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_pilot_00.detector import fm_eq8, prefix_grids
from exps.nar_warm_01 import run as W
from exps.tcs_00 import run as T
from exps.vbench_00 import run as V

ORDER = D.ORDER
SEEDS = (2711, 2712)
GATE = {'gain': .005, 'ci_lower': 0., 'same_sign_seeds': True, 'devr_drop': .010, 'fm_increase': .005,
        'adapter_bytes': 6 * 2**20, 'replay_bytes': 24 * 2**20, 'passes': 2}
REPLAY = 24 * 2**20             # owner 2026-11-18: q100 JPEG, "a bit more memory is fine"; < CADIC 10k-patch coreset
CAPS = {'wall_seconds': 12600, 'gpu_seconds': 10800, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 600_000_000}
SOURCES = ('exps/cmsb/step.py', 'exps/cmsb/cms.py', 'exps/cmsb/dev.py', 'exps/cmsb/test_cms.py',
           'exps/hope_s_01/run.py', 'exps/nar_lite_01/run.py', 'exps/nar_pilot_00/core.py',
           'exps/nar_pilot_00/full.py', 'agents/tasks/task_cms_bounded_00.md')

# Pre-declared steps. 'candidate' arms are gated against 'reference'; 'controls' are reported, never gated.
STEPS = {
    'S1': {'reference': 'F_last', 'candidate': {'CMS2': C.Spec()}, 'controls': {}},
    # S1 = NO_GAIN -> nothing stacked; S2 is gated against F_last. NL identification: CMS2R must also beat
    # SINGLE_R = one persistent level with the SAME replay reservoir, same current + replay gradient draws, same bytes.
    'S2': {'reference': 'F_last',
           'candidate': {'CMS2R': C.Spec(levels=(C.Level('fast', 1, True), C.Level('slow', 10, False, replay=True)),
                                         replay_bytes=REPLAY)},
           'controls': {'SINGLE_R': C.Spec(levels=(C.Level('single', 1, False, replay=True),), replay_bytes=REPLAY)},
           'nl_control': 'SINGLE_R'},
}


def out(step):
    return R.ROOT / f'results/hope_cad/cmsb_{step.lower()}'


def bind(step):
    W.OUT, W.CAPS, W.SOURCES = out(step), CAPS, SOURCES


def train_lock():
    return R.read(V.OUT / 'LOCK.json')


def loader(lk):
    def load(c):
        data = V.load_train(lk, c)
        return data['features'], T.roles_for(data['rows']), data['rows']
    return load


def encoder_fn():
    """JPEG bytes -> tokens with the TRAINING front end (full fp32 DINOv2-reg B/14, mean of blocks 6-9)."""
    from exps.dinomaly import encoder as E
    from exps.nar_b14_01 import run as B
    enc = E.load(B.ENC_PATH, 'base', 'cuda').eval()

    @torch.no_grad()
    def encode(b):
        return B.tokens(enc, C.decode(b)[None].cuda())[0]
    return encode


# ---------------------------------------------------------------- reference chain F (warm single LoRA)
def train_f(lk, seed, budget):
    """Identical to VBENCH-00 train() for another seed; returns the archive (one expert per task)."""
    original = R.SEED
    R.SEED = seed
    try:
        base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
        model, loras, _ = R.model_new(base)
        fixed = K.init_state(loras, seed)
        lam = torch.zeros(len(loras), dtype=torch.float64)
        archive = []
        for stage, c in enumerate(ORDER):
            data = V.load_train(lk, c)
            roles = T.roles_for(data['rows'])
            K.set_state(loras, W.initial_state(fixed, archive))
            H.acquire(model, loras, data['features'], roles, stage, lam, V.GAMMA, learn=False, budget=budget)
            archive.append(K.get_state(loras))
            R.log('TRAIN F', seed, stage + 1, c)
            del data
        del model, loras
        gc.collect(); torch.cuda.empty_cache()
        return archive
    finally:
        R.SEED = original


def f_archive(lk, seed, budget, step):
    if seed == V.SEED:
        K.require(R.sha(V.OUT / 'final.pt') == R.read(V.OUT / 'TRAIN_COMPLETE.json')['final_sha256'], 'VBENCH seal')
        return torch.load(V.OUT / 'final.pt', weights_only=True)['archive']
    s1 = out('S1') / f'F_s{seed}.pt'
    if step != 'S1' and (out('S1') / 'FINAL_SEAL.json').exists():
        K.require(R.sha(s1) == R.read(out('S1') / 'FINAL_SEAL.json')['files'][s1.name], 'S1 F seal')
        return torch.load(s1, weights_only=True)['archive']
    path = out(step) / f'F_s{seed}.pt'
    archive = train_f(lk, seed, budget)
    W.bounded_save(path, {'archive': archive, 'seed': seed}, sum(K.tensor_state_bytes(a) for a in archive), budget)
    return archive


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_f(archive, shas):
    """Stage-wise grids for F_last (expert t only) and F_archive (prefix min)."""
    dtype = D.DTYPE
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    model.eval().to(dtype)
    arch = [{n: (a.to(dtype), b.to(dtype)) for n, (a, b) in s.items()} for s in archive]
    g = {}
    for c in ORDER:
        for kind in ('devs', 'devr'):
            e = L.expert_grids(model, loras, arch, D.load(kind, c, shas)['features'], dtype)
            pre = prefix_grids(e)
            for t in range(len(ORDER)):
                g['F_last', kind, t, c] = e[t]
                g['F_archive', kind, t, c] = pre[t]
    del model, loras
    gc.collect(); torch.cuda.empty_cache()
    return g


@torch.no_grad()
def eval_cms(name, spec, stages, shas):
    ev = C.evaluator(spec)
    g = {}
    for c in ORDER:
        i = ORDER.index(c)
        fs = D.load('devs', c, shas)['features']
        fr = D.load('devr', c, shas)['features']
        for t in range(i, len(ORDER)):
            g[name, 'devs', t, c] = C.grids(spec, ev, stages[t], fs).min(0).values
        g[name, 'devr', len(ORDER) - 1, c] = C.grids(spec, ev, stages[-1], fr).min(0).values
    del ev
    gc.collect(); torch.cuda.empty_cache()
    return g


def cells_for(arm, g, shas):
    last = len(ORDER) - 1
    cs, cr = {}, {}
    for i, c in enumerate(ORDER):
        ds = D.load('devs', c, shas)
        for t in range(i, len(ORDER)):
            cs[t, c] = D.metrics_s(g[arm, 'devs', t, c], ds)['all']
        rows = D.load('devr', c, shas)['rows']
        cr[c] = D.cell(g[arm, 'devr', last, c], *D.geometry_r(rows))
    return {'S_stage': cs, 'R_final': cr,
            'S_final_macro': float(np.mean([cs[last, c]['pixel_ap'] for c in ORDER])),
            'S_acq_macro': float(np.mean([cs[i, c]['pixel_ap'] for i, c in enumerate(ORDER)])),
            'S_fm_eq8': fm_eq8(cs, ORDER, 'pixel_ap'),
            'S_final_auroc': float(np.mean([cs[last, c]['image_auroc'] for c in ORDER])),
            'R_final_macro': float(np.mean([cr[c]['pixel_ap'] for c in ORDER])),
            'R_final_auroc': float(np.mean([cr[c]['image_auroc'] for c in ORDER]))}


def gate(cand, ref, ledger):
    """cand/ref: {seed: cells}. Bootstrap over classes on the seed-mean per-class final Dev-S AP."""
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    last = len(ORDER) - 1
    pc = lambda arm: [np.mean([arm[s]['S_stage'][last, c]['pixel_ap'] for s in SEEDS]) for c in ORDER]
    ci = bootstrap_delta(pc(cand), pc(ref), seed=T.BOOT['seed'], B=T.BOOT['B'])
    per_seed = {s: cand[s]['S_final_macro'] - ref[s]['S_final_macro'] for s in SEEDS}
    devr = float(np.mean([cand[s]['R_final_macro'] - ref[s]['R_final_macro'] for s in SEEDS]))
    fm = float(np.mean([cand[s]['S_fm_eq8']['macro'] - ref[s]['S_fm_eq8']['macro'] for s in SEEDS]))
    guards = {'gain': ci['point'] >= GATE['gain'], 'ci_lower': ci['lo'] > GATE['ci_lower'],
              'same_sign_seeds': all(v > 0 for v in per_seed.values()),
              'devr_no_drop': devr >= -GATE['devr_drop'], 'fm_no_increase': fm <= GATE['fm_increase'],
              'memory': ledger['adapter_bytes_bf16'] <= GATE['adapter_bytes'] and ledger['replay_bytes'] <= GATE['replay_bytes']}
    return {'devs_final_delta': ci, 'per_seed_delta': per_seed, 'devr_final_delta': devr, 'devs_fm_delta': fm,
            'guards': guards, 'passed': all(guards.values())}


def recovery(arm, cells):
    """Descriptive: fraction of the F_last -> F_archive headroom recovered, per set, seed-mean."""
    out = {}
    for k in ('S_final_macro', 'R_final_macro'):
        a, lo, hi = (np.mean([cells[x][s][k] for s in SEEDS]) for x in (arm, 'F_last', 'F_archive'))
        out[k] = float((a - lo) / (hi - lo))
    return out


# ---------------------------------------------------------------- lock / execute
def lock(step):
    bind(step)
    o = out(step)
    K.require(not (o / 'LOCK.json').exists(), 'LOCK exists')
    s0 = R.read(S0.OUT / 'RESULT.json')
    K.require(s0['primary_set'] == 'Dev-S', 'S0 proxy check failed: Dev-R primary protocol (3 seeds) required')
    o.mkdir(parents=True, exist_ok=True)
    cfg = STEPS[step]
    R.write_json(o / 'LOCK.json', {
        'step': step, 'sources': {s: R.sha(R.ROOT / s) for s in SOURCES}, 'seeds': list(SEEDS), 'gate': GATE,
        'reference': cfg['reference'], 'candidate': {k: v.dict() for k, v in cfg['candidate'].items()},
        'controls': {k: v.dict() for k, v in cfg['controls'].items()}, 'caps': CAPS,
        'dev_cache_sha256': s0['dev_cache_sha256'], 's0_result_sha256': R.sha(S0.OUT / 'RESULT.json'),
        'train_lock_sha256': R.sha(V.OUT / 'LOCK.json'), 'torch': torch.__version__})
    R.log('LOCK', step, R.sha(o / 'LOCK.json'))


def execute(step):
    bind(step)
    o = out(step)
    K.require(not (o / 'STARTED.json').exists(), 'no implicit restart or resume')
    lk = R.read(o / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    shas = lk['dev_cache_sha256']
    budget = W.Budget()
    R.configure()
    R.write_json(o / 'STARTED.json', {'lock_sha256': R.sha(o / 'LOCK.json'), 'started_unix': time.time()})
    cfg = STEPS[step]
    tl = train_lock()
    try:
        cells, ledgers = {}, {}
        for seed in SEEDS:
            with budget.phase(f'F_s{seed}', gpu=True):
                g = eval_f(f_archive(tl, seed, budget, step), shas)
                for arm in ('F_last', 'F_archive'):
                    cells.setdefault(arm, {})[seed] = cells_for(arm, g, shas)
                del g
            for name, spec in {**cfg['candidate'], **cfg['controls']}.items():
                with budget.phase(f'{name}_s{seed}', gpu=True):
                    enc = encoder_fn() if spec.replay_bytes else None
                    stages, trace, ledger = C.train(spec, seed, ORDER, loader(tl), R.log, budget, enc)
                    del enc
                    K.require(ledger['adapter_bytes_bf16'] <= GATE['adapter_bytes'] and
                              ledger['replay_bytes'] <= GATE['replay_bytes'], 'memory ledger over budget')
                    W.bounded_save(o / f'{name}_s{seed}.pt', {'stages': stages, 'spec': spec.dict(), 'trace': trace},
                                   C.state_bytes(stages[0]) * len(stages), budget)
                    g = eval_cms(name, spec, stages, shas)
                    cells.setdefault(name, {})[seed] = cells_for(name, g, shas)
                    ledgers[name] = ledger
                    del g, stages
                R.log('ARM', name, seed, json.dumps({k: v for k, v in cells[name][seed].items()
                                                     if k.endswith(('macro', 'auroc'))}))
        ref = cells[cfg['reference']]
        gates = {name: gate(cells[name], ref, ledgers[name]) for name in cfg['candidate']}
        if cfg.get('nl_control'):
            for name in cfg['candidate']:
                gates[name]['nl_identification'] = gate(cells[name], cells[cfg['nl_control']], ledgers[name])
                gates[name]['passed_nl'] = gates[name]['passed'] and gates[name]['nl_identification']['passed']
        summary = {arm: {s: {k: v for k, v in cs.items() if k not in ('S_stage', 'R_final')} for s, cs in d.items()}
                   for arm, d in cells.items()}
        result = {'step': step, 'gates': gates, 'summary': summary, 'ledgers': ledgers,
                  'headroom_recovered': {n: recovery(n, cells) for n in list(cfg['candidate']) + list(cfg['controls'])},
                  'cells': {arm: {str(s): {'S_stage': {f'{t + 1:02}_{c}': v for (t, c), v in cs['S_stage'].items()},
                                           'R_final': cs['R_final']} for s, cs in d.items()} for arm, d in cells.items()},
                  'verdict': {n: 'GAIN' if g_['passed'] else 'NO_GAIN' for n, g_ in gates.items()},
                  'exposure': 'Dev-S (held-out train normals) and Dev-R (VisA-DEV) only; VisA-FINAL/MVTec untouched',
                  'resources': budget.json()}
        R.write_json(o / 'RESULT.json', result)
        R.write_json(o / 'FINAL_SEAL.json', {'lock_sha256': R.sha(o / 'LOCK.json'),
                                             'files': {p.name: R.sha(p) for p in o.iterdir() if p.is_file()}})
        R.log('COMPLETE', step, json.dumps(result['verdict']), json.dumps({n: g_['devs_final_delta'] for n, g_ in gates.items()}))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(o / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'execute'))
    p.add_argument('step', choices=tuple(STEPS))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase == 'execute':
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'lock': lock, 'execute': execute}[a.phase](a.step)
