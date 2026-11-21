"""CMS-B S3: BC-CMS (lam_T=1) vs schedule-matched BC-REPLAY (lam_T=0), 2 seeds, gated against the sealed SINGLE_R.
Plan: agents/reports/cmsb_s3_plan_2026_11_19.md (amendment §2.4 in agents/tasks/task_cms_bounded_00.md).

    python -B -m exps.cmsb.s3 smoke --owner-authorized     # normal-only preflight P1-P6 (<= 15 GPU min)
    python -B -m exps.cmsb.s3 lock
    python -B -m exps.cmsb.s3 execute --owner-authorized
"""
from __future__ import annotations

import argparse
import gc
import json
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from exps.cmsb import bc as BC, dev as D, s0 as S0, step as ST
from exps.dinomaly import encoder as E
from exps.hope_s_01 import run as H
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_pilot_00.detector import fm_eq8
from exps.nar_warm_01 import run as W
from exps.tcs_00 import run as T
from exps.vbench_00 import run as V

OUT = R.ROOT / 'results/hope_cad/cmsb_s3'
S2 = R.ROOT / 'results/hope_cad/cmsb_s2'
ORDER, SEEDS = ST.ORDER, ST.SEEDS
ARMS = {'BC_CMS': 1.0, 'BC_REPLAY': 0.0}
PREFLIGHT = {'rho_max': .95, 'teacher_gap_min': .02, 'projected_e4_gpu_max': 2.0 * 3600, 'smoke_wall_max': 15 * 60}
GATE = dict(ST.GATE, continual_state_bytes=48 * 2**20)
CAPS = {'wall_seconds': 12600, 'gpu_seconds': 8100, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 600_000_000}
SOURCES = ('exps/cmsb/s3.py', 'exps/cmsb/bc.py', 'exps/cmsb/test_bc.py', 'exps/cmsb/cms.py', 'exps/cmsb/step.py',
           'exps/cmsb/dev.py', 'exps/hope_s_01/run.py', 'exps/nar_lite_01/run.py', 'exps/nar_pilot_00/core.py',
           'exps/nar_pilot_00/full.py', 'agents/tasks/task_cms_bounded_00.md',
           'agents/reports/cmsb_s3_plan_2026_11_19.md')
DIAG_EVERY = 10


def bind():
    W.OUT, W.CAPS, W.SOURCES = OUT, CAPS, SOURCES


@torch.no_grad()
def normal_check(lr, feats):
    """Held-out EVAL normals: slow-level reconstruction error and output/encoder std ratio (collapse guard)."""
    m = lr.model_s
    was = m.training
    m.eval()
    en, de = m(feats.cuda().float())
    e = float((1 - F.cosine_similarity(en[0], de[0], dim=1)).mean())
    ratio = float(de[0].std(dim=(2, 3)).mean() / en[0].std(dim=(2, 3)).mean())
    m.train(was)
    return {'eval_error': e, 'std_ratio': ratio}


# ---------------------------------------------------------------- E2 smoke (normal only)
def smoke():
    bind()
    OUT.mkdir(parents=True, exist_ok=True)
    K.require(not (OUT / 'SMOKE.json').exists(), 'smoke done; preserve')
    R.configure()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    load = ST.loader(ST.train_lock())
    lr = BC.BCLearner(2711, 1.0, encode=ST.encoder_fn())
    recs = []
    for c in ORDER[:2]:
        f, roles, rows = load(c)
        ev = f[roles['EVAL'][:8]].clone()
        before = normal_check(lr, ev)
        ts = time.monotonic()
        tr = lr.task(c, f, roles, rows, diag_every=DIAG_EVERY)
        sec = time.monotonic() - ts
        after = normal_check(lr, ev)
        recs.append({'task': c, 'seconds': sec, 'before': before, 'after': after, 'diag': tr['diag'],
                     'fast_first_last': [tr['fast'][0], float(np.mean(tr['fast'][-10:]))],
                     'consol_first_last': [tr['consol'][0], tr['consol'][-1]],
                     'reservoir_tags_at_consolidation': tr['reservoir_tags_at_consolidation'],
                     'counters': dict(lr.counters[len(recs)])})
        R.log('SMOKE', c, round(sec, 1), 's', json.dumps(before), json.dumps(after))
    diag2 = recs[1]['diag']
    rho = float(np.mean([d['rho'] for d in diag2])); gap = float(np.mean([d['teacher_gap'] for d in diag2]))
    per_task = recs[1]['seconds']                                   # task with replay = representative
    projected = per_task * len(ORDER) * len(ARMS) * len(SEEDS) + 4 * 150   # + per-run evaluation estimate
    wall = time.monotonic() - t0
    checks = {
        'P4_finite_decreasing': all(r['fast_first_last'][1] < r['fast_first_last'][0] for r in recs),
        'P5_slow_improves_on_current': all(r['after']['eval_error'] < r['before']['eval_error'] for r in recs),
        'P5_no_collapse': all(r['after']['std_ratio'] > .5 for r in recs),
        'P6_rho_below': rho < PREFLIGHT['rho_max'], 'P6_teacher_gap': gap >= PREFLIGHT['teacher_gap_min'],
        'P1_projected_cost': projected <= PREFLIGHT['projected_e4_gpu_max'],
        'smoke_wall': wall <= PREFLIGHT['smoke_wall_max'],
        'P3_ledger': lr.ledger()['continual_state_bytes'] <= GATE['continual_state_bytes']}
    res = {'records': recs, 'rho_mean_task2': rho, 'teacher_gap_mean_task2': gap, 'projected_e4_seconds': projected,
           'smoke_wall_seconds': wall, 'peak_cuda': torch.cuda.max_memory_allocated(), 'ledger': lr.ledger(),
           'checks': checks, 'decision': 'GO' if all(checks.values()) else 'STOP', 'preflight_rules': PREFLIGHT,
           'note': 'normal-only mechanics; no anomaly data opened'}
    R.write_json(OUT / 'SMOKE.json', res)
    R.log('SMOKE_DECISION', res['decision'], json.dumps(checks), 'rho', round(rho, 4), 'gap', round(gap, 4),
          'projected_h', round(projected / 3600, 2))


# ---------------------------------------------------------------- reuse of sealed S2 cells
def summarize(cs, cr):
    last = len(ORDER) - 1
    return {'S_stage': cs, 'R_final': cr,
            'S_final_macro': float(np.mean([cs[last, c]['pixel_ap'] for c in ORDER])),
            'S_acq_macro': float(np.mean([cs[i, c]['pixel_ap'] for i, c in enumerate(ORDER)])),
            'S_fm_eq8': fm_eq8(cs, ORDER, 'pixel_ap'),
            'S_final_auroc': float(np.mean([cs[last, c]['image_auroc'] for c in ORDER])),
            'R_final_macro': float(np.mean([cr[c]['pixel_ap'] for c in ORDER])),
            'R_final_auroc': float(np.mean([cr[c]['image_auroc'] for c in ORDER]))}


def sealed_s2(arm):
    seal = R.read(S2 / 'FINAL_SEAL.json')
    K.require(R.sha(S2 / 'RESULT.json') == seal['files']['RESULT.json'], 'S2 result seal')
    res = R.read(S2 / 'RESULT.json')
    out = {}
    for s in SEEDS:
        d = res['cells'][arm][str(s)]
        cs = {(int(k[:2]) - 1, k[3:]): v for k, v in d['S_stage'].items()}
        out[s] = summarize(cs, d['R_final'])
        K.require(abs(out[s]['R_final_macro'] - res['summary'][arm][str(s)]['R_final_macro']) < 1e-12, 'S2 reuse parity')
    return out


# ---------------------------------------------------------------- lock / execute
def lock():
    bind()
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    sm = R.read(OUT / 'SMOKE.json')
    K.require(sm['decision'] == 'GO', 'preflight STOP: full S3 not allowed')
    s0 = R.read(S0.OUT / 'RESULT.json')
    K.require(R.read(S2 / 'LOCK.json')['dev_cache_sha256'] == s0['dev_cache_sha256'], 'dev cache identity')
    K.require('Amendment S3' in (R.ROOT / 'agents/tasks/task_cms_bounded_00.md').read_text(), 'amendment not appended')
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES}, 'arms': ARMS, 'seeds': list(SEEDS), 'gate': GATE,
        'kf': BC.KF, 'kc': BC.KC, 'gamma': BC.GAMMA, 'slow_lr': BC.SLOW_LR, 'replay_bytes': BC.REPLAY,
        'nl_criterion': 'Dev-R final macro AP: BC_CMS > SINGLE_R and BC_CMS > BC_REPLAY in both seeds',
        'operational_gate': 'step.gate(BC_CMS vs sealed SINGLE_R) + Dev-R seed-mean delta > 0 + continual state <= 48 MiB',
        'reuse': {'S2_RESULT_sha256': R.sha(S2 / 'RESULT.json'), 'arms': ['SINGLE_R', 'F_last', 'F_archive']},
        'smoke_sha256': R.sha(OUT / 'SMOKE.json'), 'dev_cache_sha256': s0['dev_cache_sha256'],
        'train_lock_sha256': R.sha(V.OUT / 'LOCK.json'), 'caps': CAPS, 'torch': torch.__version__})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def cells_for(stages, shas):
    g = {}
    cs, cr = {}, {}
    last = len(ORDER) - 1
    for i, c in enumerate(ORDER):
        ds = D.load('devs', c, shas)
        es = BC.stage_grids(stages, ds['features'])
        for t in range(i, len(ORDER)):
            cs[t, c] = D.metrics_s(es[t], ds)['all']
        rr = D.load('devr', c, shas)
        er = BC.stage_grids(stages[-1:], rr['features'])
        cr[c] = D.cell(er[0], *D.geometry_r(rr['rows']))
        del es, er
    gc.collect(); torch.cuda.empty_cache()
    return summarize(cs, cr)


@torch.no_grad()
def latency(state, n=23):
    dtype = D.DTYPE
    enc = L.lite_encoder(dtype)
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    model.eval().to(dtype)
    arch = [{k: (a.to(dtype), b.to(dtype)) for k, (a, b) in state.items()}]
    times = []
    for i, r in enumerate(V.train_rows(ORDER[0])[:n]):
        t = time.perf_counter()
        x = E.preprocess(V.VISA / r['relative_path'])[None].cuda().to(dtype)
        g = L.expert_grids(model, loras, arch, L.tokens(enc, x).cpu(), dtype)[0]
        L.smooth392(g); torch.cuda.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t)
    del enc, model, loras
    return {'latency_mean_s': float(np.mean(times)), 'latency_p95_s': float(np.quantile(times, .95)), 'decoder_passes': 1}


def nl_criterion(cells):
    per = {s: {'vs_SINGLE_R': cells['BC_CMS'][s]['R_final_macro'] - cells['SINGLE_R'][s]['R_final_macro'],
               'vs_BC_REPLAY': cells['BC_CMS'][s]['R_final_macro'] - cells['BC_REPLAY'][s]['R_final_macro']} for s in SEEDS}
    devs = {s: cells['BC_CMS'][s]['S_final_macro'] - cells['BC_REPLAY'][s]['S_final_macro'] for s in SEEDS}
    return {'devr_deltas': per, 'devs_delta_vs_BC_REPLAY': devs,
            'passed': all(v['vs_SINGLE_R'] > 0 and v['vs_BC_REPLAY'] > 0 for v in per.values())}


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    lk = R.read(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    shas = lk['dev_cache_sha256']
    budget = W.Budget()
    R.configure()
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time()})
    try:
        cells = {a: sealed_s2(a) for a in ('SINGLE_R', 'F_last', 'F_archive')}
        ledgers, timing, lat = {}, {}, None
        load = ST.loader(ST.train_lock())
        for seed in SEEDS:
            for arm, lam in ARMS.items():
                with budget.phase(f'{arm}_s{seed}', gpu=True):
                    enc = ST.encoder_fn()
                    lr = BC.BCLearner(seed, lam, encode=enc)
                    stages, traces, t0 = [], [], time.monotonic()
                    for c in ORDER:
                        f, roles, rows = load(c)
                        tr = lr.task(c, f, roles, rows, diag_every=DIAG_EVERY, budget=budget)
                        stages.append(K.get_state(lr.loras_s))
                        traces.append({'task': c, 'fast_last10': float(np.mean(tr['fast'][-10:])),
                                       'consol_last': tr['consol'][-1], 'diag': tr['diag'],
                                       'normal_check': normal_check(lr, f[roles['EVAL'][:8]].clone())})
                        R.log('TRAIN', arm, seed, c, json.dumps(traces[-1]['normal_check']))
                        del f
                    update_s = time.monotonic() - t0
                    led = lr.ledger()
                    K.require(led['deploy_adapter_bytes_bf16'] <= GATE['adapter_bytes'] and
                              led['replay_bytes'] <= GATE['replay_bytes'] and
                              led['continual_state_bytes'] <= GATE['continual_state_bytes'], 'memory ledger over budget')
                    counters = {str(t): dict(v) for t, v in lr.counters.items()}
                    W.bounded_save(OUT / f'{arm}_s{seed}.pt', {'stages': stages, 'traces': traces, 'state': lr.state(),
                                                               'counters': counters},
                                   200_000_000, budget)
                    del lr, enc
                    gc.collect(); torch.cuda.empty_cache()
                    cells.setdefault(arm, {})[seed] = cells_for(stages, shas)
                    ledgers[f'{arm}_s{seed}'] = dict(led, counters=counters)
                    timing[f'{arm}_s{seed}'] = update_s
                    if lat is None:
                        lat = latency(stages[-1])
                R.log('ARM', arm, seed, json.dumps({k: v for k, v in cells[arm][seed].items() if k.endswith(('macro', 'auroc'))}))
        led0 = ledgers[f'BC_CMS_s{SEEDS[0]}']
        op = ST.gate(cells['BC_CMS'], cells['SINGLE_R'], led0)
        op['guards']['devr_seedmean_positive'] = op['devr_final_delta'] > 0
        op['guards']['continual_state'] = all(l['continual_state_bytes'] <= GATE['continual_state_bytes'] for l in ledgers.values())
        op['passed'] = all(op['guards'].values())
        nl = nl_criterion(cells)
        desc = {'BC_REPLAY_vs_SINGLE_R': ST.gate(cells['BC_REPLAY'], cells['SINGLE_R'], ledgers[f'BC_REPLAY_s{SEEDS[0]}']),
                'BC_CMS_vs_BC_REPLAY': ST.gate(cells['BC_CMS'], cells['BC_REPLAY'], led0)}
        rec = {a: ST.recovery(a, cells) for a in ('BC_CMS', 'BC_REPLAY', 'SINGLE_R')}
        summary = {a: {s: {k: v for k, v in cs.items() if k not in ('S_stage', 'R_final')} for s, cs in d.items()}
                   for a, d in cells.items()}
        result = {'operational_gate_BC_CMS_vs_SINGLE_R': op, 'nl_criterion': nl, 'descriptive': desc,
                  'headroom_recovered': rec, 'summary': summary, 'ledgers': ledgers, 'update_seconds': timing,
                  'inference': lat,
                  'per_category': {a: {str(s): {'S_final': {c: cs['S_stage'][len(ORDER) - 1, c] for c in ORDER},
                                                'R_final': cs['R_final']} for s, cs in d.items()} for a, d in cells.items()},
                  'cells': {a: {str(s): {'S_stage': {f'{t + 1:02}_{c}': v for (t, c), v in cs['S_stage'].items()},
                                         'R_final': cs['R_final']} for s, cs in d.items()} for a, d in cells.items()},
                  'verdict': {'operational': 'GAIN' if op['passed'] else 'NO_GAIN',
                              'nl_specific': 'NL_TRANSFER_SUPPORTED' if nl['passed'] else 'NL_TRANSFER_NOT_SUPPORTED',
                              'branch': 'GO_S5_BC_CMS' if (op['passed'] and nl['passed'])
                                        else 'CLOSE_CMS_PROMOTE_SINGLE_R'},
                  'exposure': 'Dev-S and Dev-R (VisA-DEV, already used for development) only; VisA-FINAL/MVTec untouched',
                  'resources': budget.json()}
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()}})
        R.log('COMPLETE', json.dumps(result['verdict']), json.dumps(nl['devr_deltas']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('smoke', 'lock', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase in ('smoke', 'execute'):
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'smoke': smoke, 'lock': lock, 'execute': execute}[a.phase]()
