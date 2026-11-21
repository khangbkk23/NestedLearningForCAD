"""CMS-B S0: bounded-memory references and synthetic-proxy check (agents/tasks/task_cms_bounded_00.md).

No training. Scores sealed detectors only:
  F-archive = VBENCH final.pt, task-free min over 12 experts (unbounded memory, O(T) passes)
  F-last    = VBENCH final.pt, archive[-1] only (one continually trained LoRA: bounded, single level)
  TCS-00 chains (6) = min over experts, for the proxy-validity check against their sealed VisA-DEV cells.

    python -B -m exps.cmsb.s0 lock
    python -B -m exps.cmsb.s0 execute --owner-authorized
"""
from __future__ import annotations

import argparse
import gc
import json
import signal
import time

import numpy as np
import torch

from exps.cmsb import dev as D
from exps.dinomaly import official as O
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_warm_01 import run as W
from exps.tcs_00 import run as T
from exps.vbench_00 import run as V
from exps.vitb8_q2.common import SingleStreamDecoder

OUT = R.ROOT / 'results/hope_cad/cmsb_s0'
ORDER = D.ORDER
GATE = {'headroom_small': .010, 'proxy_spearman_min': .3}
BOOT = T.BOOT
CAPS = {'wall_seconds': 5400, 'gpu_seconds': 3600, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 200_000_000}
SOURCES = ('exps/cmsb/s0.py', 'exps/cmsb/dev.py', 'exps/cmsb/test_s0.py', 'exps/ptag/synth.py',
           'exps/ptag/common.py', 'dataset/anomaly_generators/perlin.py', 'exps/tcs_00/run.py',
           'exps/vbench_00/run.py', 'exps/nar_lite_01/run.py', 'exps/nar_pilot_00/core.py',
           'exps/nar_pilot_00/full.py', 'agents/tasks/task_cms_bounded_00.md')
TCS_CHAINS = tuple(T.chain_name(g, s) for g in T.GAMMAS for s in T.SEEDS)


def bind():
    W.OUT, W.CAPS, W.SOURCES = OUT, CAPS, SOURCES


def lock():
    bind()
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    K.require((V.OUT / 'FINAL_SEAL.json').exists() and (T.OUT / 'FINAL_SEAL.json').exists(), 'sealed inputs')
    OUT.mkdir(parents=True, exist_ok=True)
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES}, 'devs_host_manifest_sha256': D.manifest_sha(),
        'dev_sha256': T.DEV_SHA, 'gate': GATE, 'bootstrap': BOOT, 'caps': CAPS, 'dtype': str(D.DTYPE),
        'detectors': {'vbench_final': R.read(V.OUT / 'TRAIN_COMPLETE.json')['final_sha256'],
                      'tcs_chains': {n: R.read(T.OUT / 'TRAIN_COMPLETE.json')['files'][f'{n}.pt'] for n in TCS_CHAINS}},
        'decision': 'proxy valid iff class-demeaned pooled Spearman(Dev-S AP, Dev-R AP) over 8 detectors x 12 '
                    'classes >= 0.3; H = macro AP(F-archive) - AP(F-last) on the primary set; H < 0.010 -> '
                    'BOUNDED_NEARLY_FREE else BOUNDED_HEADROOM',
        'torch': torch.__version__})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def verify():
    lk = R.read(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    K.require(lk['devs_host_manifest_sha256'] == D.manifest_sha(), 'host manifest')
    return lk


def detector(path, sha):
    K.require(R.sha(path) == sha, f'sealed detector changed: {path.name}')
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    model.eval().to(D.DTYPE)
    st = torch.load(path, weights_only=True)
    return model, loras, [{n: (a.to(D.DTYPE), b.to(D.DTYPE)) for n, (a, b) in s.items()} for s in st['archive']]


def base_regenerates():
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    torch.manual_seed(V.SEED)
    m = SingleStreamDecoder(checkpoint=True).cuda()
    O.init_trainable(m)
    st = m.state_dict()
    ok = set(st) == set(base) and all(torch.equal(st[k].cpu(), base[k]) for k in base)
    return {'bitwise_equal': ok, 'params': sum(v.numel() for v in base.values()),
            'fp32_bytes': sum(v.numel() * 4 for v in base.values())}


@torch.no_grad()
def score(lk, shas, budget):
    grids = {}
    model, loras, archive = detector(V.OUT / 'final.pt', lk['detectors']['vbench_final'])
    for c in ORDER:
        for kind in ('devs', 'devr'):
            g = L.expert_grids(model, loras, archive, D.load(kind, c, shas)['features'], D.DTYPE)
            grids['F_archive', kind, c] = g.min(0).values
            grids['F_last', kind, c] = g[-1].clone()
        budget.check()
    R.log('SCORE vbench')
    del model, loras, archive
    for n in TCS_CHAINS:
        model, loras, archive = detector(T.OUT / f'{n}.pt', lk['detectors']['tcs_chains'][n])
        for c in ORDER:
            grids[n, 'devs', c] = L.expert_grids(model, loras, archive, D.load('devs', c, shas)['features'],
                                                 D.DTYPE).min(0).values
        R.log('SCORE', n); budget.check()
        del model, loras, archive
        gc.collect(); torch.cuda.empty_cache()
    path = OUT / 'grids.pt'
    W.bounded_save(path, {'|'.join(k): v for k, v in grids.items()}, 60_000_000, budget)
    return grids, R.sha(path)


def parity(grids):
    """Fresh F-archive Dev-R grids vs the sealed VBENCH final-stage grids of the same DEV images (descriptive)."""
    num = den = 0.
    for c in ORDER:
        d = torch.load(V.OUT / f'scores_{c}.pt', weights_only=True)
        idx = [i for i, r in enumerate(d['rows']) if r['dev']]
        ref = d['prefix_grids'][-1][idx]
        new = grids['F_archive', 'devr', c]
        num += float((new - ref).abs().sum()); den += float(ref.abs().sum())
    return {'relative_mean_abs_diff': num / den}


def metrics(grids, shas, budget):
    tcs = R.read(T.OUT / 'CELLS.json')
    cells = {}
    for c in ORDER:
        ds = D.load('devs', c, shas)
        rr = D.load('devr', c, shas)['rows']
        geo = D.geometry_r(rr)
        for det in ('F_archive', 'F_last') + TCS_CHAINS:
            cells[det, c] = {'S': D.metrics_s(grids[det, 'devs', c], ds)}
            cells[det, c]['R'] = (D.cell(grids[det, 'devr', c], *geo) if det in ('F_archive', 'F_last')
                                  else tcs[f'{det}|{c}|full'])
        R.log('METRIC', c, round(cells['F_archive', c]['S']['all']['pixel_ap'], 4),
              round(cells['F_last', c]['S']['all']['pixel_ap'], 4))
        budget.check()
    return cells


def analyse(cells):
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    from scipy.stats import spearmanr
    dets = ('F_archive', 'F_last') + TCS_CHAINS
    ap = lambda d, c, s: cells[d, c][s]['all']['pixel_ap'] if s == 'S' else cells[d, c]['R']['pixel_ap']
    xs, ys = [], []
    for c in ORDER:
        ms, mr = np.mean([ap(d, c, 'S') for d in dets]), np.mean([ap(d, c, 'R') for d in dets])
        xs += [ap(d, c, 'S') - ms for d in dets]; ys += [ap(d, c, 'R') - mr for d in dets]
    rho = float(spearmanr(xs, ys).statistic)
    raw = float(spearmanr([ap(d, c, 'S') for d in dets for c in ORDER], [ap(d, c, 'R') for d in dets for c in ORDER]).statistic)
    valid = rho >= GATE['proxy_spearman_min']
    H = {s: bootstrap_delta([ap('F_archive', c, s) for c in ORDER], [ap('F_last', c, s) for c in ORDER],
                            seed=BOOT['seed'], B=BOOT['B']) for s in ('S', 'R')}
    sign = sum(np.sign(ap('F_archive', c, 'S') - ap('F_last', c, 'S')) == np.sign(ap('F_archive', c, 'R') - ap('F_last', c, 'R'))
               for c in ORDER)
    primary = 'S' if valid else 'R'
    macro = {d: {s: float(np.mean([ap(d, c, s) for c in ORDER])) for s in ('S', 'R')} for d in dets}
    auroc = {d: {'S': float(np.mean([cells[d, c]['S']['all']['image_auroc'] for c in ORDER])),
                 'R': float(np.mean([cells[d, c]['R']['image_auroc'] for c in ORDER]))} for d in dets}
    return {'proxy': {'demeaned_pooled_spearman': rho, 'raw_pooled_spearman': raw, 'valid': valid,
                      'n_cells': len(xs), 'headroom_sign_agreement_classes': int(sign)},
            'primary_set': 'Dev-S' if primary == 'S' else 'Dev-R',
            'headroom': {'Dev-S': H['S'], 'Dev-R': H['R']},
            'verdict': 'BOUNDED_NEARLY_FREE' if H[primary]['point'] < GATE['headroom_small'] else 'BOUNDED_HEADROOM',
            'macro_pixel_ap': macro, 'macro_image_auroc': auroc}


def ledger():
    st = torch.load(V.OUT / 'final.pt', weights_only=True)['archive']
    per = K.tensor_state_bytes(st[0])
    return {'expert_bytes_fp32': per, 'F_archive_adapter_bytes': per * len(st), 'F_last_adapter_bytes': per,
            'F_archive_decoder_passes': len(st), 'F_last_decoder_passes': 1,
            'note': 'adapter state only; encoder shared; BASE regenerable from seed (see base_regeneration)'}


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    budget = W.Budget()
    lk = verify()
    R.configure()
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time()})
    try:
        with budget.phase('build_dev_sets', gpu=True):
            shas = D.build(R.log)
        with budget.phase('scoring', gpu=True):
            grids, gsha = score(lk, shas, budget)
            base = base_regenerates()
        with budget.phase('metrics'):
            cells = metrics(grids, shas, budget)
        res = analyse(cells)
        verify()
        result = {**res, 'parity_F_archive_devr_vs_vbench': parity(grids), 'memory': ledger(),
                  'base_regeneration': base, 'dev_cache_sha256': shas, 'grids_sha256': gsha,
                  'cells': {f'{d}|{c}': v for (d, c), v in cells.items()},
                  'exposure': 'Dev-S: held-out train normals only. Dev-R: VisA-DEV (240), already consumed by '
                              'TCS-00; VisA-FINAL and MVTec not accessed.',
                  'resources': budget.json()}
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()}})
        R.log('COMPLETE', res['verdict'], json.dumps(res['proxy']), json.dumps(res['headroom']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase == 'execute':
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'lock': lock, 'execute': execute}[a.phase]()
