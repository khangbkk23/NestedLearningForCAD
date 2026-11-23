"""NANM-NL-02 Stage 0 v2 driver (optimizer correction). Registration: agents/tasks/task_nanm_nl_02_stage0_v2.md.

    source docs/recovery/recovery_env.sh
    PY=~/miniconda3/envs/nested_learning/bin/python
    PYTHONPATH=$PWD $PY -B -m exps.nanm_nl_02_v2.run audit     # v1 lock intact, DESIGN custody, hashes -> AUDIT_V2.json
    PYTHONPATH=$PWD $PY -B -m exps.nanm_nl_02_v2.run select    # DESIGN: checks, optimizer/lr, EWC c, EMA decay -> DESIGN_V2_SELECTION.json
    PYTHONPATH=$PWD $PY -B -m exps.nanm_nl_02_v2.run design    # DESIGN: all arms, common start, decomposition, gates -> DESIGN_V2_RESULTS.json
    PYTHONPATH=$PWD $PY -B -m exps.nanm_nl_02_v2.run lock      # PREREG_V2_LOCK.json (create-exclusive)
    PYTHONPATH=$PWD $PY -B -m exps.nanm_nl_02_v2.run phase0b   # refuses unless PHASE0B_V2_AUTHORIZED exists and the v2 lock verifies

The v1 modules exps/nanm_nl_02/{run,core}.py are imported read-only (they stay hash-locked by v1 PREREG_LOCK.json).
Writes only results/hope_cad/nanm_nl_02_v2/ (E:). Only the 4 DESIGN caches are opened by audit/select/design (asserted).
"""
from __future__ import annotations

import json
import math
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from exps.nanm_nl_02 import core as C
from exps.nanm_nl_02 import run as R
from exps.nanm_nl_02_v2 import core_v2 as O
from exps.nanm_r0 import audit as A
from nanm import g3

ROOT = R.ROOT
RUN1 = R.RUN
RUN2 = ROOT / 'results/hope_cad/nanm_nl_02_v2'
TASKFILE2 = ROOT / 'agents/tasks/task_nanm_nl_02_stage0_v2.md'
CODE2 = [ROOT / 'exps/nanm_nl_02_v2/run.py', ROOT / 'exps/nanm_nl_02_v2/core_v2.py', ROOT / 'exps/nanm_nl_02_v2/test_v2.py']
F64 = torch.float64
GiB = R.GiB
STEPS, BATCH, N_FIT = R.STEPS, R.BATCH, R.N_FIT
LR_GRID = {'sgd': (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),                      # = v1 ETA_GRID (regression oracle)
           'adam': (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
           'sign': (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)}
KIND_ORDER = ('sgd', 'sign', 'adam')                                  # exact-tie order: stateless before stateful
EWC_GRID, EMA_GRID = R.EWC_GRID, R.EMA_GRID
MAX_REJECT_FRAC = 0.01
N_THREADS = 6                                                         # = torch default here and v1 Phase 0a (bitwise parity)
BOOT, BOOT_SEED = 10000, 2711
log = R.log


def fsha(p):
    return R.fsha(p)


def hp_key(kind, lr):
    return f'{O.OPT_ID[kind]}|{kind}|{lr:g}'


# ------------------------------------------------------------------ fast training (v1 train_stage with the optimizer swapped)
def train_stage(stream, t, arm, phi, aux, hp, opt, steps=None, diag=None):
    """Mirrors exps.nanm_nl_02.run.train_stage line by line; the only change is `opt.step` in place of core.sgd_step.
    For kind 'sgd' it is bitwise identical to v1 (tested, and checked on DESIGN as M13)."""
    steps = STEPS if steps is None else steps
    P, Q = phi
    fit = stream.data[t]['FIT']; fimg = stream.data[t]['FIT_img']
    reps = R.replay_sets(stream, t, arm) if arm in ('C2', 'C4', 'C5') else []
    if arm in ('C2', 'C4', 'C5'):
        assert R.reserve_alloc(arm, t + 1) == O.reserve_alloc_v2(arm, t + 1, opt.kind)
    if arm == 'C4' and t >= 1:
        aux.basis = R.proj_basis(stream, t, P, Q)
    if arm == 'C5' and aux.ema is None:
        aux.ema = (P.clone(), Q.clone())
    lam = None
    if arm == 'C3' and aux.F is not None:
        lam = hp['ewc_c'] / float(torch.cat([aux.F[0].flatten(), aux.F[1].flatten()]).mean().clamp_min(1e-30))
    proj = None
    if arm == 'C4' and aux.basis is not None and opt.kind != 'sgd':
        B_ = aux.basis
        proj = lambda G: G - B_ @ (B_.T @ G)
    g = R.batch_gen(stream, t)
    n_tok = BATCH + sum(len(r[0]) for r in reps)
    for s in range(steps):
        idx = torch.randperm(N_FIT, generator=g)[:BATCH]
        Pg = P.detach().requires_grad_(True); Qg = Q.detach().requires_grad_(True)
        L, _ = R.loss_terms(stream, t, Pg, Qg, [(fit[idx].double(), fimg[idx])] + reps)
        if lam is not None:
            L = L + 0.5 * lam * ((aux.F[0] * (Pg - aux.anchor[0]) ** 2).sum() + (aux.F[1] * (Qg - aux.anchor[1]) ** 2).sum())
        gP, gQ = torch.autograd.grad(L, (Pg, Qg))
        if diag is not None and s == 0:
            diag.setdefault('first_grad_Q', []).append(float(gQ.norm())); diag.setdefault('first_grad_P', []).append(float(gP.norm()))
        if arm == 'C4' and aux.basis is not None:
            gP = gP - aux.basis @ (aux.basis.T @ gP); gQ = gQ - aux.basis @ (aux.basis.T @ gQ)
        P, Q, h = opt.step(P, Q, gP, gQ, proj)
        if arm == 'C5':
            dcy = hp['ema']; aux.ema = (dcy * aux.ema[0] + (1 - dcy) * P, dcy * aux.ema[1] + (1 - dcy) * Q)
    if diag is not None:
        diag['token_grads'] = diag.get('token_grads', 0) + steps * n_tok
    if arm == 'C3':
        FP, FQ = R.probe_grads_sq(stream, t, P, Q, (stream.data[t]['FEEDBACK'].double(), stream.data[t]['FEEDBACK_img']))
        aux.F = (FP, FQ) if aux.F is None else (aux.F[0] + FP, aux.F[1] + FQ)
        aux.anchor = (P.clone(), Q.clone())
    return (P, Q)


def run_traj(stream, arm, hp, role='EVAL', diag=None, stage_stats=False):
    """Full deployable trajectory; the optimizer state persists across boundaries. Returns ell, phis, opt snapshots
    (end of each stage; used for the common start) and per-stage diagnostics."""
    ell, phis, snaps, per = [], [], [], []
    phi = C.init_phi(stream.seed); aux = R.Aux()
    opt = O.FastOpt(hp['opt'], hp['lr']) if arm not in ('C0', 'C0q') else None
    for t in range(len(stream.data)):
        if arm in ('C0', 'C0q'):
            Lm, ld = R.metric(stream, t, arm)
        else:
            t0 = time.time(); before = dict(opt.stats)
            phi = train_stage(stream, t, arm, phi, aux, hp, opt, diag=diag)
            sp = R.scoring_phi(arm, phi, aux)
            s, _ = C.logdet_A(*sp)
            if diag is not None and float(s) <= 0:
                diag['nonpositive_det'] = diag.get('nonpositive_det', 0) + 1
            Lm, ld = R.metric(stream, t, arm, sp)
            snaps.append(opt.clone())
            if stage_stats:
                d = {k: opt.stats[k] - before[k] for k in ('steps', 'rejected', 'nonfinite_grad', 'halvings', 'step_norm_sum', 'grad_norm_sum')}
                d['grad_norm_max_cum'] = opt.stats['grad_norm_max']; d['min_logdet_cum'] = opt.stats['min_logdet']
                d['mean_step_norm'] = d['step_norm_sum'] / max(1, d['steps'] - d['rejected'])
                d['mean_grad_norm'] = d['grad_norm_sum'] / max(1, d['steps'] - d['nonfinite_grad'])
                d.update(O.phi_stats(*sp)); d['wall_s'] = time.time() - t0
                per.append(d)
        ell.append([stream.nll(t, j, Lm, ld, role) for j in range(t + 1)])
        phis.append(phi)
    if diag is not None and opt is not None:
        for k in ('steps', 'rejected', 'nonfinite_grad', 'halvings'):
            diag[k] = diag.get(k, 0) + opt.stats[k]
    return ell, phis, snaps, per


# ------------------------------------------------------------------ Phase 2: candidate summary from the C1 trajectory
def c1_summary(stream, ell, phis, ref):
    """ACQ, historical, damage and geometry from one C1 trajectory (paired, same HOLDOUT queries). ref: C0 / C0q ell."""
    acq = [ell[t][t] for t in range(4)]
    dmg, geo = [], []
    for t in range(1, 4):
        Lg, lg = R.metric(stream, t, 'phi', phis[t - 1])
        for j in range(t):
            g_ = stream.nll(t, j, Lg, lg)
            dmg.append(ell[t][j] - g_); geo.append(g_ - ell[t - 1][j])
    hist = [sum(ell[t][:t]) / t for t in range(1, 4)]
    hist0 = [sum(ref['C0'][t][:t]) / t for t in range(1, 4)]
    return {'acq': acq, 'hist': hist,
            'delta_acq': sum(min(ref['C0'][t][t], ref['C0q'][t][t]) - acq[t] for t in range(4)) / 4,
            'delta_acq_c0': sum(ref['C0'][t][t] - acq[t] for t in range(4)) / 4,
            'hist_gain_vs_c0': sum(h0 - h for h0, h in zip(hist0, hist)) / 3,
            'damage': dmg, 'D': sum(dmg) / 6, 'geometry': geo,
            'worst_vs_c0': max(ell[t][j] - ref['C0'][t][j] for t in range(1, 4) for j in range(t)),
            'worst_final_vs_own_acq': max(ell[3][j] - ell[j][j] for j in range(3)),
            'criterion': R.criterion(ell)}


def admissible(diag, crit):
    return math.isfinite(crit) and diag.get('nonfinite_grad', 0) == 0 and diag.get('nonpositive_det', 0) == 0 and \
        diag.get('rejected', 0) <= MAX_REJECT_FRAC * diag['steps']


def mean(x):
    x = list(x); return sum(x) / len(x)


# ------------------------------------------------------------------ phases
def storage():
    return R.storage()


def design_only(au):
    allowed = {str(R.fpath(*t.split('/'))) for t in au['design']}
    return sorted(set(R.OPENED)), set(R.OPENED) <= allowed


def verify_v1_lock():
    lk = json.loads((RUN1 / 'PREREG_LOCK.json').read_text())
    files = {**lk['registration'], **lk['source'], **lk['canonical_g3']}
    bad = [f for f, h in files.items() if fsha(ROOT / f) != h]
    bad += [n for n, h in (('AUDIT.json', lk['audit']), ('DESIGN_SELECTION.json', lk['design_selection'])) if fsha(RUN1 / n) != h]
    return {'prereg_lock_sha256': fsha(RUN1 / 'PREREG_LOCK.json'), 'mismatches': bad, 'ok': not bad}


def audit():
    RUN2.mkdir(parents=True, exist_ok=True)
    assert not (RUN2 / 'AUDIT_V2.json').exists(), 'audit already written'
    au = json.loads((RUN1 / 'AUDIT.json').read_text())
    st0 = storage()
    v1 = verify_v1_lock()
    cust = {}
    for t in au['design']:                                        # DESIGN caches only; SELECTED custody stays v1 AUDIT's record
        f = au['files'][t]; p = Path(f['file'])
        cust[t] = {'sha256': fsha(p), 'expected': f['sha256'], 'json_sha256': fsha(p.with_suffix('.json')),
                   'json_expected': f['json_file_sha256']}
        cust[t]['ok'] = cust[t]['sha256'] == cust[t]['expected'] and cust[t]['json_sha256'] == cust[t]['json_expected']
        log('custody', t, cust[t]['ok'])
    rec = {'registration_v2_sha256': fsha(TASKFILE2), 'registration_v1_sha256': fsha(R.TASKFILE),
           'code_v2_sha256': {p.name: fsha(p) for p in CODE2}, 'v1_lock': v1, 'design_custody': cust,
           'v1_roles_problems': au['roles_problems'], 'encoder_ok_v1_audit': au['encoder_ok'],
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           'git_status_short': subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True).stdout,
           'storage': st0, 'storage_ok': st0['D_free'] >= 30 * GiB, 'torch': torch.__version__, 'threads': N_THREADS}
    rec['pass'] = v1['ok'] and all(c['ok'] for c in cust.values()) and not au['roles_problems'] and rec['storage_ok'] and au['encoder_ok']
    R.wjson(RUN2 / 'AUDIT_V2.json', rec, exclusive=True)
    log('AUDIT_V2', 'PASS' if rec['pass'] else 'FAIL', v1['mismatches'])


def guard(stage):
    a2 = json.loads((RUN2 / 'AUDIT_V2.json').read_text())
    assert a2['pass'], 'AUDIT_V2 failed'
    assert fsha(TASKFILE2) == a2['registration_v2_sha256'], 'v2 registration changed after audit'
    assert {p.name: fsha(p) for p in CODE2} == a2['code_v2_sha256'], 'v2 code changed after audit'
    assert verify_v1_lock()['ok'], 'v1 lock no longer verifies'
    return a2


def noisy_override(au):
    fit = set().union(*[set(au['files'][t]['roles']['FIT']) for t in au['design']])
    def ov(paths, feats):
        class F:
            def __getitem__(self, i):
                return feats[i] if paths[i] in fit else torch.randn(feats[i].shape, generator=torch.Generator().manual_seed(i))
        return F()
    return ov


def select():
    a2 = guard('select')
    assert not (RUN2 / 'DESIGN_V2_SELECTION.json').exists(), 'select already ran'
    au = json.loads((RUN1 / 'AUDIT.json').read_text()); dsel1 = json.loads((RUN1 / 'DESIGN_SELECTION.json').read_text())
    U = dsel1['U']
    st0 = storage(); t0 = time.time()
    streams, data = R.design_streams(au)
    files, ok_open = design_only(au); assert ok_open, files
    log('DESIGN loaded', list(data))
    mech = {}
    # M4 noise replacement (as v1)
    nd = {t: R.load_task(t, i, want=('FIT',), feat_override=noisy_override(au)) for i, t in enumerate(au['design'])}
    ok = True
    for (n, s), stm in streams.items():
        if s != R.SEEDS[0]:
            continue
        for k, task in enumerate([au['design'][i] for i in R.PERMS[n]]):
            st2, _ = C.g3_state_fit_only(nd[task]['FIT'], nd[task]['FIT_img'], s, k=R.K_KEYS)
            ok &= torch.equal(st2.kc, stm.states[k].kc) and torch.equal(st2.S, stm.states[k].S) and st2.n == stm.states[k].n
    mech['M4_noise_replacement_invariant'] = bool(ok)
    stm = streams[('a', R.SEEDS[0])]
    # M5 zero-step identity for every optimizer kind
    q16 = stm.data[0]['EVAL'][:512]; m5 = {}
    for kind in O.KINDS:
        opt = O.FastOpt(kind, 1e-3); P0, Q0 = train_stage(stm, 0, 'C1', C.init_phi(R.SEEDS[0]), R.Aux(), {}, opt, steps=0)
        m5[kind] = torch.equal(C.A_of(P0, Q0), torch.eye(C.D, dtype=F64)) and torch.equal(
            g3.score(q16, stm.Kv[0].float(), stm.W[0].float()), g3.score(q16, stm.Kv[0].float(), C.Lmap_phi(stm.W[0], P0, Q0)[0].float()))
    mech['M5_zero_step_identity'] = m5
    # references
    ref = {k: {a: R.run_traj(stm_, a, {})[0] for a in ('C0', 'C0q')} for k, stm_ in streams.items()}
    ref_crit = {a: mean(R.criterion(ref[k][a]) for k in streams) for a in ('C0', 'C0q')}
    # ---- Phase 2 grid (C1, all optimizers)
    cand = {}
    for kind in KIND_ORDER:
        for lr in LR_GRID[kind]:
            hp = {'opt': kind, 'lr': lr}; dg = {}; cells = {}; tw = time.time()
            for k, stm_ in streams.items():
                ell, phis, snaps, per = run_traj(stm_, 'C1', hp, diag=dg, stage_stats=True)
                cells[f'{k[0]}|{k[1]}'] = {**c1_summary(stm_, ell, phis, ref[k]), 'ell': ell, 'stages': per}
            crit = mean(c['criterion'] for c in cells.values())
            crit = crit if math.isfinite(crit) else math.inf
            st = [p for c in cells.values() for p in c['stages']]
            cand[hp_key(kind, lr)] = {
                'opt': kind, 'lr': lr, 'criterion': crit, 'admissible': admissible(dg, crit),
                'mean9_delta_acq': mean(c['delta_acq'] for c in cells.values()),
                'mean9_delta_acq_c0': mean(c['delta_acq_c0'] for c in cells.values()),
                'cells_delta_acq_positive': sum(c['delta_acq'] > 0 for c in cells.values()),
                'mean9_hist_gain_vs_c0': mean(c['hist_gain_vs_c0'] for c in cells.values()),
                'mean9_D': mean(c['D'] for c in cells.values()), 'mean9_geometry': mean(mean(c['geometry']) for c in cells.values()),
                'worst_task_vs_c0': max(c['worst_vs_c0'] for c in cells.values()),
                'worst_final_vs_own_acq': max(c['worst_final_vs_own_acq'] for c in cells.values()),
                'steps': dg['steps'], 'rejected': dg.get('rejected', 0), 'nonfinite_grad': dg.get('nonfinite_grad', 0),
                'halvings': dg.get('halvings', 0), 'nonpositive_det': dg.get('nonpositive_det', 0),
                'first_grad_Q_min': min(dg['first_grad_Q']), 'token_grads': dg['token_grads'],
                'mean_grad_norm': mean(p['mean_grad_norm'] for p in st), 'max_grad_norm': max(p['grad_norm_max_cum'] for p in st),
                'mean_step_norm': mean(p['mean_step_norm'] for p in st), 'max_A_minus_I': max(p['A_minus_I_F'] for p in st),
                'max_cond_A': max(p['cond_A'] for p in st), 'min_logdet_A': min(p['min_logdet_cum'] for p in st),
                'min_det_sign': min(p['det_sign'] for p in st), 'all_finite': all(p['finite'] for p in st),
                'persistent_opt_bytes': O.OPT_STATE_B[kind], 'wall_s': time.time() - tw, 'cells': cells}
            c_ = cand[hp_key(kind, lr)]
            log('cand', hp_key(kind, lr), 'crit %.4f adm %s dacq %.2f D %.2f hist %.2f rej %d cond %.1f %.0fs' % (
                crit, c_['admissible'], c_['mean9_delta_acq'], c_['mean9_D'], c_['mean9_hist_gain_vs_c0'], c_['rejected'],
                c_['max_cond_A'], c_['wall_s']))
    # M13 O0 regression parity with v1 Phase 0a (criteria bitwise) and v1 arms on one cell
    m13 = {str(e): cand[hp_key('sgd', e)]['criterion'] == dsel1['criteria']['eta'][str(e)] for e in LR_GRID['sgd']}
    hp1 = {'eta': dsel1['selected']['eta'], 'ewc_c': dsel1['selected']['ewc_c'], 'ema': dsel1['selected']['ema']}
    hp0 = {'opt': 'sgd', 'lr': hp1['eta'], 'ewc_c': hp1['ewc_c'], 'ema': hp1['ema']}
    for arm in ('C2', 'C3', 'C4', 'C5'):
        e1, p1 = R.run_traj(stm, arm, hp1); e2, p2, _, _ = run_traj(stm, arm, hp0)
        m13[arm] = e1 == e2 and all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(p1, p2))
    mech['M13_O0_regression_parity'] = m13
    log('M13', m13)
    # ---- selection (registered): lowest criterion among admissible; exact ties -> KIND_ORDER, then grid order
    adm = [k for k, v in cand.items() if v['admissible']]
    if not adm:
        out = {'outcome': 'NL02_V2_PROTOCOL_BLOCKED', 'reason': 'no admissible optimizer candidate',
               'candidates': {k: {x: y for x, y in v.items() if x != 'cells'} for k, v in cand.items()}}
        R.wjson(RUN2 / 'DESIGN_V2_SELECTION.json', out, exclusive=True); log('BLOCKED: no admissible candidate'); return
    order = {hp_key(k, lr): (KIND_ORDER.index(k), LR_GRID[k].index(lr)) for k in KIND_ORDER for lr in LR_GRID[k]}
    best = min(adm, key=lambda k: (cand[k]['criterion'], order[k]))
    kind, lr = cand[best]['opt'], cand[best]['lr']
    best_per_kind = {kd: min((k for k in adm if cand[k]['opt'] == kd), key=lambda k: (cand[k]['criterion'], order[k]), default=None)
                     for kd in KIND_ORDER}
    log('selected', best, cand[best]['criterion'])
    sel = {'ewc_c': {}, 'ema': {}}; dgx = {}
    for cc in EWC_GRID:
        dg = {}; v = mean(R.criterion(run_traj(s_, 'C3', {'opt': kind, 'lr': lr, 'ewc_c': cc}, diag=dg)[0]) for s_ in streams.values())
        v = v if math.isfinite(v) else math.inf
        sel['ewc_c'][str(cc)] = v; dgx[f'ewc|{cc}'] = {**{k: dg.get(k, 0) for k in ('steps', 'rejected', 'nonfinite_grad', 'nonpositive_det')}, 'admissible': admissible(dg, v)}
        log('ewc', cc, v, dgx[f'ewc|{cc}'])
    for dc in EMA_GRID:
        dg = {}; v = mean(R.criterion(run_traj(s_, 'C5', {'opt': kind, 'lr': lr, 'ema': dc}, diag=dg)[0]) for s_ in streams.values())
        v = v if math.isfinite(v) else math.inf
        sel['ema'][str(dc)] = v; dgx[f'ema|{dc}'] = {**{k: dg.get(k, 0) for k in ('steps', 'rejected', 'nonfinite_grad', 'nonpositive_det')}, 'admissible': admissible(dg, v)}
        log('ema', dc, v, dgx[f'ema|{dc}'])
    ewc_ok = [c for c in EWC_GRID if dgx[f'ewc|{c}']['admissible']]; ema_ok = [c for c in EMA_GRID if dgx[f'ema|{c}']['admissible']]
    if not ewc_ok or not ema_ok:
        blocked = 'no admissible EWC c' if not ewc_ok else 'no admissible EMA decay'
    else:
        blocked = None
    ewc_c = min(ewc_ok, key=lambda c: (sel['ewc_c'][str(c)], EWC_GRID.index(c))) if ewc_ok else None
    ema = min(ema_ok, key=lambda c: (sel['ema'][str(c)], EMA_GRID.index(c))) if ema_ok else None
    hp = {'opt': kind, 'lr': lr, 'ewc_c': ewc_c, 'ema': ema}
    # M6 / M7 for the selected candidate; M8 future invariance; M9 determinism; M10 bytes; M11; M12
    cs = cand[best]
    mech['M6_first_grad_Q_min'] = cs['first_grad_Q_min']; mech['M6_ok'] = cs['first_grad_Q_min'] > 0
    mech['M7_rejected_frac'] = cs['rejected'] / cs['steps']; mech['M7_nonpositive_det'] = cs['nonpositive_det']
    mech['M7_nonfinite_grad'] = cs['nonfinite_grad']
    mech['M7_ok'] = mech['M7_rejected_frac'] <= MAX_REJECT_FRAC and cs['nonpositive_det'] == 0 and cs['nonfinite_grad'] == 0
    e1, ph1, sn1, _ = run_traj(stm, 'C1', hp); e2, ph2, sn2, _ = run_traj(stm, 'C1', hp)
    same_state = lambda a, b: a.t == b.t and (a.kind != 'adam' or all(torch.equal(x, y) for x, y in zip(a.m + a.v, b.m + b.v)))
    mech['M9_determinism'] = e1 == e2 and all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(ph1, ph2)) and \
        all(same_state(a, b) for a, b in zip(sn1, sn2))
    noise = lambda n, k: A.unit(torch.randn(n, C.D, generator=torch.Generator().manual_seed(k))).half()
    stm_n = R.Stream([stm.data[0], stm.data[1]] + [{**stm.data[k], 'FIT': noise(N_FIT, k), 'FEEDBACK': noise(32, 10 + k), 'RESERVE': noise(32, 20 + k)}
                                                   for k in (2, 3)], stm.seed, stm.name)
    m8 = {}
    for arm in ('C1', 'C2'):
        _, p1, s1, _ = run_traj(stm, arm, hp); _, p2, s2, _ = run_traj(stm_n, arm, hp)
        m8[arm] = all(torch.equal(p1[t][0], p2[t][0]) and torch.equal(p1[t][1], p2[t][1]) and same_state(s1[t], s2[t]) for t in (0, 1))
    mech['M8_future_invariance'] = m8
    mech['M8_eval_structural'] = 'train_stage/loss_terms receive FIT, FEEDBACK, RESERVE only (v1 functions, unchanged)'
    m10 = {}
    for arm in ('C1', 'C2', 'C3', 'C4', 'C5'):
        okb = True; rows = []
        for n in range(1, 5):
            fb = 32 * n if arm in ('C2', 'C3', 'C4') else 0
            lg = O.ledger_v2(arm, kind, n, [R.K_KEYS] * n, fb, O.reserve_alloc_v2(arm, n, kind))
            okb &= lg['total_charged'] <= O.budget_v2(n, kind, R.K_KEYS); rows.append(lg)
        m10[arm] = {'ok': okb, 'final': rows[-1], 'budget_final': O.budget_v2(4, kind, R.K_KEYS)}
    mech['M10_bytes'] = m10
    gB = torch.Generator().manual_seed(11)
    Bm = torch.eye(C.D, dtype=F64) + 0.02 * torch.randn(C.D, C.D, generator=gB, dtype=F64)
    Binv = torch.linalg.inv(Bm); ldB = float(torch.linalg.slogdet(Bm)[1])
    v = stm.data[0]['EVAL'][:256].double(); K = stm.Kv[0]
    Lm, ld = C.Lmap_phi(stm.W[0], *ph1[0])
    a_ = C.nll_mixture(v, K, Lm, ld); b_ = C.nll_mixture(v @ Bm, K @ Bm, Binv @ Lm, ld - ldB)
    mech['M11_reference_measure_max_abs_err_adapted'] = float((b_ - a_ - ldB).abs().max()); mech['M11_ok'] = mech['M11_reference_measure_max_abs_err_adapted'] < 1e-6
    files, ok_open = design_only(au)
    mech['M2_opened_files'] = files; mech['M2_only_design'] = ok_open
    st1 = storage()
    mech['M12_storage'] = {'D_free_min_ok': min(st0['D_free'], st1['D_free']) >= 30 * GiB, 'vhdx_growth': st1['VHDX'] - st0['VHDX']}
    mech['M1_custody'] = all(c['ok'] for c in a2['design_custody'].values()); mech['M3_roles_disjoint'] = not au['roles_problems']
    mech['M14_v1_lock_and_g3_integrity'] = verify_v1_lock()['ok']
    flags = [mech['M1_custody'], mech['M2_only_design'], mech['M3_roles_disjoint'], mech['M4_noise_replacement_invariant'],
             all(m5.values()), mech['M6_ok'], mech['M7_ok'], all(m8.values()), mech['M9_determinism'],
             all(x['ok'] for x in m10.values()), mech['M11_ok'],
             mech['M12_storage']['D_free_min_ok'] and mech['M12_storage']['vhdx_growth'] <= 0, all(m13.values()),
             mech['M14_v1_lock_and_g3_integrity'], blocked is None]
    out = {'U': U, 'U_source': 'v1 DESIGN_SELECTION.json (DESIGN-only, unchanged)', 'selected': hp, 'selected_key': best,
           'best_per_optimizer': best_per_kind, 'reference_criteria': ref_crit,
           'criteria': {'opt_lr': {k: v['criterion'] for k, v in cand.items()}, **sel}, 'grid_diag': dgx,
           'candidates': {k: {x: y for x, y in v.items() if x != 'cells'} for k, v in cand.items()},
           'candidate_cells': {k: v['cells'] for k, v in cand.items()},
           'reference_ell': {f'{k[0]}|{k[1]}': v for k, v in ref.items()},
           'mechanical': mech, 'mechanical_pass': all(flags), 'blocked_reason': blocked,
           'storage_before': st0, 'storage_after': st1, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           'seconds': round(time.time() - t0, 1), 'threads': torch.get_num_threads(), 'selected_accessed': False}
    R.wjson(RUN2 / 'DESIGN_V2_SELECTION.json', out, exclusive=True)
    log('SELECT', 'mechanical', 'PASS' if out['mechanical_pass'] else 'FAIL', hp)


# ------------------------------------------------------------------ Phase 3/5: full evaluation (DESIGN now; SELECTED only in phase0b)
def evaluate(streams, hp, U, role='EVAL'):
    """All arms + common-start paired interference decomposition + gates (v1 gate functions, unchanged).
    Common start at boundary t >= 1: every arm starts from C1's phi_{t-1} AND C1's optimizer state at the end of t-1."""
    cells, g0a_cells, g0b_cells = {}, [], []
    for (n, s), stm in streams.items():
        tc = time.time(); res, phis, snaps, stats = {}, {}, {}, {}
        for arm in ('C0', 'C0q', 'C1', 'C2', 'C3', 'C4', 'C5'):
            res[arm], phis[arm], snaps[arm], stats[arm] = run_traj(stm, arm, hp, role, stage_stats=arm not in ('C0', 'C0q'))
        res['R1'] = run_R1(stm, hp, role)
        c1p, c1s = phis['C1'], snaps['C1']
        pairs, acq_gap = [], {x: [] for x in ('C2', 'C3', 'C4', 'C5')}
        for t in range(1, 4):
            start = c1p[t - 1]
            Lb, lb = R.metric(stm, t - 1, 'phi', start); Lg, lg = R.metric(stm, t, 'phi', start)
            Ln, ln = R.metric(stm, t, 'phi', c1p[t])
            ctrl = {}
            for x in ('C2', 'C3', 'C4', 'C5'):
                aux = R.Aux()
                if x == 'C3':
                    F = None
                    for j in range(t):
                        FP, FQ = R.probe_grads_sq(stm, t, *start, (stm.data[j]['FEEDBACK'].double(), stm.data[j]['FEEDBACK_img']))
                        F = (FP, FQ) if F is None else (F[0] + FP, F[1] + FQ)
                    aux.F, aux.anchor = F, start
                if x == 'C5':
                    aux.ema = start
                opt = c1s[t - 1].clone()
                ph = train_stage(stm, t, x, start, aux, hp, opt)
                ctrl[x] = R.metric(stm, t, 'phi', R.scoring_phi(x, ph, aux))
            acq_n = stm.nll(t, t, Ln, ln)
            for x in ctrl:
                acq_gap[x].append(stm.nll(t, t, *ctrl[x]) - acq_n)
            for j in range(t):
                before = stm.nll(t - 1, j, Lb, lb); pre = stm.nll(t, j, Lg, lg); after = stm.nll(t, j, Ln, ln)
                nll_ctrl = {x: stm.nll(t, j, *ctrl[x]) for x in ctrl}
                phR2 = train_stage_task(stm, t, j, hp)
                pairs.append({'t': t, 'j': j, 'before_prev_stage': before, 'before_update': pre, 'after_c1': after, 'after_ctrl': nll_ctrl,
                              **C.decompose(before, pre, after, nll_ctrl, stm.nll(t, j, *R.metric(stm, t, 'phi', phR2))),
                              'c0_same_stage': res['C0'][t][j]})
        acq = {a: [res[a][t][t] for t in range(4)] for a in ('C0', 'C0q', 'C1', 'C2', 'C3', 'C4', 'C5')}
        g0a_cells.append({'ACQ': {a: acq[a] for a in ('C0', 'C0q', 'C1')}})
        g0b_cells.append({'damage': [p['damage'] for p in pairs], 'rec': {x: [p['recovery'][x] for p in pairs] for x in ('C2', 'C3', 'C4', 'C5')},
                          'acqcs_gap': {x: sum(v) / len(v) for x, v in acq_gap.items()}})
        cells[f'{n}|{s}'] = {'ell': res, 'acq': acq, 'pairs': pairs, 'acqcs_gap': acq_gap, 'stage_stats': stats, 'wall_s': time.time() - tc}
        log('cell', n, s, '%.0fs' % (time.time() - tc), 'dacq %.2f D %.2f' % (
            mean(min(acq['C0'][t], acq['C0q'][t]) - acq['C1'][t] for t in range(4)), mean(p['damage'] for p in pairs)))
    ga, gb = R.gate_g0a(g0a_cells, U), R.gate_g0b(g0b_cells, U)
    r1 = mean(min(mean(c['ell']['C0'][3]), mean(c['ell']['C0q'][3])) - mean(c['ell']['R1'][3]) for c in cells.values())
    return {'cells': cells, 'G0a': ga, 'G0b': gb, 'R1_headroom_diagnostic': r1, 'g0a_cells': g0a_cells, 'g0b_cells': g0b_cells}


def run_R1(stream, hp, role='EVAL'):
    """v1 run_R1 with the selected optimizer (fresh state at every stage; privileged diagnostic)."""
    ell = []
    for t in range(len(stream.data)):
        P, Q = C.init_phi(stream.seed); g = R.batch_gen(stream, t, 'R1'); per = BATCH // (t + 1); opt = O.FastOpt(hp['opt'], hp['lr'])
        for s in range(STEPS * (t + 1)):
            groups = []
            for j in range(t + 1):
                idx = torch.randperm(N_FIT, generator=g)[:per]
                groups.append((stream.data[j]['FIT'][idx].double(), stream.data[j]['FIT_img'][idx]))
            Pg = P.detach().requires_grad_(True); Qg = Q.detach().requires_grad_(True)
            L, _ = R.loss_terms(stream, t, Pg, Qg, groups)
            gP, gQ = torch.autograd.grad(L, (Pg, Qg))
            P, Q, _ = opt.step(P, Q, gP, gQ)
        Lm, ld = R.metric(stream, t, 'R', (P, Q))
        ell.append([stream.nll(t, j, Lm, ld, role) for j in range(t + 1)])
    return ell


def train_stage_task(stm, t, j, hp):
    """R2 (privileged diagnostic): from init, fresh optimizer, 200 steps on task j's FIT under W_t, K_<=t."""
    P, Q = C.init_phi(stm.seed); g = R.batch_gen(stm, t, f'R2|{j}'); opt = O.FastOpt(hp['opt'], hp['lr'])
    for s in range(STEPS):
        idx = torch.randperm(N_FIT, generator=g)[:BATCH]
        Pg = P.detach().requires_grad_(True); Qg = Q.detach().requires_grad_(True)
        L, _ = R.loss_terms(stm, t, Pg, Qg, [(stm.data[j]['FIT'][idx].double(), stm.data[j]['FIT_img'][idx])])
        gP, gQ = torch.autograd.grad(L, (Pg, Qg))
        P, Q, _ = opt.step(P, Q, gP, gQ)
    return (P, Q)


def boot_ci(x, stat=lambda v: sum(v) / len(v)):
    rng = random.Random(BOOT_SEED); n = len(x)
    bs = sorted(stat([x[rng.randrange(n)] for _ in range(n)]) for _ in range(BOOT))
    sd = (sum((v - mean(x)) ** 2 for v in x) / (n - 1)) ** 0.5
    return {'mean': mean(x), 'sd_cells': sd, 'se': sd / n ** 0.5, 'ci95_boot': (bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT) - 1]),
            'min': min(x), 'max': max(x)}


def summarize(ev, U):
    ga, gb = ev['G0a'], ev['G0b']
    D_c = [mean(c['damage']) for c in ev['g0b_cells']]
    xs = gb['X_star']
    rec_c = {x: [mean(c['rec'][x]) for c in ev['g0b_cells']] for x in ('C2', 'C3', 'C4', 'C5')}
    unrec = [d - (rec_c[xs][i] if xs else 0.0) for i, d in enumerate(D_c)]
    geo = [mean(p['geometry'] for p in c['pairs']) for c in ev['cells'].values()]
    worst = {a: max(c['ell'][a][t][j] - c['ell']['C0'][t][j] for c in ev['cells'].values() for t in range(1, 4) for j in range(t))
             for a in ('C1', 'C2', 'C3', 'C4', 'C5')}
    return {'delta_acq': boot_ci(ga['delta_acq_cells']), 'D': boot_ci(D_c), 'unrecovered_D_minus_Rec': boot_ci(unrec),
            'rec': {x: boot_ci(v) for x, v in rec_c.items()}, 'geometry': boot_ci(geo),
            'recovered_fraction_X_star': (gb['Rec'] / gb['D']) if gb['D'] else None,
            'worst_old_task_regression_vs_c0': worst,
            'thresholds': {'G0a_0.10U': 0.10 * U, 'G0b_D_0.10U': 0.10 * U, 'G0b_unrec_0.05U': 0.05 * U, 'guard_0.02U': 0.02 * U}}


def v2_outcome(mech_pass, ga, gb):
    """Registered DESIGN outcome mapping (registration §V2.8)."""
    if not mech_pass:
        return 'NL02_V2_PROTOCOL_BLOCKED', 'mechanical'
    if not ga['pass']:
        return 'NL02_V2_ADAPTATION_NOT_SUPPORTED', ga['outcome']
    if not gb['pass']:
        return 'NL02_V2_CONVENTIONAL_SUFFICIENT', gb['outcome']
    return 'NL02_V2_DESIGN_READY', 'PASS'


def design():
    guard('design')
    sp = RUN2 / 'DESIGN_V2_SELECTION.json'; assert sp.exists(), 'select has not run'
    assert not (RUN2 / 'DESIGN_V2_RESULTS.json').exists(), 'design already ran'
    dsel = json.loads(sp.read_text())
    if dsel.get('outcome') == 'NL02_V2_PROTOCOL_BLOCKED':
        raise SystemExit('selection blocked')
    au = json.loads((RUN1 / 'AUDIT.json').read_text()); U = dsel['U']; hp = dsel['selected']
    st0 = storage(); t0 = time.time()
    streams, _ = R.design_streams(au)
    files, ok_open = design_only(au); assert ok_open, files
    ev = evaluate(streams, hp, U)
    files, ok_open = design_only(au)
    summ = summarize(ev, U)
    st1 = storage()
    storage_ok = min(st0['D_free'], st1['D_free']) >= 30 * GiB and st1['VHDX'] - st0['VHDX'] <= 0
    mech_pass = dsel['mechanical_pass'] and ok_open and storage_ok
    outcome, sub = v2_outcome(mech_pass, ev['G0a'], ev['G0b'])
    ledgers = {arm: O.ledger_v2(arm, hp['opt'], 4, [R.K_KEYS] * 4, 128 if arm in ('C2', 'C3', 'C4') else 0, O.reserve_alloc_v2(arm, 4, hp['opt']))
               for arm in ('C1', 'C2', 'C3', 'C4', 'C5')}
    out = {'outcome': outcome, 'sub_outcome': sub, 'role': 'DESIGN HOLDOUT (diagnostic + selection evidence; NOT a confirmation)',
           'hp': hp, 'U': U, 'G0a': ev['G0a'], 'G0b': ev['G0b'], 'summary': summ, 'R1_headroom_diagnostic': ev['R1_headroom_diagnostic'],
           'ledger_final_boundary': ledgers, 'budget_final': O.budget_v2(4, hp['opt'], R.K_KEYS),
           'cells': ev['cells'], 'opened_files': files, 'only_design_opened': ok_open, 'selected_accessed': False,
           'mechanical_pass_select': dsel['mechanical_pass'], 'storage_before': st0, 'storage_after': st1, 'storage_ok': storage_ok,
           'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'seconds': round(time.time() - t0, 1),
           'threads': torch.get_num_threads()}
    R.wjson(RUN2 / 'DESIGN_V2_RESULTS.json', out, exclusive=True)
    log('DESIGN', outcome, sub, 'G0a', ev['G0a']['mean9_delta_acq'], ev['G0a']['cells_positive'], 'G0b D', ev['G0b']['D'], 'Rec', ev['G0b']['Rec'])


def lock():
    a2 = guard('lock')
    for f in ('DESIGN_V2_SELECTION.json', 'DESIGN_V2_RESULTS.json'):
        assert (RUN2 / f).exists(), f
    lk1 = json.loads((RUN1 / 'PREREG_LOCK.json').read_text())
    rec = {'version': 'NANM-NL-02 Stage 0 v2',
           'registration_v2': {str(TASKFILE2.relative_to(ROOT)): fsha(TASKFILE2)},
           'registration_v1': lk1['registration'], 'source_v2': {str(p.relative_to(ROOT)): fsha(p) for p in CODE2},
           'source_v1_imported': lk1['source'], 'canonical_g3': lk1['canonical_g3'],
           'v1_prereg_lock': fsha(RUN1 / 'PREREG_LOCK.json'), 'v1_audit': lk1['audit'], 'v1_design_selection': lk1['design_selection'],
           'audit_v2': fsha(RUN2 / 'AUDIT_V2.json'), 'design_v2_selection': fsha(RUN2 / 'DESIGN_V2_SELECTION.json'),
           'design_v2_results': fsha(RUN2 / 'DESIGN_V2_RESULTS.json'), 'data': lk1['data'], 'encoder_checkpoint': lk1['encoder_checkpoint'],
           'time': time.strftime('%FT%T'), 'note': 'frozen after DESIGN-only v2 calibration and before any SELECTED tensor is decoded'}
    for f, h in {**rec['registration_v1'], **rec['source_v1_imported'], **rec['canonical_g3']}.items():
        assert fsha(ROOT / f) == h, f
    R.wjson(RUN2 / 'PREREG_V2_LOCK.json', rec, exclusive=True)
    log('V2 LOCK written')


def verify_v2_lock():
    lk = json.loads((RUN2 / 'PREREG_V2_LOCK.json').read_text())
    for f, h in {**lk['registration_v2'], **lk['registration_v1'], **lk['source_v2'], **lk['source_v1_imported'], **lk['canonical_g3']}.items():
        assert fsha(ROOT / f) == h, f'v2 lock mismatch {f}'
    for f, k in (('AUDIT_V2.json', 'audit_v2'), ('DESIGN_V2_SELECTION.json', 'design_v2_selection'), ('DESIGN_V2_RESULTS.json', 'design_v2_results')):
        assert fsha(RUN2 / f) == lk[k], f
    assert fsha(RUN1 / 'PREREG_LOCK.json') == lk['v1_prereg_lock']
    return lk


def phase0b():
    """One-shot SELECTED run under the v2 lock. NOT authorized in this task; refuses without the owner's file."""
    assert (RUN2 / 'PHASE0B_V2_AUTHORIZED').exists(), 'Phase 0b v2 is not authorized'
    assert not (RUN2 / 'RESULT_V2.json').exists(), 'Phase 0b v2 already ran (one-shot)'
    verify_v2_lock()
    dsel = json.loads((RUN2 / 'DESIGN_V2_SELECTION.json').read_text()); dres = json.loads((RUN2 / 'DESIGN_V2_RESULTS.json').read_text())
    au = json.loads((RUN1 / 'AUDIT.json').read_text()); U = dsel['U']; hp = dsel['selected']
    st0 = storage()
    data = {t: R.load_task(t, i) for i, t in enumerate(au['selected'])}
    streams = {(n, s): R.Stream([data[au['selected'][i]] for i in perm], s, n) for n, perm in R.PERMS.items() for s in R.SEEDS}
    ev = evaluate(streams, hp, U)
    out = {'verdict': R.stage0_verdict(not dsel['mechanical_pass'], ev['G0a'], ev['G0b']), 'hp': hp, 'U': U,
           'G0a': ev['G0a'], 'G0b': ev['G0b'], 'summary': summarize(ev, U), 'cells': ev['cells'],
           'R1_headroom_diagnostic': ev['R1_headroom_diagnostic'], 'design_v2_outcome': dres['outcome'],
           'storage_before': st0, 'storage_after': storage()}
    R.wjson(RUN2 / 'RESULT_V2.json', out, exclusive=True)
    log('PHASE0B_V2', out['verdict'])


if __name__ == '__main__':
    torch.set_grad_enabled(True)
    torch.set_num_threads(N_THREADS)
    {'audit': audit, 'select': select, 'design': design, 'lock': lock, 'phase0b': phase0b}[sys.argv[1]]()
