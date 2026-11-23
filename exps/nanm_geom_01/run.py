"""NANM-GEOM-01 driver. Registration: agents/tasks/task_nanm_geom_01.md (hash-locked by `prereg`).

    source docs/recovery/recovery_env.sh
    python -B -m exps.nanm_geom_01.run prereg   # create-exclusive GEOM01_PREREG.json (registration/code/canonical hashes)
    python -B -m exps.nanm_geom_01.run run      # DESIGN-only factorial + controls -> GEOM01_RESULTS.json (one shot)

Reads only the 4 DESIGN VisA train caches on K: (mmap) and the NL-02 v1/v2 result JSONs. Writes only
results/hope_cad/nanm_geom_01/ (E:). Canonical G3 is called, never modified. SELECTED tensors are never opened.
"""
from __future__ import annotations

import json
import math
import resource
import subprocess
import sys
import time

import torch

from exps.nanm_geom_01 import geom as G
from exps.nanm_nl_02 import core as C
from exps.nanm_nl_02 import run as R
from exps.nanm_nl_02_v2 import run as V
from nanm import g3

ROOT = R.ROOT
OUT = ROOT / 'results/hope_cad/nanm_geom_01'
TASKFILE = ROOT / 'agents/tasks/task_nanm_geom_01.md'
CODE = [ROOT / 'exps/nanm_geom_01/geom.py', ROOT / 'exps/nanm_geom_01/run.py', ROOT / 'exps/nanm_geom_01/test_geom.py']
N_THREADS = 6
ALPHAS = {'X2a': 0.25, 'X2b': 0.75, 'X2c': 0.9}
ELIGIBLE = ('X1', 'X2a', 'X2b', 'X2c', 'X3')
GiB = R.GiB
log = R.log
mean = V.mean


def fsha(p):
    return R.fsha(p)


def rel(p):
    return str(p.relative_to(ROOT))


# ------------------------------------------------------------------ prereg
def prereg():
    p = OUT / 'GEOM01_PREREG.json'
    assert not p.exists(), 'prereg already written'
    v1 = V.verify_v1_lock(); assert v1['ok'], v1
    V.verify_v2_lock()
    lk = json.loads((R.RUN / 'PREREG_LOCK.json').read_text())
    rec = {'registration': {rel(TASKFILE): fsha(TASKFILE)}, 'code': {rel(f): fsha(f) for f in CODE},
           'canonical_g3': {f: fsha(ROOT / f) for f in lk['canonical_g3']}, 'canonical_g3_v1_lock': lk['canonical_g3'],
           'v1_prereg_lock': fsha(R.RUN / 'PREREG_LOCK.json'), 'v2_prereg_lock': fsha(V.RUN2 / 'PREREG_V2_LOCK.json'),
           'v2_results': fsha(V.RUN2 / 'DESIGN_V2_RESULTS.json'), 'U': json.loads((R.RUN / 'DESIGN_SELECTION.json').read_text())['U'],
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           'storage': R.storage()}
    assert rec['canonical_g3'] == lk['canonical_g3'], 'canonical G3 changed'
    R.wjson(p, rec, exclusive=True)
    log('PREREG written', fsha(p)[:16])


def verify_prereg():
    pr = json.loads((OUT / 'GEOM01_PREREG.json').read_text())
    bad = [f for f, h in {**pr['registration'], **pr['code'], **pr['canonical_g3']}.items() if fsha(ROOT / f) != h]
    v1 = V.verify_v1_lock()
    try:
        V.verify_v2_lock(); v2 = True
    except AssertionError as e:
        v2 = False; bad.append(f'v2 lock: {e}')
    return pr, {'mismatches': bad, 'v1_lock_ok': v1['ok'], 'v2_lock_ok': v2, 'ok': not bad and v1['ok'] and v2}


# ------------------------------------------------------------------ per-cell computation
def combo_eval(stm, j, w_idx, k_idx, lmap_fn, with_energy):
    Vq = stm.data[j]['EVAL'].double()
    Lm, ld = lmap_fn(w_idx)
    v, N, F = G.nll_parts(Vq, stm.Kv[k_idx], Lm, ld)
    out = {'nll': float(v.mean()), 'N': N, 'F': float(F.mean())}, v
    if with_energy:
        e = g3.score(stm.data[j]['EVAL'], stm.Kv[k_idx].float(), stm.W[w_idx].float())
        out[0]['energy'] = float(e.mean())
        out[0]['_e'] = e
    return out


def factorial_pair(stm, j, old, new, lmap_fn, with_energy):
    """old/new: stage indices. Returns combos, effects on nll/N/F(/energy) and per-query statistics."""
    res, vec = {}, {}
    for name, (w, k) in {'A': (old, old), 'B': (old, new), 'C': (new, old), 'D': (new, new)}.items():
        res[name], vec[name] = combo_eval(stm, j, w, k, lmap_fn, with_energy)
    eff = {q: G.factorial(*(res[c][q] for c in 'ABCD')) for q in (('nll', 'N', 'F', 'energy') if with_energy else ('nll', 'N', 'F'))}
    dq = vec['D'] - vec['A']
    stats = {'perquery_delta_sd': float(dq.std()), 'perquery_delta_mean': float(dq.mean()),
             'perquery_delta_frac_pos': float((dq > 0).double().mean()),
             'spearman_nll_A_D': G.spearman(vec['A'], vec['D'])}
    if with_energy:
        eA, eD = res['A'].pop('_e'), res['D'].pop('_e')
        for c in 'BC':
            res[c].pop('_e')
        stats['spearman_energy_A_D'] = G.spearman(eA, eD)
        de = (eD - eA).double()
        stats['energy_ratio_D_over_A_median'] = float((eD / eA.clamp_min(1e-12)).median())
        stats['energy_delta_sd'] = float(de.std())
    return res, eff, stats


def control_maps(stm, t):
    """(kind, payload) per control at stage t (A = I, task-free)."""
    S = sum(s.S for s in stm.states[:t + 1]); n = sum(s.n for s in stm.states[:t + 1])
    out = {'X0': ('L', (stm.W[t], stm.ldW[t])), 'X1': ('L', (stm.W[0], stm.ldW[0]))}
    for x, a in ALPHAS.items():
        W, _ = G.whiten_alpha(S, n, a); out[x] = ('L', (W, G.logdet(W)))
    out['X3'] = ('L', C.maps_for(None, sigma2=C.iso_sigma2(S, n)))
    groups = []
    for i in range(t + 1):
        Wi, _ = C.stage_W([stm.states[i]])
        groups.append((stm.states[i].keys.double(), Wi, G.logdet(Wi)))
    out['X4'] = ('G', groups)
    return out


def control_nll(stm, t, j, kind, payload):
    Vq = stm.data[j]['EVAL'].double()
    if kind == 'L':
        return float(C.nll_mixture(Vq, stm.Kv[t], *payload).mean())
    return float(G.nll_pertask_mixture(Vq, payload).mean())


def control_bytes(T, n_keys):
    keys = n_keys * g3.KEY_BYTES
    b = {x: keys + C.S_B for x in ('X0', 'X1', 'X2a', 'X2b', 'X2c')}
    b['X3'] = keys + 16                                                   # sigma^2 (fp64) + count
    b['X4'] = keys + T * C.S_B                                            # one S (or W) per task
    return b


def run_cell(stm, stored, hp, task_names):
    tc = time.time(); rec = {'pairs': [], 'cumulative': [], 'checks': {}}
    # v2-selected C1 trajectory (secondary condition; reconstruction of the stored +21.06 term)
    t0 = time.time()
    e1, phis, _, _ = V.run_traj(stm, 'C1', hp)
    rec['c1_wall_s'] = time.time() - t0
    rec['checks']['c1_ell_maxerr'] = max(abs(e1[t][j] - stored['ell']['C1'][t][j]) for t in range(4) for j in range(t + 1))
    I_map = lambda w: (stm.W[w], stm.ldW[w])
    k = 0; m3, m4, m6 = [], [], []
    for t in range(1, 4):
        phi_map = (lambda ph: (lambda w: C.Lmap_phi(stm.W[w], *ph)))(phis[t - 1])
        for j in range(t):
            sp = stored['pairs'][k]; k += 1
            assert sp['t'] == t and sp['j'] == j
            cI, eI, sI = factorial_pair(stm, j, t - 1, t, I_map, True)
            cP, eP, sP = factorial_pair(stm, j, t - 1, t, phi_map, False)
            m3 += [abs(cI['A']['nll'] - stored['ell']['C0'][t - 1][j]), abs(cI['D']['nll'] - stored['ell']['C0'][t][j])]
            m4 += [abs(cP['A']['nll'] - sp['before_prev_stage']), abs(cP['D']['nll'] - sp['before_update']),
                   abs(eP['nll']['delta'] - sp['geometry'])]
            bound = math.log(len(stm.Kv[t]) / len(stm.Kv[t - 1])) + 1e-9
            m6 += [eI['nll']['k_only'] <= bound, (cI['D']['nll'] - cI['C']['nll']) <= bound,
                   eP['nll']['k_only'] <= bound, (cP['D']['nll'] - cP['C']['nll']) <= bound]
            rec['pairs'].append({'t': t, 'j': j, 'task': task_names[j], 'new_task': task_names[t],
                                 'I': {'combos': cI, 'effects': eI, 'stats': sI},
                                 'phi': {'combos': cP, 'effects': eP, 'stats': sP}, 'keys_bound': bound - 1e-9})
    for j in range(3):                                                    # cumulative: own stage j -> final stage 3
        c, e, s = factorial_pair(stm, j, j, 3, I_map, True)
        rec['cumulative'].append({'j': j, 'task': task_names[j], 'combos': c, 'effects': e, 'stats': s})
    rec['checks'].update(m3_maxerr=max(m3), m4_maxerr=max(m4), m6_all=all(m6))
    # M7: fp64 energy vs canonical fp32 g3.score (stage-3 state, every task)
    rel_err = []
    for j in range(4):
        e32 = g3.score(stm.data[j]['EVAL'], stm.Kv[3].float(), stm.W[3].float()).double()
        e64 = G.energy64(stm.data[j]['EVAL'].double(), stm.Kv[3], stm.W[3])
        rel_err.append(float(((e32 - e64).abs() / e64.clamp_min(1e-12)).median()))
    rec['checks']['m7_energy_rel_median_max'] = max(rel_err)
    # controls (A = I, task-free, deployable trajectories)
    ctl = {x: [] for x in ('X0', 'X1', 'X2a', 'X2b', 'X2c', 'X3', 'X4')}; wall = {x: 0.0 for x in ctl}
    for t in range(4):
        maps = control_maps(stm, t)
        for x, (kind, payload) in maps.items():
            t0 = time.time()
            ctl[x].append([control_nll(stm, t, j, kind, payload) for j in range(t + 1)])
            wall[x] += time.time() - t0
    rec['checks']['x0_vs_stored_c0_maxerr'] = max(abs(ctl['X0'][t][j] - stored['ell']['C0'][t][j]) for t in range(4) for j in range(t + 1))
    rec['controls'] = {'ell': ctl, 'eval_wall_s': wall, 'criterion': {x: R.criterion(v) for x, v in ctl.items()}}
    rec['checks']['finite'] = all(math.isfinite(v) for x in ctl.values() for r in x for v in r) and all(
        math.isfinite(p[c]['combos'][q][m]) for p in rec['pairs'] for c in ('I', 'phi') for q in 'ABCD' for m in ('nll', 'N', 'F'))
    rec['wall_s'] = time.time() - tc
    return rec


# ------------------------------------------------------------------ aggregation
def agg_effects(cells, cond, q='nll', sel=lambda p: True):
    """Per-cell mean over selected pairs of each effect, then boot CI over cells (only cells with >= 1 selected pair)."""
    out = {}
    for e in ('delta', 'w_only', 'k_only', 'interaction', 'shapley_W', 'shapley_K'):
        per = [mean(p[cond]['effects'][q][e] for p in c['pairs'] if sel(p)) for c in cells.values() if any(sel(p) for p in c['pairs'])]
        out[e] = V.boot_ci(per) if len(per) > 1 else ({'mean': per[0]} if per else None)
    out['n_cells'] = len(per)
    return out


def control_summary(cells, U):
    out = {}
    for x in ('X1', 'X2a', 'X2b', 'X2c', 'X3', 'X4'):
        H, A_, geo = [], [], []
        for c in cells.values():
            e0, ex = c['controls']['ell']['X0'], c['controls']['ell'][x]
            H.append(mean(e0[t][j] - ex[t][j] for t in range(1, 4) for j in range(t)))
            A_.append(mean(ex[t][t] - e0[t][t] for t in range(4)))
            geo.append(mean(ex[t][j] - ex[t - 1][j] for t in range(1, 4) for j in range(t)))
        crit = [c['controls']['criterion'][x] - c['controls']['criterion']['X0'] for c in cells.values()]
        out[x] = {'H': mean(H), 'H_ci': V.boot_ci(H)['ci95_boot'], 'H_pos': sum(h > 0 for h in H), 'Acq': mean(A_),
                  'Acq_ci': V.boot_ci(A_)['ci95_boot'], 'geo': mean(geo), 'criterion_minus_X0': mean(crit),
                  'eligible': x in ELIGIBLE,
                  'actionable': x in ELIGIBLE and mean(H) >= 0.05 * U and sum(h > 0 for h in H) >= 7 and mean(A_) <= 0.02 * U}
    out['X0'] = {'geo': mean(mean(c['controls']['ell']['X0'][t][j] - c['controls']['ell']['X0'][t - 1][j]
                                   for t in range(1, 4) for j in range(t)) for c in cells.values())}
    return out


def summarize(cells, U, task_list):
    s = {'main_I': {q: agg_effects(cells, 'I', q) for q in ('nll', 'N', 'F', 'energy')},
         'reconstruct_phi': {q: agg_effects(cells, 'phi', q) for q in ('nll', 'N', 'F')}}
    s['by_boundary_I'] = {t: agg_effects(cells, 'I', 'nll', lambda p, t=t: p['t'] == t) for t in (1, 2, 3)}
    s['by_boundary_phi'] = {t: agg_effects(cells, 'phi', 'nll', lambda p, t=t: p['t'] == t) for t in (1, 2, 3)}
    s['by_hist_task_I'] = {n: agg_effects(cells, 'I', 'nll', lambda p, n=n: p['task'] == n) for n in task_list}
    s['by_new_task_I'] = {n: agg_effects(cells, 'I', 'nll', lambda p, n=n: p['new_task'] == n) for n in task_list}
    for key, f in (('by_order_I', lambda k: k.split('|')[0]), ('by_seed_I', lambda k: k.split('|')[1])):
        groups = sorted({f(k) for k in cells})
        s[key] = {g: {e: mean(mean(p['I']['effects']['nll'][e] for p in c['pairs']) for k, c in cells.items() if f(k) == g)
                      for e in ('delta', 'shapley_W', 'shapley_K', 'interaction')} for g in groups}
    m = s['main_I']['nll']
    d = m['delta']['mean']
    s['shares_I'] = {'W': m['shapley_W']['mean'] / d, 'K': m['shapley_K']['mean'] / d, 'interaction': m['interaction']['mean'] / d,
                     'normalisation_N_of_delta': s['main_I']['N']['delta']['mean'] / d, 'fit_F_of_delta': s['main_I']['F']['delta']['mean'] / d}
    s['cells_w_larger'] = sum(mean(p['I']['effects']['nll']['shapley_W'] for p in c['pairs']) >
                              mean(p['I']['effects']['nll']['shapley_K'] for p in c['pairs']) for c in cells.values())
    s['perquery_I'] = {k: mean(mean(p['I']['stats'][k] for p in c['pairs']) for c in cells.values())
                       for k in ('perquery_delta_sd', 'perquery_delta_mean', 'perquery_delta_frac_pos', 'spearman_nll_A_D',
                                 'spearman_energy_A_D', 'energy_ratio_D_over_A_median', 'energy_delta_sd')}
    s['cumulative_I'] = {e: V.boot_ci([mean(r['effects']['nll'][e] for r in c['cumulative']) for c in cells.values()])
                         for e in ('delta', 'w_only', 'k_only', 'interaction', 'shapley_W', 'shapley_K')}
    s['cumulative_I_N'] = mean(mean(r['effects']['N']['delta'] for r in c['cumulative']) for c in cells.values())
    s['cumulative_I_F'] = mean(mean(r['effects']['F']['delta'] for r in c['cumulative']) for c in cells.values())
    s['controls'] = control_summary(cells, U)
    return s


# ------------------------------------------------------------------ run
def run():
    pr, ver = verify_prereg()
    assert not (OUT / 'GEOM01_RESULTS.json').exists(), 'GEOM-01 already ran (one shot)'
    torch.set_num_threads(N_THREADS)
    st0 = R.storage(); t0 = time.time()
    U = pr['U']
    au = json.loads((R.RUN / 'AUDIT.json').read_text())
    v2 = json.loads((V.RUN2 / 'DESIGN_V2_RESULTS.json').read_text())
    hp = json.loads((V.RUN2 / 'DESIGN_V2_SELECTION.json').read_text())['selected']
    streams, _ = R.design_streams(au)
    log('DESIGN streams built', len(streams), '%.0fs' % (time.time() - t0))
    cells = {}
    for (n, s), stm in streams.items():
        names = [au['design'][i] for i in R.PERMS[n]]
        cells[f'{n}|{s}'] = run_cell(stm, v2['cells'][f'{n}|{s}'], hp, names)
        c = cells[f'{n}|{s}']
        log('cell', n, s, '%.0fs' % c['wall_s'], 'delta_I %.3f phiW %.3f phiK %.3f | delta_phi %.3f | checks %s' % (
            mean(p['I']['effects']['nll']['delta'] for p in c['pairs']), mean(p['I']['effects']['nll']['shapley_W'] for p in c['pairs']),
            mean(p['I']['effects']['nll']['shapley_K'] for p in c['pairs']), mean(p['phi']['effects']['nll']['delta'] for p in c['pairs']),
            {k: (round(v, 12) if isinstance(v, float) else v) for k, v in c['checks'].items()}))
    summ = summarize(cells, U, au['design'])
    opened, only_design = V.design_only(au)
    st1 = R.storage()
    geo_phi = summ['reconstruct_phi']['nll']['delta']['mean']
    ch = {
        'M1_only_design_opened': only_design,
        'M2_hashes': ver['ok'],
        'M3_C0_endpoints_maxerr': max(c['checks']['m3_maxerr'] for c in cells.values()),
        'M4_phi_endpoints_maxerr': max(c['checks']['m4_maxerr'] for c in cells.values()),
        'M4_c1_ell_maxerr': max(c['checks']['c1_ell_maxerr'] for c in cells.values()),
        'M4_geometry_mean9': geo_phi, 'M4_geometry_v2_stored': v2['summary']['geometry']['mean'],
        'M5_shapley_maxerr': max(abs(e['shapley_W'] + e['shapley_K'] - e['delta']) for c in cells.values() for p in c['pairs']
                                 for e in (p['I']['effects']['nll'], p['phi']['effects']['nll'])),
        'M6_keys_bound_all': all(c['checks']['m6_all'] for c in cells.values()),
        'M7_finite': all(c['checks']['finite'] for c in cells.values()),
        'M7_energy_rel_median_max': max(c['checks']['m7_energy_rel_median_max'] for c in cells.values()),
        'M8_storage': st0['D_free'] >= 30 * GiB and st1['D_free'] >= 30 * GiB,
        'x0_vs_stored_c0_maxerr': max(c['checks']['x0_vs_stored_c0_maxerr'] for c in cells.values()),
    }
    ch['pass'] = {
        'M1': ch['M1_only_design_opened'], 'M2': ch['M2_hashes'], 'M3': ch['M3_C0_endpoints_maxerr'] <= 1e-9 and ch['x0_vs_stored_c0_maxerr'] <= 1e-9,
        'M4': ch['M4_phi_endpoints_maxerr'] <= 1e-6 and ch['M4_c1_ell_maxerr'] <= 1e-6 and abs(geo_phi - ch['M4_geometry_v2_stored']) <= 1e-6,
        'M5': ch['M5_shapley_maxerr'] <= 1e-9, 'M6': ch['M6_keys_bound_all'],
        'M7': ch['M7_finite'] and ch['M7_energy_rel_median_max'] <= 1e-3, 'M8': ch['M8_storage']}
    mech_ok = all(ch['pass'].values())
    m = summ['main_I']['nll']
    elig = {x: summ['controls'][x] for x in ELIGIBLE}
    vd, why = G.verdict(mech_ok, m['delta']['mean'], m['shapley_W']['mean'], m['shapley_K']['mean'], m['interaction']['mean'],
                        summ['cells_w_larger'], len(cells), elig, U)
    res = {'verdict': vd, 'verdict_reason': why, 'U': U, 'thresholds': {'0.02U': 0.02 * U, '0.05U': 0.05 * U},
           'hp_v2_selected': hp, 'checks': ch, 'mechanical_pass': mech_ok, 'summary': summ,
           'bytes_T4': control_bytes(4, sum(len(s.keys) for s in next(iter(streams.values())).states)),
           'cells': cells, 'opened_files': opened, 'selected_accessed': not only_design, 'prereg_verification': ver,
           'storage_before': st0, 'storage_after': st1, 'vhdx_delta': st1['VHDX'] - st0['VHDX'],
           'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'seconds': time.time() - t0, 'threads': N_THREADS}
    R.wjson(OUT / 'GEOM01_RESULTS.json', res, exclusive=True)
    log('VERDICT', vd, why, '| checks', ch['pass'])


if __name__ == '__main__':
    torch.set_num_threads(N_THREADS)
    {'prereg': prereg, 'run': run}[sys.argv[1]]()
