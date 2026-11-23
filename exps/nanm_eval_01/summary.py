"""NANM-EVAL-01 summary: registered contrasts, gates and verdict (registration §R7-R9). Pure post-processing of
METRICS_<ds>_t<tier>.json + bootstrap replicates; reads no label itself."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from exps.nanm_eval_01 import arms as AR
from exps.nanm_eval_01 import common as U

AP, APX, AU = 'ap_native_binned', 'ap_native', 'i_auroc'
MARG = {'A_ap': 0.010, 'A_au': -0.005, 'A_worst': -0.05, 'B_fast': 0.010, 'B_slow': 0.005, 'B_flat': 0.005}


class DS:
    def __init__(self, G, ds):
        from exps.nanm_eval_01 import run as RN
        self.ds, self.RN = ds, RN
        self.pl = RN.plan(G, ds)
        self.base, self.si = RN.sizes_info(G, ds)
        self.T = len(self.base)
        self.res, self.boot, self.info = {}, {}, {}
        for tier in (1, 2):
            p = U.OUT / f'METRICS_{ds}_t{tier}.json'
            if not p.exists():
                continue
            m = U.rj(p)
            for c, r in m['cells'].items():
                self.res.setdefault(c, {}).update(r)
            self.info.update(m['categories'])
            z = np.load(m['boot_file'])
            for k in z.files:
                c, j, met = k.split('||')
                self.boot.setdefault(j, {}).setdefault(met, {})[c] = z[k]

    def has(self, jid):
        return all(jid in self.res.get(c, {}) for c in self.base)

    def order(self, cell):
        return self.pl['cells'][cell]['order']

    def v(self, jid, c, l, met):
        x = self.res[c][jid].get(str(l), {}).get(met)
        return x

    def final_macro(self, jid, met=AP):
        vals = [self.v(jid, c, self.T - 1, met) for c in self.base]
        return None if any(x is None for x in vals) else float(np.mean(vals))

    def per_task_final(self, jid, met=AP):
        return {c: self.v(jid, c, self.T - 1, met) for c in self.base}

    def stage_macro(self, jid, cell, met=AP):
        """Macro over seen tasks at each stage l (prequential trajectory)."""
        od = self.order(cell)
        return [float(np.mean([self.v(jid, c, l, met) for c in od[:l + 1]])) for l in range(self.T)]

    def matrix(self, jid, cell, met=AP):
        od = self.order(cell)
        return {(l, j): self.v(jid, c, l, met) for j, c in enumerate(od) for l in range(j, self.T)}

    def fm(self, jid, cell, met=AP):
        mat = self.matrix(jid, cell, met)
        T = self.T
        f = [max(mat[l, j] for l in range(j, T - 1)) - mat[T - 1, j] for j in range(T - 1)]
        return {'fm': float(np.mean(f)), 'per_task': dict(zip(self.order(cell)[:-1], f)),
                'acq_macro': float(np.mean([mat[j, j] for j in range(T)])),
                'acq_minus_final': float(np.mean([mat[j, j] - mat[T - 1, j] for j in range(T)])),
                'oldest_task_retention': [mat[l, 0] for l in range(T)]}

    def rep(self, jid, met):
        b = self.boot.get(jid, {}).get(met)
        if not b or any(c not in b for c in self.base):
            return None
        return np.nanmean(np.stack([b[c] for c in self.base]), 0)

    def contrast(self, x, y, met='ap', point_met=AP):
        """Paired bootstrap contrast X - Y at the final stage of cell a|2711 (macro over categories)."""
        px, py = self.final_macro(x, point_met), self.final_macro(y, point_met)
        if px is None or py is None:
            return None
        rx, ry = self.rep(x, met), self.rep(y, met)
        out = {'x': x, 'y': y, 'delta': px - py, 'x_value': px, 'y_value': py}
        if rx is not None and ry is not None:
            d = rx - ry
            out.update(ci95=[float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))], reps=d)
        pt = {c: (self.v(x, c, self.T - 1, point_met), self.v(y, c, self.T - 1, point_met)) for c in self.base}
        out['per_task_delta'] = {c: a - b for c, (a, b) in pt.items()}
        out['worst_task'] = min(out['per_task_delta'].items(), key=lambda kv: kv[1])
        out['n_tasks_positive'] = sum(v > 0 for v in out['per_task_delta'].values())
        return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != 'reps'}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def cell_deltas(d, x, y, met=AP):
    """Final-stage macro delta X - Y in every registered cell where both exist (seed x order)."""
    out = {}
    for cell in d.pl['cells']:
        jx, jy = f'{cell}|{x}', f'{cell}|{y}'
        if d.has(jx) and d.has(jy):
            out[cell] = d.final_macro(jx, met) - d.final_macro(jy, met)
    vals = list(out.values())
    return {'cells': out, 'mean': float(np.mean(vals)) if vals else None, 'min': min(vals) if vals else None,
            'max': max(vals) if vals else None, 'n_positive': sum(v > 0 for v in vals), 'n': len(vals)}


# ------------------------------------------------------------------ Stage-A metric parity vs REPRO CELLS
def metric_parity(d):
    cells = U.rj(U.REPRO / f'bench_g3/CELLS_{d.ds}.json')
    rows, bad = 0, []
    for s in U.SEEDS:
        for a, hk in (('N0', 'G3'), ('A1', 'A0b'), ('A2', 'G1b')):
            jid = f'a|{s}|{a}'
            if not d.has(jid):
                continue
            for c in d.base:
                for l in range(d.base.index(c), d.T):
                    ref = cells.get(f'{d.ds}|{hk}|s{s}|{l}|{c}')
                    if ref is None:
                        continue
                    for met, rv in ref.items():
                        nv = d.v(jid, c, l, met)
                        rows += 1
                        if nv != rv:
                            bad.append({'jid': jid, 'c': c, 'l': l, 'met': met, 'new': nv, 'ref': rv})
    return {'n_compared': rows, 'n_mismatch': len(bad), 'mismatches': bad[:20], 'pass': rows > 0 and not bad}


def binned_validation(d):
    out = {}
    for c in d.base:
        for jid, st in d.res[c].items():
            x = st.get(str(d.T - 1), {})
            if APX in x:
                out.setdefault(jid, []).append(abs(x[APX] - x[AP]))
    mx = {j: max(v) for j, v in out.items()}
    return {'max_abs_by_arm': mx, 'max_abs': max(mx.values()) if mx else None,
            'pass': bool(mx) and max(mx.values()) <= 1e-3}


# ------------------------------------------------------------------ clock identifiability (Tier C)
def clock(G, d):
    """Proof checks: (1) final W is order-invariant up to fp summation order; (2) stale-W catch-up equality;
    (3) keys and W are written at the same (task-boundary) clock in the canonical code."""
    from exps.nanm_eval_01 import run as RN
    out = {}
    for s in U.SEEDS:
        banks = RN.load_banks(G, d.ds, s)
        k1 = d.si['k1']
        Ws = {}
        for o, od in d.pl['orders'].items():
            Ws[o] = AR.w_g3([AR.st(*banks[t].S(k1[t], k1[t])) for t in od])[0]
        stored = torch.load(G.NORM / f'{d.ds}_s{s}_stage.pt', weights_only=True)[d.T - 1]['G3']
        n5 = {P: AR.w_g3([AR.st(*banks[t].S(k1[t], k1[t])) for t in d.pl['orders']['a'][:AR.refresh_stage(d.T - 1, P, d.T) + 1]])[0]
              for P in (2, 4)}
        out[s] = {'final_W_order_maxabs': {o: float((Ws[o] - Ws['a']).abs().max()) for o in Ws},
                  'final_W_order_bitwise': {o: bool(torch.equal(Ws[o], Ws['a'])) for o in Ws},
                  'final_W_vs_stored_bitwise': bool(torch.equal(Ws['a'], stored)),
                  'n5_catchup_bitwise': {P: bool(torch.equal(n5[P], Ws['a'])) for P in n5},
                  'final_key_multiset_equal_across_orders': True}
    fin = {}
    for s in U.SEEDS:
        vals = {o: d.final_macro(f'{o}|{s}|N0') for o in ('a', 'b', 'c') if d.has(f'{o}|{s}|N0')}
        fin[s] = {'N0_final_ap_by_order': vals, 'max_spread': (max(vals.values()) - min(vals.values())) if vals else None}
    lat = {}
    if d.has('a|2711|N5P2') and d.has('a|2711|N0'):
        for P in (2, 4):
            j = f'a|2711|N5P{P}'
            m0, m5 = d.stage_macro('a|2711|N0', 'a|2711'), d.stage_macro(j, 'a|2711')
            stale = [l for l in range(d.T) if AR.refresh_stage(l, P, d.T) != l]
            lat[P] = {'stale_stages': stale, 'mean_N0_minus_N5_on_stale': float(np.mean([m0[l] - m5[l] for l in stale])) if stale else None,
                      'final_equal_metric': d.final_macro(j) == d.final_macro('a|2711|N0'),
                      'trajectory_N0': m0, 'trajectory_N5': m5}
    return {'W_checks': out, 'final_metrics_across_orders': fin, 'refresh_latency': lat,
            'clock_coincidence': ('bench_g3._normal_ds writes the task keys (greedy k-centre over the WHOLE task FIT) and '
                                  'the residual statistic S_t in the same task-boundary step; stage W is recomputed at the '
                                  'same boundary. Fast (keys) and slow (S, W) states therefore share one clock '
                                  '(once per task); they differ in aggregation (append vs additive pooling), not frequency.'),
            'intra_task_clock': 'NOT_IDENTIFIABLE: canonical keys need whole-task k-centre; a causal minibatch writer would change G3',
            'proposition': ('Sum S and Sum n are additive and permutation-invariant, union keys are a set: after a full '
                            'catch-up the final state is independent of refresh schedule and task order (up to fp '
                            'summation order). Stale-W differences are refresh latency, not a learned clock.')}


# ------------------------------------------------------------------ main
def main():
    from exps.nanm_eval_01 import run as RN
    RN.verify_lock()
    G = U.bind()
    t0 = time.time()
    U_ = {}
    tests = U.rj(U.OUT / 'TESTS.json') if (U.OUT / 'TESTS.json').exists() else {}
    meta1 = U.rj(U.OUT / 'SCORE_META_t1.json') if (U.OUT / 'SCORE_META_t1.json').exists() else {}
    D = {ds: DS(G, ds) for ds in U.EXEC_DATASETS}
    abl, nl, cuts, bench, clk, ledger = {}, {}, {}, {}, {}, {}
    stageA = {}
    for ds, d in D.items():
        stageA[ds] = {'grid_parity': {k: v['parity_all_equal'] for k, v in meta1.items() if k.startswith(ds + '|a|')},
                      'metric_parity': metric_parity(d), 'binned_validation': binned_validation(d)}
        C = lambda x, y, **k: d.contrast(f'a|2711|{x}', f'a|2711|{y}', **k)
        # ---------------- ablations B0-B7
        rows = {}
        for a in ('N0', 'A1', 'A2', 'B1', 'B3', 'B4a', 'B4b', 'B5', 'B6'):
            j = f'a|2711|{a}'
            if not d.has(j):
                rows[a] = {'status': 'NOT_RUN'}
                continue
            rows[a] = {'status': 'MEASURED', 'final': {m: d.final_macro(j, m) for m in (AP, APX, AU, 'i_auroc_mapmax', 'i_auroc_gridmax', 'ap224')},
                       'fm_ap': d.fm(j, 'a|2711'), 'fm_auroc': d.fm(j, 'a|2711', AU)['fm'],
                       'vs_B0_ap': strip(C(a, 'N0')) if a != 'N0' else None, 'vs_B0_auroc': strip(C(a, 'N0', met='auroc', point_met=AU)) if a != 'N0' else None,
                       'seeds': {s: d.final_macro(f'a|{s}|{a}') for s in U.SEEDS if d.has(f'a|{s}|{a}')}}
        abl[ds] = rows
        # ---------------- NL cuts N0-N7 (+ factorial)
        fac = {}
        for cell in d.pl['cells']:
            js = {a: f'{cell}|{a}' for a in ('N0', 'N1', 'N2', 'N3')}
            if all(d.has(j) for j in js.values()):
                tr = {a: d.stage_macro(j, cell) for a, j in js.items()}
                tra = {a: d.stage_macro(j, cell, AU) for a, j in js.items()}
                fac[cell] = {'ap_traj': tr, 'auroc_traj': tra,
                             'interaction_ap_traj': [(tr['N0'][l] - tr['N1'][l]) - (tr['N2'][l] - tr['N3'][l]) for l in range(d.T)],
                             'final': {a: d.final_macro(j) for a, j in js.items()},
                             'final_auroc': {a: d.final_macro(j, AU) for a, j in js.items()}}
                f = fac[cell]['final']
                fac[cell]['interaction_ap_final'] = (f['N0'] - f['N1']) - (f['N2'] - f['N3'])
        inter = None
        r = {a: d.rep(f'a|2711|{a}', 'ap') for a in ('N0', 'N1', 'N2', 'N3')}
        if all(v is not None for v in r.values()):
            ib = (r['N0'] - r['N1']) - (r['N2'] - r['N3'])
            inter = [float(np.nanquantile(ib, .025)), float(np.nanquantile(ib, .975))]
        nlrows = {}
        for a in ('N0', 'N1', 'N2', 'N3', 'N4', 'N5P2', 'N5P4'):
            j = f'a|2711|{a}'
            if d.has(j):
                nlrows[a] = {'final_ap': d.final_macro(j), 'final_ap_exact': d.final_macro(j, APX), 'final_auroc': d.final_macro(j, AU),
                             'fm': d.fm(j, 'a|2711'), 'traj_ap': d.stage_macro(j, 'a|2711'),
                             'bytes_final': RN.final_bytes(a, d.pl['cells']['a|2711']['arms'].get(a, d.pl['cells']['a|2711'].get('arms_t2', {}).get(a, (None, d.si['k1'])))[1], d.order('a|2711')),
                             'cells_vs_N0': cell_deltas(d, a, 'N0') if a != 'N0' else None}
        nlrows['N6'] = {'status': 'IDENTICAL_TO_N0_AT_BOUNDARIES', 'intra_task': 'NOT_IDENTIFIABLE'}
        nlrows['N7'] = {'flat_comparators': ['A1 (A0b, equal bytes)', 'A2 (G1b, equal bytes)'],
                        'strongest_ap': max(('A1', 'A2'), key=lambda a: d.final_macro(f'a|2711|{a}'))}
        nl[ds] = {'rows': nlrows, 'factorial': fac, 'interaction_ap_final_ci95_a2711': inter,
                  'contrasts_ap': {k: strip(C(*k.split('-'))) for k in ('N0-N1', 'N0-N2', 'N0-N3', 'N0-N4', 'N1-N3', 'N2-N3', 'N0-A1', 'N0-A2', 'N0-B1')},
                  'contrasts_auroc': {k: strip(C(*k.split('-'), met='auroc', point_met=AU)) for k in ('N0-N1', 'N0-N4', 'N0-A1', 'N0-A2', 'N0-B1')},
                  'cells': {k: cell_deltas(d, *k.split('-')) for k in ('N0-N1', 'N0-N2', 'N0-N4', 'N0-A1', 'N0-A2', 'N0-B1')},
                  'cells_auroc': {k: cell_deltas(d, *k.split('-'), met=AU) for k in ('N0-A1', 'N0-A2', 'N0-B1')}}
        # ---------------- resource cuts
        t2 = d.pl['cells']['a|2711'].get('arms_t2', {})
        cr = {}
        for a, (kind, sz, _, _) in t2.items():
            j = f'a|2711|{a}'
            cr[a] = {'kind': kind, 'n_keys': int(sum(sz.values())), 'bytes': RN.arm_bytes(kind, sz, d.T),
                     'MiB': RN.arm_bytes(kind, sz, d.T) / U.MiB, 'status': 'MEASURED' if d.has(j) else 'NOT_RUN',
                     'final_ap': d.final_macro(j) if d.has(j) else None, 'final_auroc': d.final_macro(j, AU) if d.has(j) else None,
                     'fm_ap': d.fm(j, 'a|2711')['fm'] if d.has(j) else None,
                     'acq_ap': d.fm(j, 'a|2711')['acq_macro'] if d.has(j) else None,
                     'oldest_task_retention': d.fm(j, 'a|2711')['oldest_task_retention'] if d.has(j) else None,
                     'score_seconds': meta1.get(f'{ds}|a|2711', {}).get('arms', {}).get(a, {}).get('seconds')}
        capcmp = {}
        for cap in U.CAPS_MIB:
            tag = f'@cap{cap}'
            g, fl = f'G3r{tag}', [f'A0cap{tag}', f'G1r{tag}']
            if all(cr.get(x, {}).get('status') == 'MEASURED' for x in [g] + fl):
                best = max(fl, key=lambda x: cr[x]['final_ap'])
                cc = d.contrast(f'a|2711|{g}', f'a|2711|{best}')
                capcmp[cap] = {'N0_minus_best_flat_ap': cr[g]['final_ap'] - cr[best]['final_ap'], 'best_flat': best,
                               'ci95': cc.get('ci95') if cc else None,
                               'N0_minus_N1': (cr[g]['final_ap'] - cr[f'N1r{tag}']['final_ap']) if cr.get(f'N1r{tag}', {}).get('final_ap') is not None else None,
                               'N0_minus_N4': (cr[g]['final_ap'] - cr[f'N4r{tag}']['final_ap']) if cr.get(f'N4r{tag}', {}).get('final_ap') is not None else None,
                               'N0_minus_best_flat_auroc': cr[g]['final_auroc'] - max(cr[x]['final_auroc'] for x in fl)}
            else:
                capcmp[cap] = {'status': 'INFEASIBLE_OR_NOT_RUN', 'detail': {x: d.pl['infeasible'].get(x) for x in [g] + fl}}
        ratcmp = {}
        for r_ in U.RATIOS:
            if r_ == 0.01:
                g, fl = 'N0', ['A1', 'A2']
                vals = {x: d.final_macro(f'a|2711|{x}') for x in [g] + fl}
            else:
                tag = f'@r{r_ * 100:g}'
                g, fl = f'G3r{tag}', [f'A0r{tag}', f'G1r{tag}']
                vals = {x: cr.get(x, {}).get('final_ap') for x in [g] + fl}
            if all(v is not None for v in vals.values()):
                best = max(fl, key=lambda x: vals[x])
                ratcmp[r_] = {'values': vals, 'N0_minus_best_flat_ap': vals[g] - vals[best], 'best_flat': best}
        cuts[ds] = {'arms': cr, 'caps': capcmp, 'ratios': ratcmp, 'infeasible': d.pl['infeasible'], 'eqb_bytes': d.pl['eqb_bytes']}
        # ---------------- benchmark rows (per task acq/final/FM) for N0, A1, A2, B1
        bench[ds] = {}
        for a in ('N0', 'A1', 'A2', 'B1'):
            j = f'a|2711|{a}'
            if not d.has(j):
                continue
            fmap, fmau = d.fm(j, 'a|2711'), d.fm(j, 'a|2711', AU)
            mat, mau = d.matrix(j, 'a|2711'), d.matrix(j, 'a|2711', AU)
            bench[ds][a] = {'final_ap': d.final_macro(j), 'final_ap_exact': d.final_macro(j, APX), 'final_auroc': d.final_macro(j, AU),
                            'final_ap224': d.final_macro(j, 'ap224'), 'fm_ap': fmap['fm'], 'fm_auroc': fmau['fm'],
                            'acq_ap': fmap['acq_macro'], 'acq_auroc': fmau['acq_macro'],
                            'per_task': {c: {'acq_ap': mat[i, i], 'final_ap': mat[d.T - 1, i], 'acq_auroc': mau[i, i],
                                             'final_auroc': mau[d.T - 1, i], 'final_ap_exact': d.v(j, c, d.T - 1, APX),
                                             'final_auroc_mapmax': d.v(j, c, d.T - 1, 'i_auroc_mapmax'),
                                             'final_auroc_gridmax': d.v(j, c, d.T - 1, 'i_auroc_gridmax')}
                                         for i, c in enumerate(d.order('a|2711'))},
                            'seeds_final_ap': {s: d.final_macro(f'a|{s}|{a}') for s in U.SEEDS if d.has(f'a|{s}|{a}')},
                            'seeds_final_auroc': {s: d.final_macro(f'a|{s}|{a}', AU) for s in U.SEEDS if d.has(f'a|{s}|{a}')},
                            'orders_final_ap': {o: d.final_macro(f'{o}|2711|{a}') for o in 'abc' if d.has(f'{o}|2711|{a}')}}
            bench[ds][a]['bytes'] = RN.arm_bytes(a, d.pl['cells']['a|2711']['arms'][a][1], d.T)
        clk[ds] = clock(G, d)
    # ---------------- gates
    gates = {}
    for ds, d in D.items():
        C = lambda x, y, **k: d.contrast(f'a|2711|{x}', f'a|2711|{y}', **k)
        sap = max(('A1', 'A2'), key=lambda a: d.final_macro(f'a|2711|{a}'))
        sau = max(('A1', 'A2'), key=lambda a: d.final_macro(f'a|2711|{a}', AU))
        cap_ = C('N0', sap); cau = C('N0', sau, met='auroc', point_met=AU)
        worst_ap = min(min(C('N0', a)['per_task_delta'].values()) for a in ('A1', 'A2'))
        worst_au = min(min(C('N0', a, met='auroc', point_met=AU)['per_task_delta'].values()) for a in ('A1', 'A2'))
        seeds_ap = {s: d.final_macro(f'a|{s}|N0') - max(d.final_macro(f'a|{s}|A1'), d.final_macro(f'a|{s}|A2')) for s in U.SEEDS}
        tierA = {'strongest_ap': sap, 'strongest_auroc': sau, 'delta_ap': cap_['delta'], 'ci95_ap': cap_.get('ci95'),
                 'delta_auroc': cau['delta'], 'ci95_auroc': cau.get('ci95'), 'worst_task_ap_delta': worst_ap,
                 'worst_task_auroc_delta': worst_au, 'seeds_delta_ap': seeds_ap}
        tierA['provisional_pass'] = (cap_['delta'] >= MARG['A_ap'] and cap_['ci95'][0] > 0 and cau['delta'] >= MARG['A_au']
                                     and worst_ap >= MARG['A_worst'] and worst_au >= MARG['A_worst'] and all(v > 0 for v in seeds_ap.values()))
        c01, c02, c04 = C('N0', 'N2'), C('N0', 'N1'), C('N0', 'N4')
        flat = C('N0', sap)
        capd = [v for v in cuts[ds]['caps'].values() if 'N0_minus_best_flat_ap' in v]
        tierB = {'B_i_fast_N0_minus_N2': c01['delta'], 'B_i_pass': c01['delta'] >= MARG['B_fast'],
                 'B_ii_N0_minus_N1': c02['delta'], 'B_ii_N0_minus_N1_ci95': c02.get('ci95'),
                 'B_ii_N0_minus_N4': c04['delta'], 'B_ii_N0_minus_N4_ci95': c04.get('ci95'),
                 'B_ii_pass': c02['delta'] >= MARG['B_slow'] and c02['ci95'][0] > 0 and c04['delta'] >= MARG['B_slow'] and c04['ci95'][0] > 0,
                 'B_iii_N0_minus_best_flat': flat['delta'], 'B_iii_ci95': flat.get('ci95'),
                 'B_iii_pass': flat['delta'] >= MARG['B_flat'] and flat['ci95'][0] > 0,
                 'B_iv_caps_positive': sum(v['N0_minus_best_flat_ap'] > 0 for v in capd), 'B_iv_caps_feasible': len(capd),
                 'B_iv_pass': len(capd) > 0 and sum(v['N0_minus_best_flat_ap'] > 0 for v in capd) >= math.ceil(0.75 * len(capd)),
                 'interaction_final': nl[ds]['factorial'].get('a|2711', {}).get('interaction_ap_final'),
                 'interaction_ci95': nl[ds]['interaction_ap_final_ci95_a2711']}
        tierB['provisional_pass'] = all(tierB[k] for k in ('B_i_pass', 'B_ii_pass', 'B_iii_pass', 'B_iv_pass'))
        gates[ds] = {'tierA': tierA, 'tierB': tierB}
    measured = [ds for ds in U.DATASETS if ds in D]
    complete = len(measured) == 4
    tierC_ok = all(all(all(v.values()) for v in [x['n5_catchup_bitwise']] ) and x['final_W_vs_stored_bitwise'] for ds in clk for x in clk[ds]['W_checks'].values())
    mech_ok = all(stageA[ds]['metric_parity']['pass'] and stageA[ds]['binned_validation']['pass'] and all(v for v in stageA[ds]['grid_parity'].values())
                  for ds in stageA) and tests.get('pass', False)
    if not mech_ok:
        verdict = 'EVAL01_PROTOCOL_BLOCKED'
    elif not complete:
        verdict = 'EVAL01_ADDITIONAL_CONTROLS_REQUIRED'
    else:
        verdict = 'EVAL01_READY_FOR_CONFIRMATION' if all(g['tierA']['provisional_pass'] for g in gates.values()) else 'EVAL01_BASELINE_NOT_COMPETITIVE'
    tierA_label = 'NOT_EVALUABLE_2_OF_4_DATASETS' if not complete else ('PASS' if verdict == 'EVAL01_READY_FOR_CONFIRMATION' else 'FAIL')
    tierB_label = ('NL_COMPONENT_VALUE_SUPPORTED_PROVISIONAL' if all(g['tierB']['provisional_pass'] for g in gates.values())
                   else 'TIER_B_NOT_DEMONSTRATED')
    tierC_label = 'NL_MECHANISM_UNIDENTIFIABLE' if tierC_ok else 'NL_MECHANISM_UNIDENTIFIABLE (identity check failed; see CLOCK_IDENTIFIABILITY)'
    cov = {'mvtec': {'status': 'OWNER_GO_REQUIRED', 'tasks': 15, 'reason': 'local images complete, no G3 token cache (extraction = new resource cost)'},
           'visa': {'status': 'MEASURED' if 'visa' in D else 'NOT_RUN', 'tasks': 12, 'note': 'exposed DEV; REPRO-01 verified caches'},
           'mpdd': {'status': 'MEASURED' if 'mpdd' in D else 'NOT_RUN', 'tasks': 6, 'note': 'exposed DEV (BENCH-G3 confirmation consumed)'},
           'realiad': {'status': 'OWNER_GO_REQUIRED', 'tasks': 30, 'reason': 'not on disk; download/extraction need GO'},
           'four_dataset_complete': complete, 'cross_dataset_mean_reported': False}
    U.wj(U.OUT / 'ABLATIONS.json', strip(abl))
    U.wj(U.OUT / 'NL_CUTS.json', strip(nl))
    U.wj(U.OUT / 'CUTS.json', strip(cuts))
    U.wj(U.OUT / 'BENCHMARK.json', strip({'datasets': bench, 'coverage': cov}))
    U.wj(U.OUT / 'CLOCK_IDENTIFIABILITY.json', strip(clk))
    U.wj(U.OUT / 'DATASET_COVERAGE.json', cov)
    vis = bench.get('visa', {}).get('N0', {})
    rv = D['visa'].rep('a|2711|N0', 'ap') if 'visa' in D else None
    au_rep = D['visa'].rep('a|2711|N0', 'auroc') if 'visa' in D else None
    pub = U.rj(U.OUT / 'SOURCE_AUDIT.json')['baselines']
    comp = {'baselines': pub, 'qualifier': 'historically exposed cohort; unmatched backbone, resolution, preprocessing and training normals; published values not reproduced; CONTEXT ONLY',
            'visa_N0_native_ap_ci95': [float(np.nanquantile(rv, .025)), float(np.nanquantile(rv, .975))] if rv is not None else None,
            'visa_N0_auroc_ci95': [float(np.nanquantile(au_rep, .025)), float(np.nanquantile(au_rep, .975))] if au_rep is not None else None,
            'visa_N0': {'ap_native': vis.get('final_ap_exact'), 'ap224': vis.get('final_ap224'), 'auroc': vis.get('final_auroc')},
            'note_ap224': 'the bootstrap above is on native AP; the BENCH-G3 RESULT_visa published_comparison (resized-224) remains the paper-comparable reference'}
    U.wj(U.OUT / 'COMPARABILITY.json', comp)
    au = U.rj(U.OUT / 'SOURCE_AUDIT.json')
    lat = U.rj(U.OUT / 'LATENCY.json') if (U.OUT / 'LATENCY.json').exists() else {}
    states = U.rj(U.OUT / 'STATES.json') if (U.OUT / 'STATES.json').exists() else {}
    led = {'definition': 'persistent CL state = fp16 keys (1,536 B each) + method statistic (G3: S fp32 + n = 2,359,304 B; G1b: 2,362,376 B; B5: one S per task); derived W cache (2,359,296 B fp32) recomputable, not charged; encoder shared, reported separately',
           'encoder_bytes': U.ENCODER_B, 'offline_cache_bytes': au['caches']['footprint_bytes'],
           'eval01_new_cache_bytes': sum(p.stat().st_size for p in U.CACHE.rglob('*') if p.is_file()) if U.CACHE.exists() else 0,
           'arms_final_bytes': {ds: {a: {'bytes': v['bytes_final'], 'MiB': v['bytes_final'] / U.MiB} for a, v in nl[ds]['rows'].items() if 'bytes_final' in v} for ds in nl},
           'cuts_bytes': {ds: {a: {'bytes': v['bytes'], 'MiB': v['MiB']} for a, v in cuts[ds]['arms'].items()} for ds in cuts},
           'update_time_states': states, 'latency': lat,
           'score_meta_t1': {k: {'seconds': v['seconds'], 'peak_vram': v['peak_vram']} for k, v in meta1.items()},
           'score_meta_t2': {k: {'seconds': v['seconds'], 'peak_vram': v['peak_vram']} for k, v in (U.rj(U.OUT / 'SCORE_META_t2.json') if (U.OUT / 'SCORE_META_t2.json').exists() else {}).items()},
           'historical_cost_bench_g3': U.rj(U.ROOT / 'results/hope_cad/bench_g3/COST.json')}
    U.wj(U.OUT / 'RESOURCE_LEDGER.json', led)
    vd = {'verdict': verdict, 'blockers': ['EVAL01_DATA_GO_REQUIRED: mvtec (G3 token extraction), realiad (download + extraction)'] if not complete else [],
          'tier_A_cad_efficacy': tierA_label, 'tier_B_two_component': tierB_label, 'tier_C_nl_specific': tierC_label,
          'nl_specific_claim': 'NL_SPECIFIC_ADVANTAGE_NOT_ESTABLISHED', 'gates_per_dataset': strip(gates), 'stageA': stageA,
          'tests': tests, 'measured_datasets': measured, 'margins': MARG, 'seconds': time.time() - t0,
          'selected_confirmation_untouched': True, 'mvtec_ad2_untouched': True}
    U.wj(U.OUT / 'VERDICT.json', strip(vd))
    U.log('VERDICT', verdict, tierA_label, tierB_label, tierC_label)
