"""NANM-EVAL-01 amendment A1 (MVTec AD, owner GO 2026-11-23): MVTec AD (15) under the locked EVAL-01 protocol.

Registration: agents/tasks/task_nanm_eval_01_amendment_a1_mvtec.md (registration §R10 executed).
Nothing in the locked EVAL-01 code is edited. This module
  1. binds the canonical BENCH-G3 oracle to an MVTec dataset key ('mvtec') whose layout is the MPDD/MVTec layout
     already handled by the oracle (CD._mvtec + K.normal_partition), with K: caches under $EVAL01_CACHE/mvtec;
  2. runs the oracle's own PREP phases for MVTec (encoder extraction, `_normal_ds`, canonical `score`), so the
     canonical G3 / A0b / G1b / E-full grids exist as the Stage-A parity reference (as REPRO-01 grids do for MPDD/VisA);
  3. drives the LOCKED exps.nanm_eval_01.run phases (states / score / metrics / latency) and summary unchanged, with
     U.OUT / U.CACHE / U.EXEC_DATASETS / U.bind rebound to the amendment locations.

    python -B -m exps.nanm_eval_01.mvtec <phase>
      lock | custody | extract | normal | canonical | states | score <t> | metrics <t> | gate | latency | summary | combine
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from exps.nanm_eval_01 import common as U

ROOT = U.ROOT
DS = 'mvtec'
MV_ROOT = ROOT / 'data/mvtec'
MV_OUT = U.OUT / 'mvtec_a1'                          # amendment outputs (E:, small JSON/logs)
MV_CACHE = U.CACHE / 'mvtec_a1'                      # K: (features, normal states, canonical grids, EVAL grids, boot)
G_OUT = MV_OUT / 'bench_g3'                          # oracle-format PREP records for MVTec
G_CACHE = MV_CACHE / 'bench_g3'
AMEND = ROOT / 'agents/tasks/task_nanm_eval_01_amendment_a1_mvtec.md'
CODE_A1 = ['exps/nanm_eval_01/mvtec.py', 'exps/nanm_eval_01/test_mvtec.py']
BUDGET_A1 = {'gpu_hours': 3.0, 'cpu_wall_hours': 10.0, 'k_new_bytes': 40 * U.GiB, 'rss_bytes': int(9.5 * U.GiB),
             'd_free_min': 30 * U.GiB}

_ORIG = {}
_ORIG_BIND = U.bind


def mv_bind():
    """REPRO-01 bind, then rebind the oracle's path constants and dataset dispatch for 'mvtec' (idempotent)."""
    G = _ORIG_BIND()
    from exps.calm02 import data as CD
    from exps.nar_pilot_00 import core as K
    from exps.nanm_r0 import audit as A
    G.OUT, G.CACHE = G_OUT, G_CACHE
    G.FEAT, G.NORM, G.SCORE = G_CACHE / 'features', G_CACHE / 'normal', G_CACHE / 'scores'
    G.DSETS = (DS,)
    if not _ORIG:
        _ORIG.update({n: getattr(G, n) for n in ('order', 'data_root', 'train_rows', 'test_rows', 'fit_index', 'roles',
                                                 'load_train', 'verify_lock')})

    def order(ds):
        return tuple(CD.MVTEC_ORDER) if ds == DS else _ORIG['order'](ds)

    def data_root(ds):
        return MV_ROOT if ds == DS else _ORIG['data_root'](ds)

    def train_rows(ds, c):
        return CD._mvtec(MV_ROOT, c)[0] if ds == DS else _ORIG['train_rows'](ds, c)

    def test_rows(ds, c):
        return CD._mvtec(MV_ROOT, c)[1] if ds == DS else _ORIG['test_rows'](ds, c)

    def fit_index(ds, rows):
        return K.normal_partition(rows)['FIT'] if ds == DS else _ORIG['fit_index'](ds, rows)

    def roles(ds, rows):
        return K.normal_partition(rows) if ds == DS else _ORIG['roles'](ds, rows)

    def load_train(ds, c):
        if ds != DS:
            return _ORIG['load_train'](ds, c)
        p = G.FEAT / f'{DS}_{c}_train.pt'
        K.require(A.sha(p) == A.rj(p.with_suffix('.json'))['sha256'], f'MVTec train cache changed: {c}')
        data = torch.load(p, weights_only=True, mmap=True)
        K.require([r['relative_path'] for r in data['rows']] == [r['relative_path'] for r in train_rows(ds, c)], 'rows')
        return data

    def verify_lock():                                # used only by the oracle `score` phase (canonical grids)
        return verify_a1()

    for n, f in (('order', order), ('data_root', data_root), ('train_rows', train_rows), ('test_rows', test_rows),
                 ('fit_index', fit_index), ('roles', roles), ('load_train', load_train), ('verify_lock', verify_lock)):
        setattr(G, n, f)
    return G


def rebind():
    """Point the locked EVAL-01 driver at the amendment locations (module attributes are read at call time)."""
    from exps.nanm_eval_01 import metrics as M
    U.bind = mv_bind
    U.OUT, U.CACHE, U.EXEC_DATASETS = MV_OUT, MV_CACHE, (DS,)
    global _ORIG_RC
    if M.run_category is not run_category:
        _ORIG_RC = M.run_category
        M.run_category = run_category


def run_category(args):
    """Spawn-pool worker: re-apply the amendment bind inside the fresh interpreter, then the locked worker body."""
    from exps.nanm_eval_01 import metrics as M
    U.bind = mv_bind
    U.OUT, U.CACHE, U.EXEC_DATASETS = MV_OUT, MV_CACHE, (DS,)
    f = _ORIG_RC or M.run_category
    return f(args)


_ORIG_RC = None


# ------------------------------------------------------------------ manifest / lock
def test_manifest():
    from exps.calm02 import data as CD
    return {c: [r['relative_path'] for r in CD._mvtec(MV_ROOT, c)[1]] for c in CD.MVTEC_ORDER}


def k_text(obj):
    return U.ssha(json.dumps(obj, sort_keys=True))


def lock():
    from exps.nanm_eval_01 import run as RN
    p = MV_OUT / 'PREREG_LOCK.json'
    assert not p.exists(), 'amendment lock exists'
    base = RN.verify_lock()                              # the original EVAL-01 lock must still verify
    tests = U.rj(MV_OUT / 'TESTS.json')
    assert tests['pass'], 'amendment tests failed'
    assert not (G_CACHE / 'features').exists() or not any((G_CACHE / 'features').glob('*_test.pt')), 'test tokens before lock'
    tm = test_manifest()
    rec = {'registration': {**base['registration'], str(AMEND.relative_to(ROOT)): U.sha(AMEND)},
           'task': base['task'],
           'code': {**base['code'], **{f: U.sha(ROOT / f) for f in CODE_A1}},
           'canonical_sources': base['canonical_sources'], 'repro_artifacts': base['repro_artifacts'],
           'base_lock_sha256': U.sha(U.ROOT / 'results/hope_cad/nanm_eval_01/PREREG_LOCK.json'),
           'test_manifest_sha256': k_text(tm), 'test_counts': {c: len(v) for c, v in tm.items()},
           'env': base['env'], 'budget': BUDGET_A1, 'storage': U.storage(), 'time': time.strftime('%FT%T'),
           'git_head': base['git_head'],
           'exposure': 'MVTec AD is consumed development (G3 was selected on MVTec in R2): all MVTec results are EXPOSED-DEV'}
    U.wj(p, rec, exclusive=True)
    U.log('LOCK A1', U.sha(p)[:16])


def verify_a1():
    lk = U.rj(MV_OUT / 'PREREG_LOCK.json')
    bad = [f for f, h in {**lk['registration'], **lk['task'], **lk['code'], **lk['canonical_sources']}.items()
           if h is not None and U.sha(ROOT / f) != h]
    bad += [n for n, h in lk['repro_artifacts'].items() if U.sha(U.REPRO / n) != h]
    assert not bad, f'LOCK_FAIL_A1: {bad}'
    assert k_text(test_manifest()) == lk['test_manifest_sha256'], 'MVTec test manifest changed'
    return lk


# ------------------------------------------------------------------ PREP (oracle functions, MVTec key)
def custody():
    verify_a1()
    p = G_OUT / 'MVTEC_MANIFEST.json'
    assert not p.exists(), 'custody exists'
    from exps.calm02 import data as CD
    from exps.nar_pilot_00 import core as K
    files = sorted(q for q in MV_ROOT.rglob('*') if q.is_file())
    man = {q.relative_to(MV_ROOT).as_posix(): U.sha(q) for q in files}
    counts, dup = {}, {}
    for c in CD.MVTEC_ORDER:
        tr, te = CD._mvtec(MV_ROOT, c)
        assert all(r['mask_path'] is None or r['mask_path'] in man for r in te), f'missing mask {c}'
        counts[c] = {'train_good': len(tr), 'test': len(te), 'test_good': sum(r['label'] == 0 for r in te),
                     'test_anomalous': sum(r['label'] for r in te), 'n_fit_images': len(K.normal_partition(tr)['FIT'])}
        trh = {man[r['relative_path']] for r in tr}
        dup[c] = sorted(r['relative_path'] for r in te if man[r['relative_path']] in trh)
    U.wj(p, {'root': str(MV_ROOT), 'license': 'CC BY-NC-SA 4.0 (data/mvtec/license.txt)', 'n_files': len(man),
             'total_bytes': sum(q.stat().st_size for q in files), 'counts': counts, 'train_test_exact_duplicates': dup,
             'files': man, 'note': 'test/ and ground_truth/ hashed as bytes only; no image or mask decoded'})
    U.log('CUSTODY', len(man), 'dups', sum(len(v) for v in dup.values()))


def extract():
    """Train and test tokens with the oracle encoder and `_extract` (fp32, images only; labels never read)."""
    verify_a1()
    rebind()
    G = mv_bind()
    man = U.rj(G_OUT / 'MVTEC_MANIFEST.json')['files']
    torch.backends.cuda.matmul.allow_tf32 = False
    enc = G.encoder()
    tm = test_manifest()
    rec, t0 = {}, time.monotonic()
    for c in G.order(DS):
        for split, rows in (('train', G.train_rows(DS, c)), ('test', [{'relative_path': r} for r in tm[c]])):
            p = G.FEAT / f'{DS}_{c}_{split}.pt'
            if not p.exists():
                assert all(man[r['relative_path']] == U.sha(MV_ROOT / r['relative_path']) for r in rows), 'custody'
                s = G._extract(enc, MV_ROOT, rows, p)
                U.log('EXTRACT', split, c, len(rows), '%.1fs' % s)
            rec[p.name] = U.rj(p.with_suffix('.json'))['sha256']
    U.wj(G_OUT / 'EXTRACT_COMPLETE.json', {'files': {k: v for k, v in rec.items() if k.endswith('_test.pt')},
                                           'train_files': {k: v for k, v in rec.items() if k.endswith('_train.pt')},
                                           'seconds': time.monotonic() - t0})


def normal():
    verify_a1()
    rebind()
    G = mv_bind()
    torch.backends.cuda.matmul.allow_tf32 = False
    G.NORM.mkdir(parents=True, exist_ok=True)
    if not (G_OUT / f'normal_{DS}.json').exists():
        with torch.no_grad():
            G._normal_ds(DS)
    U.wj(G_OUT / 'NORMAL.json', {DS: U.rj(G_OUT / f'normal_{DS}.json')})


def canonical():
    """Oracle `score` phase for MVTec: G3 / A0b / G1b per seed and E-full grids (Stage-A parity reference)."""
    verify_a1()
    rebind()
    G = mv_bind()
    torch.backends.cuda.matmul.allow_tf32 = False
    G.score()


# ------------------------------------------------------------------ locked EVAL-01 phases on the amendment key
def tests_record():
    """Write TESTS.json from the pytest log produced by pipeline_mvtec.sh (pass = exit code 0)."""
    log_p = MV_OUT / 'logs/tests.log'
    txt = log_p.read_text()
    rc = int(sys.argv[2])
    U.wj(MV_OUT / 'TESTS.json', {'command': 'python -B -m pytest -q --capture=sys exps/nanm_eval_01/test_mvtec.py',
                                 'exit_code': rc, 'pass': rc == 0, 'tail': txt.strip().splitlines()[-3:],
                                 'base_tests': U.rj(U.ROOT / 'results/hope_cad/nanm_eval_01/TESTS.json')})


def audit_copy():
    """Amendment SOURCE_AUDIT: base audit (baselines, encoder, canonical sources) + MVTec custody/cache footprint."""
    base = U.rj(U.ROOT / 'results/hope_cad/nanm_eval_01/SOURCE_AUDIT.json')
    feat = G_CACHE / 'features'
    fp = sum(q.stat().st_size for q in feat.glob('*.pt')) if feat.exists() else 0
    base['caches'] = {**base.get('caches', {}), 'footprint_bytes': fp, 'mvtec_feature_bytes': fp}
    base['amendment'] = 'A1 MVTec AD'
    U.wj(MV_OUT / 'SOURCE_AUDIT.json', base)


def gate():
    from exps.nanm_eval_01 import summary as S
    patch_summary(S)
    G = mv_bind()
    d = S.DS(G, DS)
    meta = U.rj(MV_OUT / 'SCORE_META_t1.json')
    rec = {DS: {'grid_parity': {k: v['parity_all_equal'] for k, v in meta.items() if k.startswith(DS + '|a|')},
                'metric_parity': S.metric_parity(d), 'binned_validation': S.binned_validation(d)}}
    ok = all(rec[DS]['grid_parity'].values()) and len(rec[DS]['grid_parity']) == 3 and rec[DS]['binned_validation']['pass']
    U.wj(MV_OUT / 'GATE_T1.json', {**rec, 'pass': ok})
    print('GATE_T1_A1', ok)
    sys.exit(0 if ok else 3)


def patch_summary(S):
    """MVTec has no historical CELLS (the server-era MVTec caches are lost): metric parity is replaced by the
    bitwise Stage-A grid parity against the oracle's canonical MVTec grids (registered in amendment A1 §3)."""
    def metric_parity(d):
        if d.ds != DS:
            return _ORIG_MP(d)
        meta = U.rj(MV_OUT / 'SCORE_META_t1.json')
        gp = {k: v['parity_all_equal'] for k, v in meta.items() if k.startswith(DS + '|a|')}
        return {'n_compared': 0, 'n_mismatch': 0, 'mismatches': [], 'replaced_by': 'stage-A bitwise grid parity',
                'grid_parity': gp, 'pass': len(gp) == 3 and all(gp.values())}
    global _ORIG_MP
    if getattr(S.metric_parity, '_a1', False):
        return
    _ORIG_MP = S.metric_parity
    metric_parity._a1 = True
    S.metric_parity = metric_parity


_ORIG_MP = None


def summary():
    from exps.nanm_eval_01 import summary as S
    rebind()
    audit_copy()
    patch_summary(S)
    S.main()


def combine():
    """Cross-dataset record (MPDD + VisA from the base run, MVTec from A1). No new statistic: per-dataset gates are
    copied; tier labels follow registration R8 with 3 of 4 datasets measured."""
    base = U.rj(U.ROOT / 'results/hope_cad/nanm_eval_01/VERDICT.json')
    a1 = U.rj(MV_OUT / 'VERDICT.json')
    gates = {**base['gates_per_dataset'], **a1['gates_per_dataset']}
    measured = sorted(set(base['measured_datasets']) | set(a1['measured_datasets']))
    mech = base['verdict'] != 'EVAL01_PROTOCOL_BLOCKED' and a1['verdict'] != 'EVAL01_PROTOCOL_BLOCKED'
    complete = len(measured) == 4
    verdict = ('EVAL01_PROTOCOL_BLOCKED' if not mech else 'EVAL01_ADDITIONAL_CONTROLS_REQUIRED' if not complete
               else 'EVAL01_READY_FOR_CONFIRMATION' if all(g['tierA']['provisional_pass'] for g in gates.values())
               else 'EVAL01_BASELINE_NOT_COMPETITIVE')
    rec = {'verdict': verdict, 'measured_datasets': measured,
           'tier_A_cad_efficacy': f'NOT_EVALUABLE_{len(measured)}_OF_4_DATASETS' if not complete else verdict,
           'tier_A_provisional_per_dataset': {ds: g['tierA']['provisional_pass'] for ds, g in gates.items()},
           'tier_B_two_component': ('NL_COMPONENT_VALUE_SUPPORTED_PROVISIONAL'
                                    if all(g['tierB']['provisional_pass'] for g in gates.values()) else 'TIER_B_NOT_DEMONSTRATED'),
           'tier_B_per_dataset': {ds: {k: g['tierB'][k] for k in ('B_i_pass', 'B_ii_pass', 'B_iii_pass', 'B_iv_pass', 'provisional_pass')}
                                  for ds, g in gates.items()},
           'tier_C_nl_specific': base['tier_C_nl_specific'] if 'UNIDENTIFIABLE' in a1['tier_C_nl_specific'] else a1['tier_C_nl_specific'],
           'nl_specific_claim': 'NL_SPECIFIC_ADVANTAGE_NOT_ESTABLISHED',
           'blockers': ['EVAL01_DATA_GO_REQUIRED: realiad (dataset access not granted / not on disk)'] if 'realiad' not in measured else [],
           'base_verdict_sha256': U.sha(U.ROOT / 'results/hope_cad/nanm_eval_01/VERDICT.json'),
           'a1_verdict_sha256': U.sha(MV_OUT / 'VERDICT.json'), 'gates_per_dataset': gates,
           'coverage': {'mpdd': 'MEASURED (exposed)', 'visa': 'MEASURED (exposed)', 'mvtec': 'MEASURED A1 (exposed: G3 selected on MVTec)',
                        'realiad': 'NOT_RUN (access gate)'}}
    U.wj(MV_OUT / 'COMBINED_VERDICT.json', rec)
    U.log('COMBINED', verdict, rec['tier_A_cad_efficacy'], rec['tier_B_two_component'], rec['tier_C_nl_specific'])


def run_phase(ph, *args):
    from exps.nanm_eval_01 import run as RN
    rebind()
    if ph in ('states', 'score', 'metrics', 'latency'):
        getattr(RN, ph)(*args)
    else:
        raise SystemExit(f'unknown phase {ph}')


if __name__ == '__main__':
    ph = sys.argv[1]
    table = {'lock': lock, 'custody': custody, 'extract': extract, 'normal': normal, 'canonical': canonical,
             'gate': gate, 'summary': summary, 'combine': combine, 'tests_record': tests_record}
    if ph in table:
        if ph in ('lock', 'tests_record'):
            table[ph]()
        else:
            rebind()
            table[ph]()
    else:
        run_phase(ph, *sys.argv[2:])
