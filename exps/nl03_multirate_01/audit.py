"""NL03-MULTIRATE-01 Phase A: source and data provenance audit (§0, §2, §4). CPU only; reads no test token.

Writes results/hope_cad/nl03_multirate_01/AUDIT.json:
  - canonical G3 / BENCH-G3 source hashes re-verified against NL03-PRESSURE-00's PREREG_LOCK, and all of that study's
    locks verified (its artifacts are immutable inputs here);
  - per task: train/good identities, the canonical role partition (FIT / CREDIT / EVAL, salted SHA, n = N // 5),
    the NL-02 SELECTED-pool exclusion (paths only), and the development splits FIT_dev / CALIB / EVAL;
  - train-feature cache provenance (sha256 of the tensor file == its JSON record; encoder sha; row identity);
  - the §4 data-availability gate.

    python -B -m exps.nl03_multirate_01.audit
"""
from __future__ import annotations

import json
import time

import torch

from exps.nanm_eval_01 import common as U
from exps.nl03_multirate_01 import spec as SP
from exps.nl03_pressure_00 import core as PC


def canonical_sources():
    lk = U.rj(PC.OUT / 'PREREG_LOCK.json')
    out, bad = {}, []
    for f, h in lk['canonical_sources'].items():
        now = U.sha(U.ROOT / f)
        out[f] = now
        if now != h:
            bad.append(f)
    return out, bad


def pressure_locks():
    from exps.nl03_pressure_00 import amend_a3 as A3
    from exps.nl03_pressure_00 import run as R
    A3.bind()
    R.verify_lock()
    return {p.name: U.sha(p) for p in sorted(PC.OUT.glob('*LOCK.json'))}


def task_audit(G, task, hash_cache):
    ds, c = task.split('/')
    PC.use(G, ds)
    rows = G.train_rows(ds, c)
    paths = [r['relative_path'] for r in rows]
    sp = SP.splits(G, task)
    rec = {'n_train_good': len(paths), 'roles': {k: len(v) for k, v in G.roles(ds, rows).items()},
           'nl02_excluded': len(PC.nl02_excluded(G, task)),
           'FIT_dev': len(sp['FIT']), 'CALIB': len(sp['CALIB']), 'EVAL': len(sp['EVAL']),
           'disjoint': not (set(sp['FIT']) & set(sp['CALIB']) or set(sp['FIT']) & set(sp['EVAL'])
                            or set(sp['CALIB']) & set(sp['EVAL'])),
           'selected_free': not ((set(sp['FIT']) | set(sp['CALIB']) | set(sp['EVAL'])) & PC.nl02_excluded(G, task)),
           'split_sha': U.sha_text(json.dumps(sp, sort_keys=True)) if hasattr(U, 'sha_text') else
           PC.ssha(json.dumps(sp, sort_keys=True))}
    img_root = SP.image_root(ds)
    rec['eval_images_exist'] = all((img_root / p).is_file() for p in sp['EVAL'] + sp['CALIB'])
    if hash_cache:
        t0 = time.time()
        data = G.load_train(ds, c)                 # oracle loader: sha256(file) == JSON record, rows == identities
        f = data['features']
        rec['cache'] = {'shape': list(f.shape), 'dtype': str(f.dtype), 'rows_match': [r['relative_path'] for r in
                        data['rows']] == paths, 'sha_verified_by_oracle_loader': True, 'seconds': time.time() - t0}
        del data
    return rec


def main():
    G = PC.bind()
    src, bad = canonical_sources()
    locks = pressure_locks()
    tasks = {}
    for ds in ('visa', 'mvtec', 'mpdd'):
        for t in PC.base_order(ds if ds != 'mvtec' else 's33'):
            if t.startswith(ds + '/'):
                tasks[t] = task_audit(G, t, hash_cache=True)
                U.log('AUDIT', t, {k: v for k, v in tasks[t].items() if k != 'cache'})
    gate = SP.data_gate(tasks)
    enc = {}
    for ds in ('visa', 'mvtec', 'mpdd'):
        PC.use(G, ds)
    out = {'canonical_sources_now': src, 'canonical_sources_changed': bad, 'pressure00_locks': locks,
           'tasks': tasks, 'gate_4': gate, 'split_rule': SP.SPLIT_RULE, 'time': time.strftime('%FT%T'),
           'note': 'no test token, mask or label was read; train features loaded via the hash-checking oracle loader'}
    SP.OUT.mkdir(parents=True, exist_ok=True)
    U.wj(SP.OUT / 'AUDIT.json', out)
    U.log('AUDIT DONE gate', gate['pass'], 'changed', bad)


if __name__ == '__main__':
    main()
