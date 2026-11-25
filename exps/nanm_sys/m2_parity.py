"""NANM-SYS-01 M2: full normal-only Tier-A parity, nanm vs the unchanged oracle (registration: task spec §18).

    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_sys.m2_parity
No test image, label or mask. First failing check stops the run.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import subprocess
import sys
import time

import torch

from exps.nanm_sys import regen as RG

SEEDS = (2711, 2712, 2713)
N_EVAL = 16
GPU2_BUS = 0x41


class Mismatch(RuntimeError):
    pass


def guard():
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise SystemExit('M2 runs on physical GPU 2 only (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2)')
    p = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or p.pci_bus_id != GPU2_BUS:
        raise SystemExit(f'refusing device bus {p.pci_bus_id:#x}')
    free, total = torch.cuda.mem_get_info(0)
    smi = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,used_memory', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout
    return {'physical_gpu': 2, 'pci_bus_id': p.pci_bus_id, 'name': p.name, 'free_gib_at_start': free / 2**30,
            'total_gib': total / 2**30, 'competing_processes_gpu2': [l for l in smi.splitlines() if '41:00.0' in l]}


@contextlib.contextmanager
def forbid_oracle(G):
    """Replace the oracle's scientific functions with raising stubs while nanm computes (anti-delegation proof)."""
    targets = [(G.A, n) for n in ('unit', 'greedy_coreset', 'image_scores')] + \
              [(G.R2, n) for n in ('sqdist', 'shrink', 'inv_sqrt_psd', 'maps_of', 'smooth_nc')] + \
              [(G.B, n) for n in ('residual_stats', 'whiten')] + [(G, n) for n in ('whitened', 'euclid', 'fit_tokens')]
    saved = [(m, n, getattr(m, n)) for m, n in targets]

    def stub(name):
        def f(*a, **k):
            raise RuntimeError(f'oracle function {name} called inside nanm')
        return f
    for m, n, _ in saved:
        setattr(m, n, stub(n))
    try:
        yield
    finally:
        for m, n, f in saved:
            setattr(m, n, f)


def eq(a, b):
    return bool(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))


def check(log, where, name, ok):
    log.append((where, name, bool(ok)))
    if not ok:
        raise Mismatch(f'{where}: {name}')


def run_config(G, ds, seed, arm, rec, ev_tok, log, cfg):
    """One (dataset, seed, arm) continual run, nanm vs oracle at every boundary."""
    from nanm import NANMSystem
    from nanm import geometry as NG
    sz, ords = rec[ds]['sizes'], G.order(ds)
    kkey, Skey, nkey, Wkey = ('g3', 'S1', 'n1', 'G3') if arm == 'G3' else ('g3c', 'SC', 'nC', 'G3@C')
    stage = G.load_stage(ds, seed, rec)
    conds = rec[ds]['seeds'][str(seed)]['cond']
    nm = NANMSystem.create(dataclasses.replace(cfg, seed=seed), 'cuda')   # per-seed contract (attempt-1 fix)
    S_o, n_o, keys_o = [], [], []
    for t, c in enumerate(ords):
        w = f'{ds}|s{seed}|{arm}|stage{t}:{c}'
        st = G.load_norm(ds, seed, c, rec)                                  # oracle state, sha-verified
        data = G.load_train(ds, c)
        tok = data['features'][torch.as_tensor(G.fit_index(ds, data['rows']))]
        k = sz[kkey][t]
        fit16_o = G.fit_tokens(ds, c)
        with forbid_oracle(G):
            fit16_n = nm.writer.prepare(tok)
            rep = nm.fit_task(tok, task_label=c, k=None if arm == 'G3' else k)
            idx_n = nm.state.selected[-1]
            St_n, nt_n = NG.residual_moment(fit16_n.float(), idx_n.cuda(), cfg.residual_chunk)
            T_n = NG.shrink(nm.state.S / nm.state.n, cfg.shrink_alpha)
        S_o.append(st[Skey]); n_o.append(st[nkey]); keys_o.append(st['keys'][:k])
        W64_o, cond_o = G.B.whiten(sum(S_o), sum(n_o))
        T_o = G.R2.shrink(sum(S_o) / sum(n_o), G.B.ALPHA)
        offs_n = [0]
        for kt in nm.state.keys:
            offs_n.append(offs_n[-1] + len(kt))
        offs_o = [0]
        for kt in keys_o:
            offs_o.append(offs_o[-1] + len(kt))
        check(log, w, 'fit_tokens_fp16', eq(fit16_n, fit16_o))
        check(log, w, 'quota_k', rep['k'] == k == len(idx_n))
        check(log, w, 'kcentre_indices', eq(idx_n, st['kc'][:k]))
        check(log, w, 'task_keys', eq(nm.state.keys[-1], st['keys'][:k]))
        check(log, w, 'union_bank', eq(nm.state.union_keys(), torch.cat(keys_o)))
        check(log, w, 'key_offsets', offs_n == offs_o)
        check(log, w, 'task_residual_moment', eq(St_n, st[Skey]))
        check(log, w, 'task_residual_count', nt_n == rep['n_res'] == st[nkey])
        check(log, w, 'S_cumulative_fp64', eq(nm.state.S, sum(S_o)))
        check(log, w, 'n_cumulative', nm.state.n == sum(n_o))
        check(log, w, 'T_regularised', eq(T_n, T_o))
        check(log, w, 'W_fp64', eq(nm._cache[0], W64_o))
        check(log, w, 'W_fp32_vs_runA_stage', eq(nm.W, stage[t][Wkey]))
        check(log, w, 'cond_vs_oracle_and_json', rep['cond_T'] == cond_o == conds[str(t)][Wkey])
        union_o32, W_o = torch.cat(keys_o).cuda().float(), stage[t][Wkey].cuda()
        for qc in ords[:t + 1]:
            q = ev_tok[qc]
            q16 = G.A.unit(q.reshape(-1, G.D)).half()
            grid_o = G.whitened(q16, union_o32, W_o).reshape(len(q), 28, 28)
            maps_o = G.R2.maps_of(grid_o)
            img_o = torch.from_numpy(G.A.image_scores(maps_o))
            with forbid_oracle(G):
                p = nm.predict(q)
            check(log, w, f'eval[{qc}].energy_grid', eq(p['energy'], grid_o))
            check(log, w, f'eval[{qc}].map392', eq(p['map392'], maps_o))
            check(log, w, f'eval[{qc}].image_score', eq(p['image_score'], img_o))
        del fit16_o, fit16_n, tok, data
    led = nm.state.ledger()
    want = rec[ds]['bytes'][arm]
    check(log, f'{ds}|s{seed}|{arm}|final', 'ledger_registered_equals_sealed_bytes',
          led['historical_registered_ledger'] == want)
    return led


def independence():
    out = subprocess.run([sys.executable, '-B', '-c', 'import sys, nanm; import nanm.system, nanm.encoder; '
                          'print([m for m in sys.modules if m.split(".")[0] in ("exps", "models", "dataset")])'],
                         capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    return {'fresh_import_foreign_modules': out.stdout.strip(), 'stderr': out.stderr.strip()[-300:]}


def main():
    dev = guard()
    RG.numerics()
    G, out = RG.setup()
    from nanm import G3Config
    cfg = G3Config()
    o = out / 'm2'
    t_all, log, ledgers, status, first_fail = time.monotonic(), [], {}, 'M2_TIER_A_PASS', None
    timing = {}
    try:
        for ds in G.DSETS:
            rec = {ds: G.A.rj(out / 'r1_runA' / f'normal_{ds}.json')}
            ev_tok = {}
            for c in G.order(ds):
                data = G.load_train(ds, c)
                ev = torch.as_tensor(G.roles(ds, data['rows'])['EVAL'][:N_EVAL])
                ev_tok[c] = data['features'][ev].clone()
            arms = ('G3',) + (('G3@C',) if ds == 'visa' else ())
            for arm in arms:
                for seed in SEEDS:
                    t0 = time.monotonic()
                    ledgers[f'{ds}|s{seed}|{arm}'] = run_config(G, ds, seed, arm, rec, ev_tok, log, cfg)
                    timing[f'{ds}|s{seed}|{arm}'] = time.monotonic() - t0
                    G.A.log('M2 PASS', ds, seed, arm, round(timing[f'{ds}|s{seed}|{arm}'], 1),
                            'checks so far', len(log))
    except Mismatch as e:
        status, first_fail = 'M2_TIER_A_FAIL', str(e)
    except torch.OutOfMemoryError as e:
        status, first_fail = 'RESOURCE_BLOCKED', f'CUDA OOM: {e}'[:2000]
    res = {'verdict': status, 'first_failure': first_fail, 'n_checks': len(log),
           'n_passed': sum(ok for *_, ok in log), 'checks': log, 'ledgers': ledgers, 'timing_s': timing,
           'seconds': time.monotonic() - t_all, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
           'device': dev, 'independence': independence(), 'contract_sha256': cfg.contract_sha256(),
           'nanm_sources': {p.name: G.A.sha(p) for p in sorted((G.ROOT / 'nanm').glob('*.py'))}}
    G.A.wj(o / 'M2_RESULT.json', res)
    RG.stamp(o, 'm2', {'verdict': status, 'device': dev})
    print(json.dumps({k: v for k, v in res.items() if k not in ('checks',)}, default=str)[:3000])


if __name__ == '__main__':
    main()
