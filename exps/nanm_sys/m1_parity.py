"""NANM-SYS-01 M1: Tier-A parity of the independent nanm package vs the UNCHANGED oracle, same physical GPU.

Scope: MPDD bracket_black -> bracket_brown, seed 2711, normal-only (FIT for writing, disjoint EVAL train/good images as
queries). No test image, label, mask or metric. Oracle imports live only in this harness.

    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_sys.m1_parity encoder     # under R.configure(), as the oracle extraction
    python -B -m exps.nanm_sys.m1_parity system      # under the oracle normal/score settings
"""
from __future__ import annotations

import sys
import time

import torch

from exps.nanm_sys import regen as RG

DS, TASKS, SEED = 'mpdd', ('bracket_black', 'bracket_brown'), 2711


def eq(a, b):
    return bool(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))


def encoder(dev):
    from exps.nar_pilot_00 import full as R
    from nanm import FrozenPatchEncoder
    G, out = RG.setup()
    R.configure()                                            # identical numerics to the oracle extraction
    B14 = G.B14()
    enc = FrozenPatchEncoder.load(B14.ENC_PATH, B14.ENC_SHA, 'cuda')
    res, t0 = {}, time.monotonic()
    for c in TASKS:
        rows = G.train_rows(DS, c)                           # all train/good rows in registered order, batch 8
        tok = enc.tokens([G.MPDD / r['relative_path'] for r in rows], batch=8)
        ref = G.load_train(DS, c)['features']                # oracle tokens (G._extract, same physical GPU, R1)
        res[c] = {'images': len(rows), 'tokens_bitwise_equal': eq(tok, ref.clone()),
                  'max_abs_diff': float((tok - ref).abs().max())}
    res['seconds'] = time.monotonic() - t0
    G.A.wj(out / 'm1' / 'ENCODER.json', res)
    RG.stamp(out / 'm1', 'encoder', {'result': res, 'device': dev})
    print(res)


def system(dev):
    from nanm import G3Config, NANMSystem
    RG.numerics()
    G, out = RG.setup()
    rec = G.A.rj(out / 'r1_runA' / f'normal_{DS}.json')
    nm = NANMSystem.create(G3Config(), 'cuda')
    ords = G.order(DS)
    t0 = time.monotonic()
    S_list, n_list, keys_o, checks = [], [], [], {}
    data = {c: G.load_train(DS, c) for c in TASKS}
    roles = {c: G.roles(DS, data[c]['rows']) for c in TASKS}
    for t, c in enumerate(TASKS):
        assert ords[t] == c
        k = rec['sizes']['g3'][t]
        # ---- unchanged oracle path (functions of the sealed sources)
        fit16_o = G.fit_tokens(DS, c)
        kc_o = G.A.greedy_coreset(fit16_o, k, seed=SEED)
        S1_o, n1_o = G.B.residual_stats(fit16_o.float(), kc_o.cuda())
        keys_o.append(fit16_o[kc_o.cuda()].cpu())
        S_list.append(S1_o); n_list.append(n1_o)
        W64_o, cond_o = G.B.whiten(sum(S_list), sum(n_list))
        W_o = W64_o.float()
        # ---- independent nanm path
        tok_fit = data[c]['features'][torch.as_tensor(G.fit_index(DS, data[c]['rows']))]
        rep = nm.fit_task(tok_fit, task_label=c)
        fit16_n = nm.writer.prepare(tok_fit)
        st = nm.state
        runA = torch.load(RG.CACHE / 'normal_runA' / f'{DS}_s{SEED}_{c}.pt', weights_only=True)
        stageA = torch.load(RG.CACHE / 'normal_runA' / f'{DS}_s{SEED}_stage.pt', weights_only=True)[t]['G3']
        ck = {'fit_tokens_fp16': eq(fit16_n, fit16_o), 'kcentre_indices': eq(st.selected[-1], kc_o),
              'keys_union_order': eq(st.union_keys(), torch.cat(keys_o)),
              'S_cumulative_fp64': eq(st.S, sum(S_list)), 'n_cumulative': st.n == sum(n_list),
              'W_fp64': eq(nm._cache[0], W64_o), 'W_fp32': eq(nm.W, W_o), 'cond': rep['cond_T'] == cond_o,
              'k': rep['k'] == k, 'n_res_task': rep['n_res'] == n1_o,
              'oracle_eq_normal_runA': {'kc_prefix': eq(kc_o, runA['kc'][:k]), 'S1': eq(S1_o, runA['S1']),
                                        'n1': n1_o == runA['n1'], 'stage_W': eq(W_o, stageA)}}
        # ---- normal-only EVAL queries of both tasks at this stage (task-free read)
        for qc in TASKS:
            ev = torch.as_tensor(roles[qc]['EVAL'])
            tok_q = data[qc]['features'][ev]
            q16 = G.A.unit(tok_q.reshape(-1, G.D)).half()                         # as G.load_test (CPU)
            e_o = G.whitened(q16, torch.cat(keys_o).cuda().float(), W_o.cuda())
            grid_o = e_o.reshape(len(ev), 28, 28)
            maps_o = G.R2.maps_of(grid_o)
            img_o = torch.from_numpy(G.A.image_scores(maps_o))
            p = nm.predict(tok_q)
            ck[f'eval_{qc}'] = {'images': len(ev), 'energies': eq(p['energy'], grid_o), 'maps392': eq(p['map392'], maps_o),
                                'image_scores': eq(p['image_score'], img_o),
                                'max_abs_energy_diff': float((p['energy'] - grid_o).abs().max())}
        checks[f'stage{t}_{c}'] = ck
        del fit16_o, fit16_n
    flat = []
    for ck in checks.values():
        for kk, v in ck.items():
            if kk == 'oracle_eq_normal_runA':
                continue
            flat += [x for x in (v.values() if isinstance(v, dict) else [v]) if isinstance(x, bool)]
    led = nm.state.ledger()
    res = {'checks': checks, 'tier_a_all_bitwise': all(flat), 'n_checks': len(flat), 'ledger_2task': led,
           'ledger_expected_registered': 1536 * sum(rec['sizes']['g3'][:2]) + 768 * 768 * 4 + 8,
           'contract_sha256': G3Config().contract_sha256(), 'seconds': time.monotonic() - t0}
    G.A.wj(out / 'm1' / 'SYSTEM.json', res)
    RG.stamp(out / 'm1', 'system', {'tier_a_all_bitwise': res['tier_a_all_bitwise'], 'device': dev})
    print({k: v for k, v in res.items() if k != 'checks'})


if __name__ == '__main__':
    d = RG.guard_env(True)
    {'encoder': encoder, 'system': system}[sys.argv[1]](d)
