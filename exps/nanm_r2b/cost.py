"""Cost profile of the two-level memory (A0 Euclidean vs G3 residual geometry) — normal train/good images only.

    python -B -m exps.nanm_r2b.cost

End-to-end per image (batch 1): preprocess -> lite bf16 encoder -> retrieval -> A1 readout. Update cost per task from
cached FIT tokens. Writes results/hope_cad/nanm_r2b/COST.json. No test image or mask is opened.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.dinomaly import encoder as E
from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1
from exps.nanm_r2 import run as R2
from exps.nanm_r2b import run as B
from exps.nar_lite_01 import run as NL
from exps.nar_pilot_00 import core as K, full as R

OUT = B.OUT


def q(ts):
    ts = np.asarray(ts) * 1e3
    return {'p50_ms': float(np.median(ts)), 'p95_ms': float(np.quantile(ts, .95)), 'mean_ms': float(ts.mean())}


@torch.no_grad()
def memory(ratio, nr):
    mems = []
    for c in A.ORDER:
        fit = A.load_norm(c, nr)['fit']
        kc = torch.load(R1.CACHE / f'mv_{c}.pt', weights_only=True)['kc_idx']
        mems.append(fit[kc[:R1.n_of(ratio, len(fit))]])
    return torch.cat(mems)                                   # fp16, persistent form


@torch.no_grad()
def inference(nr):
    enc = NL.lite_encoder(torch.bfloat16)
    rows = R.rows_for('bottle', 'train')
    idx = K.normal_partition(rows)['EVAL'][:20]
    paths = [R.DATA / rows[i]['relative_path'] for i in idx]
    stage = torch.load(R2.CACHE / 'stage_metrics.pt', weights_only=True)
    rec = A.rj(OUT / 'STATS.json')
    res = {c: B.load_res(c, rec) for c in A.ORDER}
    W2 = B.whiten(sum(res[c]['R2pct'] for c in A.ORDER), sum(res[c]['n2pct'] for c in A.ORDER))[0].float()
    out = {}
    for ratio, W in ((0.01, stage[14]['G3']), (0.02, W2)):
        mem16 = memory(ratio, nr)
        for arm in ('A0', 'G3'):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            Wg = W.cuda()
            memc = mem16.cuda().float()
            if arm == 'G3':
                memc = memc @ Wg                              # cached at the task boundary (slow level)
            tp, te, tr, tt = [], [], [], []
            for k in range(23):
                p = paths[max(0, k - 3)]
                t0 = time.perf_counter(); x = E.preprocess(p)[None]; t1 = time.perf_counter()
                torch.cuda.synchronize(); t2 = time.perf_counter()
                z = A.unit(NL.tokens(enc, x.cuda().to(torch.bfloat16))[0].float())
                torch.cuda.synchronize(); t3 = time.perf_counter()
                if arm == 'G3':
                    z = z @ Wg
                e = 0.5 * R2.sqdist(z, memc).min(1).values.reshape(1, 28, 28)
                sm = NL.smooth392(e.cpu())
                sm.flatten(1).topk(max(1, int(NL.TOP * 392 * 392)), dim=1).values.mean(1)
                torch.cuda.synchronize(); t4 = time.perf_counter()
                if k >= 3:
                    tp.append(t1 - t0); te.append(t3 - t2); tr.append(t4 - t3); tt.append(t4 - t2 + t1 - t0)
            out[f'{arm}@{ratio:g}'] = {
                'preprocess': q(tp), 'encoder': q(te), 'retrieval_and_readout': q(tr), 'end_to_end': q(tt),
                'peak_cuda_bytes_incl_encoder': torch.cuda.max_memory_allocated(),
                'peak_cuda_bytes_above_encoder': torch.cuda.max_memory_allocated() - base,
                'memory_patches': len(mem16), 'memory_bytes_fp16': mem16.numel() * 2,
                'aux_bytes': R2.ACC_BYTES if arm == 'G3' else 0}
            A.log('INF', arm, ratio, round(out[f'{arm}@{ratio:g}']['end_to_end']['p50_ms'], 1))
            del memc, Wg
    out['encoder_persistent_bytes'] = sum(v.numel() * v.element_size() for v in enc.state_dict().values())
    return out


@torch.no_grad()
def update(nr):
    """Per-task slow+fast update cost from cached FIT tokens (feature extraction reported separately)."""
    out = {}
    acc = torch.zeros(768, 768, dtype=torch.float64, device='cuda'); n_acc = 0
    for c in A.ORDER:
        fit = A.load_norm(c, nr)['fit'].cuda().float()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        n1 = R1.n_of(0.01, len(fit))
        kc = A.greedy_coreset(fit, n1)                       # fast level: memory write (k-centre)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        S, cnt = B.residual_stats(fit, kc.cuda())            # fast -> slow: residual sufficient statistics
        acc += S.cuda(); n_acc += cnt
        torch.cuda.synchronize(); t2 = time.perf_counter()
        W, _ = B.whiten(acc.cpu(), n_acc)                    # slow level: geometry update at the boundary
        mw = fit[kc.cuda()] @ W.float().cuda()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        out[c] = {'n_fit_images': len(fit) // 784, 'kcentre_s': t1 - t0, 'residual_stats_s': t2 - t1,
                  'geometry_update_s': t3 - t2, 'total_s': t3 - t0}
        A.log('UPD', c, {k: round(v, 3) for k, v in out[c].items()})
        del fit, mw; torch.cuda.empty_cache()
    return out


def main():
    nr = A.rj(A.OUT / 'NORMAL.json')
    inf = inference(nr)
    upd = update(nr)
    tot = {k: sum(v[k] for v in upd.values()) for k in ('kcentre_s', 'residual_stats_s', 'geometry_update_s', 'total_s')}
    f = A.rj(NL.OUT / 'COST.json')
    A.wj(OUT / 'COST.json', {'inference': inf, 'update_per_task': upd, 'update_total_15_tasks': tot,
                             'F_reference_nar_lite_01': {'end_to_end': f['end_to_end'],
                                                         'total_persistent_bytes': f['total_persistent_bytes'],
                                                         'cuda_peak_bytes': f['cuda_peak_bytes'],
                                                         'train_seconds_15_tasks_nar_b14_01': 559},
                             'device': torch.cuda.get_device_name(0),
                             'note': 'normal train/good images only; retrieval brute-force fp32; whitened memory '
                                     'cached at the task boundary; feature extraction of FIT images excluded from '
                                     'update cost (encoder time per image is in inference.encoder)'})
    A.log('COST written')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
