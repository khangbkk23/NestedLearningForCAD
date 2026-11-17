"""CPU-only NAR handoff/resource preflight. No feature, checkpoint or mask loading.

This is not the real-feature pilot runner. Prerequisites currently block that runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import torch
import torch.nn as nn

from exps.dinomaly import official as O
from exps.nar_pilot_00 import core as K
from exps.vitb8_q2.common import SingleStreamDecoder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_audit(root):
    out = root / 'results/hope_cad/norm_00'
    lock = out / 'LOCK.json'
    lk = json.loads(lock.read_text())
    checks = {p: sha(root/p) == h for p, h in lk['sources'].items()}
    per = {}
    elapsed = None
    for line in (out/'logs/evaluate.out').read_text().splitlines():
        category, sep, rest = line.partition(' ')
        if sep and rest.startswith('{'):
            row, n = json.JSONDecoder().raw_decode(rest)
            K.require(category not in per, 'duplicate completed category in log')
            per[category] = row
            elapsed = rest[n:].strip()
    result = json.loads((out/'RESULT.json').read_text()) if (out/'RESULT.json').exists() else None
    if result is not None:
        from exps.norm_00.run import ORDER, WEAK, verdict
        from exps.nl_sgm_01a.am03_eval import bootstrap_delta
        import numpy as np
        K.require(all(checks.values()), 'NORM locked source mismatch')
        K.require(set(result['per_category']) == set(ORDER) == set(per), 'NORM incomplete category set')
        K.require(result['per_category'] == per, 'NORM stdout/result mismatch')
        K.require(set(result['parity_rel']) == set(ORDER) and max(result['parity_rel'].values()) < 1e-4,
                  'NORM parity failure')
        for metric in ('pixel_ap_native', 'pixel_ap_224', 'image_auroc'):
            a = [per[c]['INC'][metric] for c in ORDER]
            b = [per[c]['IMN'][metric] for c in ORDER]
            K.require(result['macro_'+metric] == {'IMN':float(np.mean(b)), 'INC':float(np.mean(a))}, 'NORM macro')
            K.require(bootstrap_delta(a, b, seed=lk['boot']['seed'], B=lk['boot']['B']) ==
                      result['INC_minus_IMN_'+metric], 'NORM paired bootstrap')
        weak = {c: per[c]['INC']['image_auroc']-per[c]['IMN']['image_auroc'] for c in WEAK}
        K.require(verdict(result['INC_minus_IMN_pixel_ap_native']['point'], weak) == result['verdict'], 'NORM rule')
    return {'lock_sha256': sha(lock), 'source_matches': checks,
            'result_exists': (out/'RESULT.json').exists(), 'per_category_from_stdout': per,
            'category_count': len(per), 'last_cumulative_wall': elapsed,
            'log_sha256': sha(out/'logs/evaluate.out'),
            'missing_parity': result is None, 'resume_implemented': False,
            'result_sha256': sha(out/'RESULT.json') if result else None,
            'verified_final_verdict': result['verdict'] if result else None}


def cpu_resource():
    torch.set_num_threads(2)
    torch.manual_seed(2711)
    model = SingleStreamDecoder(checkpoint=True).cpu()
    O.init_trainable(model)
    base_bytes = sum(x.numel()*x.element_size() for x in model.state_dict().values())
    names = tuple(n for n, m in model.named_modules() if isinstance(m, nn.Linear))
    loras = K.attach(model, names, rank=8)
    phi = K.init_state(loras, 2711)
    K.set_state(loras, phi)
    # Synthetic full-size grid. No dataset or historical learned checkpoint is read.
    x = torch.randn(1, 784, 768)
    y = torch.randn(1, 784, 768)
    cfg = K.Cfg(steps=1, warmup=1, m=1)
    def loss(v):
        en, de = model(v)
        return O.global_cosine(en, de)
    opt = lambda groups: O.StableAdamW(groups, lr=cfg.lr, betas=cfg.betas,
                                      weight_decay=cfg.weight_decay, eps=cfg.eps, amsgrad=True)
    model.train()
    start = time.perf_counter()
    lam, log = K.acquire(loras, lambda i: loss(x), lambda: loss(y), cfg,
                         torch.zeros(len(names), dtype=torch.float64), True, opt)
    elapsed = time.perf_counter()-start
    a_bytes = K.tensor_state_bytes(K.get_state(loras))
    return {'device': 'cpu', 'threads': torch.get_num_threads(), 'torch': torch.__version__,
            'platform': platform.platform(), 'cuda_available': torch.cuda.is_available(),
            'input': [1, 784, 768], 'rank': 8, 'sites': names, 'site_count': len(names),
            'base_source': 'deterministic initialization; shape/cost probe only',
            'base_bytes': base_bytes, 'adapter_bytes': a_bytes,
            'slow_phi_bytes': a_bytes, 'slow_lam_bytes': lam.numel()*lam.element_size(),
            'detector_tensor_bytes_T3': base_bytes+3*a_bytes+a_bytes+lam.numel()*lam.element_size(),
            'detector_tensor_bytes_T15': base_bytes+15*a_bytes+a_bytes+lam.numel()*lam.element_size(),
            'active_adapter_bytes': a_bytes, 'optimizer_moment_bytes_estimate': 3*a_bytes,
            'window_snapshot_bytes': a_bytes,
            'one_fit_plus_one_credit_s': elapsed,
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'fit': log['fit'], 'credit': log['credit'], 'lam_max_abs': float(lam.abs().max()),
            'gpu_runtime_estimate_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    K.require(not args.output.exists(), 'preflight output exists; preserve it')
    root = Path(__file__).resolve().parents[2]
    result = {'norm': norm_audit(root), 'cpu': cpu_resource(),
              'status': 'NAR_PILOT_BLOCKED',
              'blockers': ['INC normal-training features not prepared',
                           'GPU wall/VRAM preflight not measured; execution requires owner decision'],
              'pilot_executed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'cpu'}, indent=2))
    print(json.dumps({k: v for k, v in result['cpu'].items() if k != 'sites'}, indent=2))


if __name__ == '__main__':
    main()
