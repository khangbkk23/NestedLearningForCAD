"""Speed and memory benchmark of RIGEL-CAD from raw images (no token cache), on the machine it runs on.

For one dataset (order a, seed 2711), the stream is learned task by task from image files. Each task's training
images are decoded, encoded once, and the same tokens are given to three learners:
  E  RIGEL-CAD at the budget (default 8 MiB; storage fp16 or int4_triangular)
  F  fast level only at the same budget (fp16)    U  unbounded (fp16)
Timed per task (CUDA-synchronised): image decode + resize (CPU), backbone forward (GPU), and each learner's
fit_task. After the last task, `n_infer` test images drawn evenly over the categories (seeded) are scored one at a
time, end to end (decode -> backbone -> patch energy -> 392 px anomaly map -> image score), with every stage timed,
plus the head alone at batch 1 for E / F / U. The head scores with the model's device-resident read-out
(RigelCAD.prepare: keys, W and keys @ W kept on the GPU, as deployed). Training images are decoded by
`decode_workers` threads per batch (inference decodes one image at a time). Peak GPU memory is recorded per phase,
and the final E state is saved to measure its size on disk. Non-default storage adds the '_int4tri' suffix to the
output names.

Scores are not computed here: accuracy comes from the stream and ad2 commands. Timings depend on the GPU, the
CPU and the disk; the report states the machine.

    python -m rigel_cad bench --dataset mvtec --data-root data/mvtec
"""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from rigel_cad import data as DATA
from rigel_cad import encoders as ENC
from rigel_cad import readout
from rigel_cad.config import TOKEN_DIM, RigelConfig
from rigel_cad.encoders import build_encoder
from rigel_cad.model import RigelCAD
from rigel_cad.pipeline import log, storage_suffix, write_json


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _tasks(dataset, root):
    """[(task name, train paths, test paths)] in stream order a."""
    root = Path(root)
    if dataset == 'ad2':
        return [(s, [root / r for r in DATA.ad2_files(root, s, 'train')],
                 [root / r for r in DATA.ad2_files(root, s, 'test_public')]) for s in DATA.ad2_scenarios(root)]
    out = []
    for c in DATA.stream_orders(dataset)['a']:
        tr, te = DATA.category_split(dataset, root, c)
        out.append((c, [root / s.relative_path for s in tr], [root / s.relative_path for s in te]))
    return out


def _pct(v):
    v = np.asarray(v, dtype=float) * 1e3
    return {'p50_ms': float(np.median(v)), 'mean_ms': float(v.mean()), 'p90_ms': float(np.percentile(v, 90)),
            'n': int(len(v))}


@torch.no_grad()
def run_bench(dataset: str, data_root, budget_mib: float = 8.0, seed: int = 2711, device: str = 'cuda',
              batch: int = 8, n_infer: int = 100, out_dir='results/rigel_cad/bench', storage: str = 'fp16',
              decode_workers: int = None) -> dict:
    out_dir = Path(out_dir)
    workers = ENC.DECODE_WORKERS if decode_workers is None else decode_workers
    tasks = _tasks(dataset, data_root)
    enc = build_encoder('dinov2_vitb14_reg4', device)
    cfg = dict(seed=seed, device=device)
    models = {'E': RigelCAD(RigelConfig(budget_mib=budget_mib, storage=storage, **cfg)),
              'F': RigelCAD(RigelConfig(budget_mib=budget_mib, slow_level=False, **cfg)),
              'U': RigelCAD(RigelConfig(budget_mib=None, **cfg))}
    # warm-up (kernels, allocator) on a few real training images; not timed
    x = torch.stack([enc.preprocess(p) for p in tasks[0][1][:batch]]).to(device)
    enc.tokens(x)
    _sync(device)
    per_task = []
    torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None
    t_all = time.time()
    for name, train, _ in tasks:
        t_dec = t_enc = 0.0
        toks = []
        for i in range(0, len(train), batch):
            t0 = time.time()
            x = enc.preprocess_many(train[i:i + batch], workers)
            t1 = time.time()
            x = x.to(device)
            toks.append(enc.tokens(x).cpu())                        # as TokenStore: tokens leave the GPU
            _sync(device)
            t_dec, t_enc = t_dec + (t1 - t0), t_enc + (time.time() - t1)
        # one device copy of the task's tokens, as run_stream does; free the encoder's cached blocks first, so a
        # 6 GB GPU does not spill into shared system memory on the largest VisA / AD 2 tasks
        torch.cuda.empty_cache() if device.startswith('cuda') else None
        x16 = torch.cat(toks).reshape(-1, TOKEN_DIM).to(device)
        del toks
        fit = {}
        for a, m in models.items():
            _sync(device)
            t0 = time.time()
            m.fit_task(x16, name)
            _sync(device)
            fit[a] = time.time() - t0
        row = {'task': name, 'n_train': len(train), 'decode_s': t_dec, 'encode_s': t_enc, 'fit_s': fit,
               'keys': {a: len(m.fast) for a, m in models.items()},
               'bytes': {a: m.persistent_bytes() for a, m in models.items()}}
        per_task.append(row)
        log('BENCH TRAIN', dataset, name, len(train), f'decode {t_dec:.1f}s encode {t_enc:.1f}s',
            {a: round(v, 2) for a, v in fit.items()})
        del x16
        torch.cuda.empty_cache() if device.startswith('cuda') else None
    train_wall = time.time() - t_all
    peak_train = torch.cuda.max_memory_allocated() if device.startswith('cuda') else None

    # ---------------------------------------------------------------- inference, batch 1, end to end
    g = np.random.default_rng(seed)
    per_cat = max(1, int(np.ceil(n_infer / len(tasks))))
    pick = [p for _, _, te in tasks for p in g.choice(te, size=min(per_cat, len(te)), replace=False).tolist()][:n_infer]
    states = {a: m.prepare() for a, m in models.items()}           # keys / W / keys @ W resident on the device
    q = enc.tokens(enc.preprocess(pick[0])[None].to(device))      # warm-up
    for a in models:
        models[a].patch_energy_grid(q, states[a])
    _sync(device)
    torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None
    tm = {k: [] for k in ('decode', 'encode', 'head_E', 'map_score', 'total_E', 'head_F', 'head_U')}
    for p in pick:
        t0 = time.time()
        x = enc.preprocess(p)[None]
        t1 = time.time()
        q = enc.tokens(x.to(device))
        _sync(device)
        t2 = time.time()
        gE = models['E'].patch_energy_grid(q, states['E'])
        _sync(device)
        t3 = time.time()
        mp = readout.anomaly_maps(gE)
        readout.image_scores(mp)
        t4 = time.time()
        for k, v in zip(('decode', 'encode', 'head_E', 'map_score', 'total_E'), (t1 - t0, t2 - t1, t3 - t2, t4 - t3,
                                                                                t4 - t0)):
            tm[k].append(v)
        for a in ('F', 'U'):
            _sync(device)
            t5 = time.time()
            models[a].patch_energy_grid(q, states[a])
            _sync(device)
            tm[f'head_{a}'].append(time.time() - t5)
    peak_inf = torch.cuda.max_memory_allocated() if device.startswith('cuda') else None
    sfx = storage_suffix(models['E'].config)
    sp = out_dir / 'states' / f'{dataset}_E{sfx}.rigel'
    models['E'].save(sp)

    n_tr = sum(r['n_train'] for r in per_task)
    tot = {k: sum(r[k] for r in per_task) for k in ('decode_s', 'encode_s')}
    fit_tot = {a: sum(r['fit_s'][a] for r in per_task) for a in models}
    res = {'dataset': dataset, 'order': [t[0] for t in tasks], 'seed': seed, 'budget_mib': budget_mib,
           'storage_E': storage, 'batch_train': batch, 'decode_workers_train': workers,
           'head_mode': 'prepared (RigelCAD.prepare)', 'machine': {
               'gpu': torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'cpu_count': os.cpu_count(),
               'platform': platform.platform(), 'python': platform.python_version()},
           'train': {'n_images': n_tr, 'wall_s': train_wall, **tot, 'fit_s': fit_tot,
                     'images_per_s_encode': n_tr / tot['encode_s'],
                     'images_per_s_decode_plus_encode': n_tr / (tot['decode_s'] + tot['encode_s']),
                     'fit_s_per_task_mean': {a: v / len(tasks) for a, v in fit_tot.items()},
                     'peak_gpu_bytes': peak_train, 'per_task': per_task},
           'infer_batch1': {'n_images': len(pick), **{k: _pct(v) for k, v in tm.items()},
                            'peak_gpu_bytes': peak_inf,
                            'keys': {a: len(m.fast) for a, m in models.items()}},
           'state': {a: m.persistent_bytes() for a, m in models.items()},
           'state_file_bytes_E': sp.stat().st_size}
    write_json(out_dir / f'BENCH_{dataset}{sfx}.json', res)
    log('BENCH DONE', dataset, 'train %.0fs, infer p50 %.1f ms/img' % (train_wall, res['infer_batch1']['total_E']['p50_ms']))
    return res
